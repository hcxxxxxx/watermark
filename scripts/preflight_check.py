#!/usr/bin/env python
"""preflight_check.py — 在花 GPU 时间之前验证环境

检查：依赖包、CUDA、数据集、checkpoint、ffmpeg、仓库模块能否导入、
      两个水印器能否构造且确实是不同的类。

运行：
    python scripts/preflight_check.py
退出码 0 = 全部通过；非 0 = 有 FAIL 项。
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    icon = {PASS: "✓", WARN: "!", FAIL: "✗"}[status]
    print(f"  {icon} [{status}] {name}" + (f" — {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    s = PASS if (v.major, v.minor) >= (3, 10) else WARN
    record(s, "Python 版本", f"{v.major}.{v.minor}.{v.micro}（需要 >= 3.10）")


def check_packages() -> None:
    required = ["torch", "torchaudio", "numpy", "scipy", "yaml", "soundfile", "pesq", "pystoi"]
    optional = ["pandas", "matplotlib", "encodec"]
    for pkg in required:
        try:
            m = importlib.import_module(pkg)
            record(PASS, f"依赖 {pkg}", getattr(m, "__version__", ""))
        except ImportError as e:
            record(FAIL, f"依赖 {pkg}", str(e))
    for pkg in optional:
        try:
            m = importlib.import_module(pkg)
            record(PASS, f"可选 {pkg}", getattr(m, "__version__", ""))
        except ImportError:
            record(WARN, f"可选 {pkg}", "未安装（部分实验需要）")


def check_cuda() -> None:
    try:
        import torch
    except ImportError:
        record(FAIL, "CUDA", "torch 未安装")
        return
    if not torch.cuda.is_available():
        record(WARN, "CUDA", "不可用 —— 将回落到 CPU，速度慢 20-50x")
        return
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    record(PASS, "CUDA", f"{n} 张卡，{names[0]}，{mem:.1f} GB")
    if mem < 10:
        record(WARN, "显存", f"{mem:.1f} GB —— DiffWave 可能不足，建议 >= 12 GB")


def check_dataset() -> None:
    root = ROOT / "data" / "LJSpeech-1.1"
    meta, wavs = root / "metadata.csv", root / "wavs"
    if not root.is_dir():
        record(FAIL, "LJSpeech 数据集", f"缺失：{root}")
        return
    if not meta.is_file():
        record(FAIL, "LJSpeech metadata.csv", f"缺失：{meta}")
        return
    n = len(list(wavs.glob("*.wav"))) if wavs.is_dir() else 0
    s = PASS if n >= 13000 else WARN
    record(s, "LJSpeech 数据集", f"{n} 条 wav（完整应为 13100）")


def check_checkpoints() -> None:
    hg_gen = ROOT / "checkpoints" / "hifigan" / "generator_v1"
    hg_cfg = ROOT / "checkpoints" / "hifigan" / "config.json"
    if hg_gen.is_file() and hg_cfg.is_file():
        record(PASS, "HiFi-GAN checkpoint", f"{hg_gen.stat().st_size / 1024**2:.0f} MB")
    else:
        missing = [str(p.relative_to(ROOT)) for p in (hg_gen, hg_cfg) if not p.is_file()]
        record(FAIL, "HiFi-GAN checkpoint", "缺失：" + ", ".join(missing))

    dw = list((ROOT / "checkpoints" / "diffwave").glob("*.pt")) if (
        ROOT / "checkpoints" / "diffwave"
    ).is_dir() else []
    dw += list((ROOT / "external" / "diffwave" / "checkpoints").glob("*.pt")) if (
        ROOT / "external" / "diffwave" / "checkpoints"
    ).is_dir() else []
    if dw:
        record(PASS, "DiffWave checkpoint", dw[0].name)
    else:
        record(WARN, "DiffWave checkpoint", "缺失 —— DiffWave 实验将跳过")


def check_ffmpeg() -> None:
    exe = shutil.which("ffmpeg")
    if not exe:
        record(FAIL, "ffmpeg", "未找到 —— MP3/AAC/speed/pitch 攻击不可用")
        return
    try:
        out = subprocess.run(
            [exe, "-version"], capture_output=True, text=True, timeout=10
        ).stdout.splitlines()[0]
        record(PASS, "ffmpeg", out[:60])
    except Exception as e:
        record(WARN, "ffmpeg", f"存在但无法执行：{e}")


def check_repo_modules() -> None:
    mods = [
        "melshield.relmel",
        "melshield.watermark",
        "melshield.attacks",
        "melshield.mel",
        "melshield.metrics",
        "melshield.coding",
        "melshield.datasets",
        "melshield.vocoders",
    ]
    for m in mods:
        try:
            importlib.import_module(m)
            record(PASS, f"模块 {m}")
        except Exception as e:
            record(FAIL, f"模块 {m}", str(e))


def check_markers() -> None:
    """最关键的一项：确认 RAWMER 和 MelShield 是不同的实现。"""
    try:
        from scripts.exp_common import build_marker

        r, rt, _ = build_marker("rawmer", 0.435)
        m, mt, _ = build_marker("melshield", 0.050)
        rn, mn = type(r).__name__, type(m).__name__
        if rn == mn:
            record(FAIL, "水印器区分", f"两者都是 {rn} —— MelShield 基线是错的！")
        else:
            record(PASS, "水印器区分", f"RAWMER={rn}(τ={rt}), MelShield={mn}(τ={mt})")
    except Exception as e:
        record(FAIL, "水印器构造", str(e))


def check_new_attacks() -> None:
    try:
        from melshield.mel import MelConfig
        from scripts.new_attacks import band_targeted_noise, describe_band

        cfg = MelConfig(
            sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024,
            n_mels=80, f_min=0.0, f_max=8000.0, power=1.0, eps=1e-5,
        )
        record(PASS, "band 噪声频段", describe_band((20, 60), cfg))

        import torch

        fn = band_targeted_noise(10.0, band=(20, 60), mel_config=cfg)
        out = fn(torch.randn(1, 22050) * 0.1, 22050)
        record(PASS, "band 噪声可执行", f"输出 {tuple(out.waveform.shape)}")
    except Exception as e:
        record(FAIL, "新攻击模块", str(e))


def main() -> int:
    print("=" * 70)
    print("  RAWMER 实验环境飞行前检查")
    print("=" * 70)

    for title, fn in [
        ("Python 与依赖", lambda: (check_python(), check_packages())),
        ("GPU", check_cuda),
        ("数据与 checkpoint", lambda: (check_dataset(), check_checkpoints())),
        ("外部工具", check_ffmpeg),
        ("仓库模块", check_repo_modules),
        ("水印器与新攻击", lambda: (check_markers(), check_new_attacks())),
    ]:
        print(f"\n--- {title} ---")
        fn()

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print("\n" + "=" * 70)
    print(f"  结果：{len(results) - n_fail - n_warn} 通过 / {n_warn} 警告 / {n_fail} 失败")
    print("=" * 70)

    if n_fail:
        print("\n必须先修复以下 FAIL 项：")
        for s, name, detail in results:
            if s == FAIL:
                print(f"  ✗ {name}: {detail}")
        return 1
    if n_warn:
        print("\n以下 WARN 项不阻塞，但会影响部分实验：")
        for s, name, detail in results:
            if s == WARN:
                print(f"  ! {name}: {detail}")
    print("\n可以运行：bash scripts/run_review_experiments.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
