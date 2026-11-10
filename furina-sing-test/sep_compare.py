"""人声分离对比：同样的 30 秒原曲，用 Roformer 系模型分三档，和现在的 htdemucs 对比。

  2_人声模型    ：Kim 版 Mel-Band Roformer 分出全部人声（含和声）
  3_去和声      ：2 的人声再过 karaoke 模型（frazer & becruily BS-Roformer），只留主唱
  4_去和声去混响：3 再过 anvuew 的 BS-Roformer 去混响
（MVSEP 的经验：先分干净人声再去和声，比直接对整首歌去和声好）

用 sep-test 的 venv 跑：
  ~/dafeiyu-tools/sep-test/.venv/bin/python -I ~/dafeiyu-tools/furina-sing-test/sep_compare.py
"""

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

TOOLS = Path.home() / "dafeiyu-tools"
HERE = TOOLS / "furina-sing-test"
MODELS = TOOLS / "sep-test" / "models"
WORK = HERE / "分离对比_raw"
OUT = HERE / "分离对比"
LIB = TOOLS / "歌曲库" / "原曲"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
SR = 44100
SONGS = [("1_唯一", LIB / "邓紫棋-唯一.m4a", 15.0), ("3_泡沫", LIB / "邓紫棋-泡沫.m4a", 157.0)]
STEPS = [  # (模型文件, 要留的那一轨, 去掉的那一轨)
    ("vocals_mel_band_roformer.ckpt", "vocals", "other"),
    ("bs_roformer_karaoke_frazer_becruily.ckpt", "vocals", "instrumental"),
    ("deverb_bs_roformer_8_384dim_10depth.ckpt", "noreverb", "reverb"),
]
NAMES = ["2_人声模型", "3_去和声", "4_去和声去混响"]


def stem_file(files, stem):
    """audio-separator 的输出文件名形如 xxx_(Vocals)_model.wav，按括号里的轨名找（不分大小写）。"""
    for f in files:
        m = re.findall(r"_\(([^)]+)\)_", Path(f).name)  # 多次分离后名字里有好几层括号，最后一层才是这次的
        if m and m[-1].lower() == stem:
            return WORK / Path(f).name
    raise RuntimeError(f"没找到 {stem} 轨：{files}")


def to_mono(path):
    a, sr = sf.read(str(path), dtype="float32")
    assert sr == SR, sr
    return a.mean(axis=1) if a.ndim == 2 else a


def main():
    from audio_separator.separator import Separator

    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sep = Separator(model_file_dir=str(MODELS), output_dir=str(WORK), output_format="WAV", sample_rate=SR)
    report = {}
    current = {}
    for name, path, start in SONGS:  # 先截出 30 秒双声道原曲
        clip = WORK / f"{name}_原曲30秒.wav"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", str(start), "-t", "30", "-i", str(path),
                        "-ac", "2", "-ar", str(SR), str(clip)], check=True)
        current[name] = clip
    for (model, keep, drop), label in zip(STEPS, NAMES):  # 一个模型只加载一次，两首歌依次过
        t = time.time()
        sep.load_model(model_filename=model)
        print(f"加载 {model} {time.time() - t:.0f} 秒", flush=True)
        for name, *_ in SONGS:
            t = time.time()
            files = sep.separate(str(current[name]))
            secs = time.time() - t
            kept, dropped = stem_file(files, keep), stem_file(files, drop)
            sf.write(str(OUT / f"{name}_{label}.wav"), to_mono(kept), SR, subtype="FLOAT")
            if label == "3_去和声":
                sf.write(str(OUT / f"{name}_附_被去掉的和声.wav"), to_mono(dropped), SR, subtype="FLOAT")
            current[name] = kept
            report[f"{name}_{label}"] = {"seconds": round(secs, 1)}
            print(f"  {name} {label} {secs:.0f} 秒", flush=True)
            (HERE / "report_sep.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # 现在的 htdemucs 结果（第一轮已分好）也放一份，名字对齐
    for name, *_ in SONGS:
        shutil.copy2(HERE / "人声" / f"{name}.wav", OUT / f"{name}_1_现在demucs.wav")
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
