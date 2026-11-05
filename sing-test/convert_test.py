"""新音色唱歌对比：同样的两段测试歌，换不同参考人声各转一遍，和原音色的缓存版放在一起试听。

用 seed-vc 的 venv 跑（不碰正在用的唱歌服务）：
  ~/dafeiyu-tools/seed-vc/.venv/bin/python -I ~/dafeiyu-tools/sing-test/convert_test.py
"""

import json
import shutil
import sys
import time
from pathlib import Path

TOOLS = Path.home() / "dafeiyu-tools"
sys.path.insert(0, str(TOOLS))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from sing_engine import SR, SingEngine, decode, finish  # noqa: E402

HERE = TOOLS / "sing-test"
OUT = HERE / "对比"
LIB = TOOLS / "歌曲库"
REFS = sorted((HERE / "参考").glob("*.wav"))
# 和原音色缓存里的片段一致（缓存 json 里记的起点、时长、升降调）
SONGS = [
    ("See You Again", LIB / "清唱" / "Charlie Puth-See You Again.wav", "清唱", 0.0,
     LIB / "缓存" / "Charlie Puth-See You Again@auto+30_v1r25s40.wav"),
    ("唯一", LIB / "原曲" / "邓紫棋-唯一.m4a", "原曲", 15.0,
     LIB / "缓存" / "邓紫棋-唯一@15.0+30_v1r25s40.wav"),
]
DURATION, SEMITONE, STEPS = 30.0, 0, 40


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    engine = SingEngine(str(REFS[0]), diffusion_steps=STEPS)
    print(f"模型加载 {time.time() - t:.0f} 秒", flush=True)

    sources = {}
    for name, path, kind, start, old in SONGS:
        if kind == "清唱":
            vocals = decode(str(path), start=start, duration=DURATION, channels=1)[:, 0]
        else:
            t = time.time()
            vocals = engine.separate(decode(str(path), start=start, duration=DURATION))
            print(f"{name} 分离人声 {time.time() - t:.0f} 秒", flush=True)
        sources[name] = vocals
        shutil.copy2(old, OUT / f"{name}_0_原音色.wav")

    report = {"refs": {}, "songs": {}}
    for ref in REFS:
        engine._prepare_reference(str(ref))
        tag = ref.stem
        report["refs"][tag] = {"median_hz": round(engine.ref_median_hz)}
        print(f"参考 {tag}：中位音高 {engine.ref_median_hz:.0f} Hz", flush=True)
        for name, *_ in SONGS:
            t = time.time()
            out = finish(engine.convert(sources[name], SEMITONE))
            secs = time.time() - t
            dst = OUT / f"{name}_{tag}.wav"
            sf.write(str(dst), out, SR, subtype="PCM_16")
            report["songs"][dst.name] = {"seconds": round(secs), "finite": bool(np.isfinite(out).all())}
            print(f"  {dst.name}  {secs:.0f} 秒", flush=True)
            (HERE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
