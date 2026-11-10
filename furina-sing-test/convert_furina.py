"""芙宁娜唱歌试验（Seed-VC）：同样三段歌，用芙宁娜参考 A/B 各转一遍，和现在的音色放一起试听。

不碰正在用的唱歌服务；分离出的人声存到 人声/，RVC 试验也用同一份，保证对比公平。
用 seed-vc 的 venv 跑：
  PYTORCH_ENABLE_MPS_FALLBACK=1 ~/dafeiyu-tools/seed-vc/.venv/bin/python -I ~/dafeiyu-tools/furina-sing-test/convert_furina.py
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

HERE = TOOLS / "furina-sing-test"
OUT = HERE / "对比"
VOC = HERE / "人声"
LIB = TOOLS / "歌曲库"
CACHE = LIB / "缓存"
VOICES = TOOLS / "voices"
REFS = [p for p in sorted((HERE / "参考").glob("芙宁娜_*.wav"))]
OLD_REFS = [("原大肥鱼", VOICES / "歌声参考_23秒.wav", ""), ("雨爱", VOICES / "歌声参考_雨爱_23秒.wav", "_eb3436")]
# (名字, 文件, 来源, 起点, 升降调)；升降调和正式服务的判断一致：
# 唯一 285 Hz、See You Again 白名单 → 0；泡沫约 410 Hz 落在 370–430 → 降 3
SONGS = [
    ("1_唯一", LIB / "原曲" / "邓紫棋-唯一.m4a", "原曲", 15.0, 0,
     "邓紫棋-唯一@15.0+30_v1r25s40{tag}.wav"),
    ("2_See You Again", LIB / "清唱" / "Charlie Puth-See You Again.wav", "清唱", 0.0, 0,
     "Charlie Puth-See You Again@auto+30_v1r25s40{tag}.wav"),
    ("3_泡沫", LIB / "原曲" / "邓紫棋-泡沫.m4a", "原曲", 157.0, -3, None),
]
DURATION, STEPS = 30.0, 40


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VOC.mkdir(parents=True, exist_ok=True)
    report_path = HERE / "report.json"
    report = {"refs": {}, "songs": {}}

    def save(name, audio, secs, engine, semitone):
        out = finish(audio)
        dst = OUT / name
        sf.write(str(dst), out, SR, subtype="PCM_16")
        p = engine.pitch_stats(out)
        report["songs"][name] = {"seconds": round(secs), "semitone": semitone,
                                 "median_hz": round(p["median_hz"]), "finite": bool(np.isfinite(out).all())}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {name}  {secs:.0f} 秒", flush=True)

    t = time.time()
    engine = SingEngine(str(REFS[0]), diffusion_steps=STEPS)
    print(f"模型加载 {time.time() - t:.0f} 秒", flush=True)

    sources = {}
    for name, path, kind, start, semitone, cached in SONGS:
        cache_file = VOC / f"{name}.wav"
        if cache_file.exists():
            vocals = sf.read(str(cache_file), dtype="float32")[0]
        elif kind == "清唱":
            vocals = decode(str(path), start=start, duration=DURATION, channels=1)[:, 0]
        else:
            t = time.time()
            vocals = engine.separate(decode(str(path), start=start, duration=DURATION))
            print(f"{name} 分离人声 {time.time() - t:.0f} 秒", flush=True)
        sf.write(str(cache_file), vocals, SR, subtype="FLOAT")
        sources[name] = vocals
        p = engine.pitch_stats(vocals)
        report["songs"][f"{name}_原唱人声"] = {"median_hz": round(p["median_hz"]), "p90_hz": round(p["p90_hz"])}
        for label, _, tag in OLD_REFS:  # 现在的两种音色有缓存的直接拿来对照
            if cached and (CACHE / cached.format(tag=tag)).exists():
                shutil.copy2(CACHE / cached.format(tag=tag), OUT / f"{name}_0_{label}.wav")

    for ref in REFS:
        engine._prepare_reference(str(ref))
        tag = ref.stem.split("_")[1]  # A / B
        report["refs"][ref.stem] = {"median_hz": round(engine.ref_median_hz)}
        print(f"参考 {ref.stem}：中位音高 {engine.ref_median_hz:.0f} Hz", flush=True)
        for name, _, _, _, semitone, _ in SONGS:
            t = time.time()
            save(f"{name}_芙宁娜{tag}_SeedVC.wav", engine.convert(sources[name], semitone),
                 time.time() - t, engine, semitone)

    # 泡沫没有缓存：用现在的两种音色也各转一份当对照
    for label, ref, _ in OLD_REFS:
        engine._prepare_reference(str(ref))
        report["refs"][label] = {"median_hz": round(engine.ref_median_hz)}
        name, *_, semitone, _ = SONGS[2]
        t = time.time()
        save(f"{name}_0_{label}.wav", engine.convert(sources[name], semitone), time.time() - t, engine, semitone)
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
