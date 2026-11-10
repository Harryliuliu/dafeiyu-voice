"""芙宁娜唱歌试验（RVC）：用社区的芙宁娜 RVC 模型转同样的三段人声，和 Seed-VC 结果放一起试听。

人声用 convert_furina.py 分离好的那份（人声/），升降调和 Seed-VC 一样；输出也做同样的响度统一和淡入淡出。
用 Applio 的 venv、在 Applio 目录里跑（它按相对路径找 rvc/models/）：
  cd ~/dafeiyu-tools/rvc-test/Applio && PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -I ~/dafeiyu-tools/furina-sing-test/convert_rvc.py
"""

import json
import sys
import time
from pathlib import Path

TOOLS = Path.home() / "dafeiyu-tools"
APPLIO = TOOLS / "rvc-test" / "Applio"
sys.path.insert(0, str(APPLIO))
sys.path.append(str(TOOLS))  # 只为拿 sing_engine.finish（它顶层只导入 numpy）

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402
from core import run_infer_script  # noqa: E402
from sing_engine import SR, finish  # noqa: E402

HERE = TOOLS / "furina-sing-test"
VOC = HERE / "人声"
OUT = HERE / "对比"
TMP = HERE / "rvc_raw"
MODEL = TOOLS / "rvc-test" / "model"
PTH = MODEL / "芙宁娜.pth"
INDEX = MODEL / "added_IVF256_Flat_nprobe_1_default_exp_name_v2.index"
SONGS = [("1_唯一", 0), ("2_See You Again", 0), ("3_泡沫", -3)]
# index_rate：越高越贴近训练时的音色，太高容易有杂音；protect 保护清辅音和呼吸声
SETTINGS = dict(index_rate=0.5, volume_envelope=1.0, protect=0.33, f0_method="rmvpe")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    report_path = HERE / "report_rvc.json"
    report = {"settings": SETTINGS, "songs": {}}
    for name, semitone in SONGS:
        src = VOC / f"{name}.wav"
        raw = TMP / f"{name}.wav"
        t = time.time()
        run_infer_script(
            pitch=semitone, input_path=str(src), output_path=str(raw), pth_path=str(PTH), index_path=str(INDEX),
            f0_autotune=False, f0_autotune_strength=1.0, proposed_pitch=False, proposed_pitch_threshold=155.0,
            export_format="WAV", embedder_model="contentvec", **SETTINGS)
        secs = time.time() - t
        audio, sr = sf.read(str(raw), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = soxr.resample(audio, sr, SR)
        out = finish(audio)
        dst = OUT / f"{name}_芙宁娜_RVC.wav"
        sf.write(str(dst), out, SR, subtype="PCM_16")
        report["songs"][dst.name] = {"seconds": round(secs, 1), "semitone": semitone, "model_sr": sr,
                                     "finite": bool(np.isfinite(out).all())}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {dst.name}  {secs:.0f} 秒", flush=True)
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
