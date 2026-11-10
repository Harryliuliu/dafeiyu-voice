"""RVC 芙宁娜批量转换：每个参数写成 输入.wav|升降调|输出.wav。输出做和正式服务一样的响度统一、淡入淡出。

  cd ~/dafeiyu-tools/rvc-test/Applio && PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -I \
     ~/dafeiyu-tools/furina-sing-test/rvc_batch.py "in.wav|0|out.wav" ...
"""

import sys
import time
from pathlib import Path

TOOLS = Path.home() / "dafeiyu-tools"
sys.path.insert(0, str(TOOLS / "rvc-test" / "Applio"))
sys.path.append(str(TOOLS))

import soundfile as sf  # noqa: E402
import soxr  # noqa: E402
from core import run_infer_script  # noqa: E402
from sing_engine import SR, finish  # noqa: E402

MODEL = TOOLS / "rvc-test" / "model"
SETTINGS = dict(index_rate=0.5, volume_envelope=1.0, protect=0.33, f0_method="rmvpe")


def main():
    tmp = TOOLS / "furina-sing-test" / "rvc_raw"
    tmp.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[1:]:
        src, semitone, dst = arg.split("|")
        raw = tmp / Path(dst).name
        t = time.time()
        run_infer_script(
            pitch=int(semitone), input_path=src, output_path=str(raw), pth_path=str(MODEL / "芙宁娜.pth"),
            index_path=str(MODEL / "added_IVF256_Flat_nprobe_1_default_exp_name_v2.index"),
            f0_autotune=False, f0_autotune_strength=1.0, proposed_pitch=False, proposed_pitch_threshold=155.0,
            export_format="WAV", embedder_model="contentvec", **SETTINGS)
        audio, sr = sf.read(str(raw), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = soxr.resample(audio, sr, SR)
        sf.write(dst, finish(audio), SR, subtype="PCM_16")
        print(f"  {Path(dst).name}  {time.time() - t:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
