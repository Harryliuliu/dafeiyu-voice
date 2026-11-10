"""Seed-VC 批量转换：--ref 参考歌声，之后每个参数写成 输入.wav|升降调|输出.wav（和正式服务同样的 40 步、后处理）。

  PYTORCH_ENABLE_MPS_FALLBACK=1 ~/dafeiyu-tools/seed-vc/.venv/bin/python -I \
     ~/dafeiyu-tools/furina-sing-test/seedvc_batch.py --ref ~/dafeiyu-tools/voices/歌声参考_23秒.wav "in.wav|0|out.wav" ...
"""

import sys
import time
from pathlib import Path

TOOLS = Path.home() / "dafeiyu-tools"
sys.path.insert(0, str(TOOLS))

import soundfile as sf  # noqa: E402
from sing_engine import SR, SingEngine, finish  # noqa: E402


def main():
    args = sys.argv[1:]
    ref = args[args.index("--ref") + 1]
    jobs = [a for a in args if "|" in a]
    t = time.time()
    engine = SingEngine(ref, diffusion_steps=40)
    print(f"模型加载 {time.time() - t:.0f} 秒，参考中位音高 {engine.ref_median_hz:.0f} Hz", flush=True)
    for job in jobs:
        src, semitone, dst = job.split("|")
        vocals = sf.read(src, dtype="float32")[0]
        if vocals.ndim == 2:
            vocals = vocals.mean(axis=1)
        t = time.time()
        sf.write(dst, finish(engine.convert(vocals, int(semitone))), SR, subtype="PCM_16")
        print(f"  {Path(dst).name}  {time.time() - t:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
