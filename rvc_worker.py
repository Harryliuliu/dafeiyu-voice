"""唱歌服务的芙宁娜 RVC 转换工作进程（Applio 3.7.1 + 社区芙宁娜 48k 模型）。

唱歌服务（seed-vc 的 venv）用子进程拉起它，一行一个 JSON 请求/回应，走标准输入输出：
  {"input": 主唱.wav, "output": 结果.wav, "semitone": -3}  → {"ok": true, "sr": 48000}
输出是模型原始采样率（48 kHz），响度统一、淡入淡出由唱歌服务做。标准输入关掉就退出。
参数是 2026-11-10 用户试听过的：rmvpe 测音高、index_rate 0.5、protect 0.33、保留原唱的强弱。

必须用 Applio 的 venv、在 Applio 目录里运行（它按相对路径找 rvc/models/）：
  cd ~/dafeiyu-tools/rvc-test/Applio && .venv/bin/python -I ~/dafeiyu-tools/rvc_worker.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

TOOLS = Path.home() / "dafeiyu-tools"
APPLIO = TOOLS / "rvc-test" / "Applio"
MODEL = TOOLS / "rvc-test" / "model"
PTH = MODEL / "芙宁娜.pth"
INDEX = MODEL / "added_IVF256_Flat_nprobe_1_default_exp_name_v2.index"
SETTINGS = dict(index_rate=0.5, volume_envelope=1.0, protect=0.33, f0_method="rmvpe")


# 回应走一个单独复制出来的标准输出；真正的标准输出改指到标准错误，
# 模型库自己 print 的进度（包括 C 扩展里的）就不会混进回应里
PROTO = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)


def reply(obj):
    PROTO.write(json.dumps(obj, ensure_ascii=False) + "\n")
    PROTO.flush()


def log(msg):
    print(f"[RVC进程] {msg}", file=sys.stderr, flush=True)


def main():
    os.chdir(APPLIO)
    sys.path.insert(0, str(APPLIO))
    import soundfile as sf
    from core import _base_infer_kwargs
    from rvc.infer.infer import VoiceConverter

    t = time.time()
    vc = VoiceConverter()  # 同一个实例反复用：模型和 contentvec 只加载一次

    def convert(src, dst, semitone):
        kw = _base_infer_kwargs(pitch=int(semitone), pth_path=str(PTH), index_path=str(INDEX),
                                export_format="WAV", embedder_model="contentvec", **SETTINGS)
        kw["audio_input_path"], kw["audio_output_path"] = src, dst
        vc.convert_audio(**kw)
        return sf.info(dst).samplerate

    # 先转一小段静音热身，把模型加载好
    warm = Path(os.environ.get("TMPDIR", "/tmp")) / f"dafeiyu_rvc_warm_{os.getpid()}.wav"
    import numpy as np
    sf.write(str(warm), np.zeros(16000, dtype=np.float32), 16000)
    try:
        convert(str(warm), str(warm.with_suffix(".out.wav")), 0)
    except Exception as e:  # noqa: BLE001  热身失败不要紧，第一首歌时再加载
        log(f"热身没成功（不影响使用）：{type(e).__name__}: {e}")
    finally:
        for f in (warm, warm.with_suffix(".out.wav")):
            f.unlink(missing_ok=True)
    log(f"模型加载完成，用时 {time.time() - t:.0f} 秒")
    reply({"ready": True})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            t = time.time()
            sr = convert(req["input"], req["output"], req.get("semitone", 0))
            resp = {"ok": True, "sr": sr, "seconds": round(time.time() - t, 1)}
        except Exception as e:  # noqa: BLE001
            log(f"出错：{type(e).__name__}: {e}")
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        reply(resp)


if __name__ == "__main__":
    main()
