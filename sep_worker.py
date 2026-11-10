"""唱歌服务的人声分离工作进程（Roformer 系模型，audio-separator）。

唱歌服务（seed-vc 的 venv）用子进程拉起它，一行一个 JSON 请求/回应，走标准输入输出：
  {"op": "vocals", "input": 双声道.wav, "output": 人声.wav}        → 分出全部人声（含和声）
  {"op": "lead",   "input": 人声.wav,   "output": 主唱.wav, "backing": 和声.wav} → 去和声，只留主唱
回应 {"ok": true} 或 {"ok": false, "error": "..."}。标准输入关掉（唱歌服务退出）就退出。
模型在启动时各加载一次（两个 Separator 各管一个），日志写标准错误。
2026-11-10 用户试听：先分人声再去和声，泡沫的电音没了；去混响区别不大，不用。

必须用 sep-test 的 venv 运行：
  ~/dafeiyu-tools/sep-test/.venv/bin/python -I ~/dafeiyu-tools/sep_worker.py
"""

import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

TOOLS = Path.home() / "dafeiyu-tools"
MODELS = TOOLS / "sep-test" / "models"
VOCALS_MODEL = "vocals_mel_band_roformer.ckpt"  # Kim 版 Mel-Band Roformer
LEAD_MODEL = "bs_roformer_karaoke_frazer_becruily.ckpt"  # frazer & becruily 的 karaoke 模型
SR = 44100


# 回应走一个单独复制出来的标准输出；真正的标准输出改指到标准错误，
# 模型库自己 print 的进度（包括 C 扩展里的）就不会混进回应里
PROTO = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)


def reply(obj):
    PROTO.write(json.dumps(obj, ensure_ascii=False) + "\n")
    PROTO.flush()


def log(msg):
    print(f"[分离进程] {msg}", file=sys.stderr, flush=True)


def stem_file(files, stem, workdir):
    """输出文件名形如 xxx_(Vocals)_model.wav，取最后一层括号里的轨名（不分大小写）。"""
    for f in files:
        names = re.findall(r"_\(([^)]+)\)_", Path(f).name)
        if names and names[-1].lower() == stem:
            return Path(workdir) / Path(f).name
    raise RuntimeError(f"没找到 {stem} 轨：{files}")


def main():
    from audio_separator.separator import Separator

    work = Path(tempfile.mkdtemp(prefix="dafeiyu_sep_"))
    t = time.time()
    seps = {}
    for key, model in (("vocals", VOCALS_MODEL), ("lead", LEAD_MODEL)):
        s = Separator(log_level=30, model_file_dir=str(MODELS), output_dir=str(work), output_format="WAV",
                      sample_rate=SR)
        s.load_model(model_filename=model)
        seps[key] = s
    log(f"模型加载完成，用时 {time.time() - t:.0f} 秒")
    reply({"ready": True})

    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                op = req["op"]
                t = time.time()
                files = seps[op].separate(req["input"])
                if op == "vocals":
                    shutil.move(str(stem_file(files, "vocals", work)), req["output"])
                    drop = stem_file(files, "other", work)
                else:
                    shutil.move(str(stem_file(files, "vocals", work)), req["output"])
                    drop = stem_file(files, "instrumental", work)  # karaoke 模型把和声放在这一轨
                    if req.get("backing"):
                        shutil.move(str(drop), req["backing"])
                if drop.exists():
                    drop.unlink()
                resp = {"ok": True, "seconds": round(time.time() - t, 1)}
            except Exception as e:  # noqa: BLE001
                log(f"出错：{type(e).__name__}: {e}")
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            reply(resp)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
