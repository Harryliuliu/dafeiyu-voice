"""Breeze TTS 2 (MLX) 与 IndexTTS2 同台词 A/B 对比。

用法（都用 breeze-test 的 venv 跑）：
  python -I ab_test.py index    # 调 127.0.0.1:9880 的 IndexTTS2 服务
  python -I ab_test.py breeze   # 本进程加载 Breeze TTS 2
  python -I ab_test.py table    # 汇总成 对比/对比表.md

峰值内存：每句生成期间每 0.1 秒采样一次进程常驻内存（RSS），取最大值。
IndexTTS2 采的是 tts_server 进程，Breeze 采的是本进程；Breeze 另记 MLX 自报的峰值。
"""

import ctypes
import importlib.util
import json
import struct
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "对比"
RAW = OUT / "原始输出_未均衡"
RESULTS = HERE / "results.json"
MATERIAL = Path.home() / "Documents" / "Claude Code" / "大肥鱼音色素材"
REF_AUDIO = MATERIAL / "参考片段1.wav"
REF_TEXT = "第一次见你的我，好慌张，找不到什么话要讲"  # 用户提供的逐字稿（片段1 是唱的）
BREEZE_DIR = HERE / "models" / "Breeze-TTS-2-mlx"
TTS_PID = Path.home() / "dafeiyu-tools" / "run" / "tts_server.pid"
SEED = 42
CFG = 4.0

# 参考片段是唱歌，所以每条指令都先要求“正常说话、别唱”
KEEP = "保持参考音频的音色，用自然说话的方式（不要唱），"

# (序号, 情绪, 台词, Breeze 前置标签, Breeze 语气指令)
LINES = [
    ("1", "开心", "太好啦！你终于来群里陪我聊天了，我今天超级开心的！", "[笑] ",
     "开心雀跃的语气，声音明亮，语速稍快，带着笑意。"),
    ("2", "傲娇", "哼，谁、谁想你了啊？我只是刚好有空，才勉强回你一句而已。", "",
     "嘴硬心软的傲娇语气：表面不耐烦、有点别扭，其实心里很高兴，尾音略带撒娇。"),
    ("3", "委屈", "我明明等了你一整天，你却连一条消息都不回我……", "[叹气] ",
     "委屈的语气，声音低落、带一点哭腔，像忍着眼泪小声抱怨。"),
    ("4", "害羞", "你、你别一直盯着我看嘛，人家会不好意思的。", "",
     "害羞的语气，声音轻柔、有点迟疑和结巴，像是脸红了不好意思。"),
    ("5", "惊讶", "什么？你说这条鱼已经养了五年了？真的假的！", "",
     "非常惊讶的语气，语调明显上扬，带着难以置信的感觉。"),
    ("6", "平静", "今天天气还不错，晚上早点休息，明天见。", "",
     "平静温和的语气，语速平稳，放松自然。"),
]


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def rss_mb(pid: int) -> float:
    """进程的物理占用（phys_footprint，含 MLX 放在统一内存里的显存），单位 MB。

    ps 的 RSS 不含 Metal 缓冲区，会把 MLX 模型的内存漏掉。"""
    buf = ctypes.create_string_buffer(512)
    if _libc.proc_pid_rusage(pid, 2, buf) != 0:  # RUSAGE_INFO_V2
        return 0.0
    # rusage_info_v2：16 字节 uuid 后，第 8 个 uint64 是 ri_phys_footprint
    return struct.unpack_from("<8Q", buf, 16)[7] / 2**20


class PeakSampler:
    def __init__(self, pid: int):
        self.pid, self.peak, self._stop = pid, 0.0, threading.Event()

    def __enter__(self):
        def run():
            while not self._stop.is_set():
                self.peak = max(self.peak, rss_mb(self.pid))
                time.sleep(0.1)
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, rss_mb(self.pid))


def load_results() -> dict:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def save_result(key: str, row: dict):
    data = load_results()
    data[key] = row
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def wav_seconds(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


def run_index():
    pid = int(TTS_PID.read_text().strip())
    health = json.loads(urllib.request.urlopen("http://127.0.0.1:9880/health", timeout=5).read())
    print("IndexTTS2 服务：", health)
    for no, emo, text, _tag, _ins in LINES:
        body = json.dumps({"text": text, "emotion": emo, "seed": SEED}).encode()
        req = urllib.request.Request("http://127.0.0.1:9880/tts", data=body,
                                     headers={"Content-Type": "application/json"})
        dst = OUT / f"{no}_{emo}_IndexTTS2.wav"
        with PeakSampler(pid) as ps:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=600) as r:
                dst.write_bytes(r.read())
                hdr = {k: r.headers.get(k) for k in ("X-Cache", "X-Gen-Seconds")}
            secs = time.perf_counter() - t0
        dur = wav_seconds(dst)
        row = {"model": "IndexTTS2", "emotion": emo, "file": dst.name, "text": text,
               "seconds": round(secs, 2), "audio_seconds": round(dur, 2),
               "peak_rss_mb": round(ps.peak), "cache": hdr.get("X-Cache"),
               "server_gen_seconds": hdr.get("X-Gen-Seconds")}
        save_result(f"index_{no}", row)
        print(row, flush=True)


def run_breeze():
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from mlx_audio.tts import load

    spec = importlib.util.spec_from_file_location(
        "loudness", Path.home() / "dafeiyu-tools" / "loudness.py")
    loudness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loudness)

    me = __import__("os").getpid()
    t0 = time.perf_counter()
    model = load(str(BREEZE_DIR))
    load_s = time.perf_counter() - t0
    print(f"Breeze 加载 {load_s:.1f} 秒，RSS {rss_mb(me):.0f} MB", flush=True)

    def synth(text, instruct):
        chunks = []
        for r in model.generate(text=text, instruct=instruct, ref_audio=str(REF_AUDIO),
                                ref_text=REF_TEXT, cfg_scale=CFG, seed=SEED):
            chunks.append(np.array(r.audio, dtype=np.float32))
        return np.concatenate(chunks), model.sample_rate

    # 预热一次（IndexTTS2 服务是热的，这样比较公平），不计入结果
    t0 = time.perf_counter()
    synth("你好。", KEEP + "平静的语气。")
    print(f"预热 {time.perf_counter() - t0:.1f} 秒", flush=True)

    for no, emo, text, tag, ins in LINES:
        mx.reset_peak_memory()
        with PeakSampler(me) as ps:
            t0 = time.perf_counter()
            audio, sr = synth(tag + text, KEEP + ins)
            secs = time.perf_counter() - t0
        RAW.mkdir(parents=True, exist_ok=True)
        sf.write(str(RAW / f"{no}_{emo}_Breeze.wav"), audio, sr)
        dst = OUT / f"{no}_{emo}_Breeze.wav"
        sf.write(str(dst), loudness.level_loudness(audio, sr), sr)
        row = {"model": "Breeze TTS 2", "emotion": emo, "file": dst.name,
               "text": tag + text, "instruct": KEEP + ins,
               "seconds": round(secs, 2), "audio_seconds": round(len(audio) / sr, 2),
               "peak_rss_mb": round(ps.peak), "mlx_peak_mb": round(mx.get_peak_memory() / 2**20),
               "load_seconds": round(load_s, 1)}
        save_result(f"breeze_{no}", row)
        print(row, flush=True)


def make_table():
    data = load_results()
    rev = (HERE / "models" / "Breeze-TTS-2-mlx.revision.txt").read_text().split()[-1]
    import importlib.metadata as md
    lines = [
        "# Breeze TTS 2 vs IndexTTS2 同台词对比", "",
        "| # | 情绪 | 模型 | 文件 | 生成耗时(秒) | 音频时长(秒) | 耗时/时长 | 峰值内存(MB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for no, emo, *_ in LINES:
        for key in (f"index_{no}", f"breeze_{no}"):
            r = data.get(key)
            if not r:
                continue
            ratio = r["seconds"] / r["audio_seconds"] if r["audio_seconds"] else 0
            mem = f'{r["peak_rss_mb"]}'
            if "mlx_peak_mb" in r:
                mem += f'（MLX 自报 {r["mlx_peak_mb"]}）'
            lines.append(f'| {no} | {emo} | {r["model"]} | {r["file"]} | {r["seconds"]} | '
                         f'{r["audio_seconds"]} | {ratio:.1f} 倍 | {mem} |')
    b = next((v for k, v in data.items() if k.startswith("breeze_")), {})
    lines += [
        "", "## 版本与设置", "",
        f"- Breeze TTS 2：Hugging Face `mlx-community/Breeze-TTS-2-mlx`（bf16 不压缩）revision `{rev}`；"
        f"运行时 mlx-audio {md.version('mlx-audio')}、mlx {md.version('mlx')}；"
        f"cfg-scale {CFG}，temperature 0.9，seed {SEED}；模型加载 {b.get('load_seconds', '?')} 秒（不计入每句耗时）",
        f"- Breeze 参考音频：参考片段1.wav，文字稿「{REF_TEXT}」",
        "- Breeze 输出做了和 IndexTTS2 服务一样的音量均衡（loudness.py，-20 dBFS）；未均衡的原始输出在 `原始输出_未均衡/`",
        f"- IndexTTS2：现用的 tts_server.py（mlx-indextts，情绪强度 0.8，diffusion_steps 15），seed {SEED}",
        "- 峰值内存 = 生成期间进程物理占用（phys_footprint，含统一内存里的模型和 MLX 缓存）的最大值；"
        "IndexTTS2 是 tts_server 进程，Breeze 是测试进程（含已加载的模型）",
        "- Breeze 的物理占用一句比一句高（13→21 GB）是 MLX 把用过的显存留作缓存、没还给系统；"
        "每句真正用到的是「MLX 自报」那一栏（约 9–10.5 GB）。真要上线可以给它设缓存上限",
        "", "## 每句 Breeze 用的文字和语气指令", "",
    ]
    for no, emo, text, tag, ins in LINES:
        lines.append(f"- {no} {emo}：「{tag}{text}」 —— {KEEP}{ins}")
    (OUT / "对比表.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"index": run_index, "breeze": run_breeze, "table": make_table}[sys.argv[1]]()
