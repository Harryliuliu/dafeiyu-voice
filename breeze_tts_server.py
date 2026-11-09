"""大肥鱼说话服务（Breeze TTS 2 版）：和 tts_server.py 接口完全一样，二选一运行，占同一个端口。

运行（必须用 breeze-test 的 venv）：
  ~/dafeiyu-tools/breeze-test/.venv/bin/python ~/dafeiyu-tools/breeze_tts_server.py
平时不用手动跑：bash ~/dafeiyu-tools/dafeiyu-voice.sh engine breeze 切过来以后，start tts 就会启动它。

接口（只监听 127.0.0.1）：
  GET  /health            状态 loading / ready / error、排队数、闲置时间，engine = "breeze"
  POST /tts               {"text", "emotion", "temperature", "seed"} → wav（emo_alpha、speed 收下但不起作用）
  POST /v1/audio/speech   OpenAI 兼容：{"input", "voice"}，voice = 情绪名或 auto → wav
  POST /shutdown          立即退出

emotion 可以写：开心/傲娇/委屈/平静/害羞/惊讶（或对应英文）、auto（看文字猜）、
none（不加语气指令，只照参考音频），也可以直接写一句中文语气描述，例如“慵懒地拖长音”。
文字开头的【标签】会被去掉；emotion 为 auto 时优先用这个标签。
模型：mlx-community/Breeze-TTS-2-mlx（bf16），许可仅限研究和非商业用途。
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from loudness import level_loudness

TOOLS = Path.home() / "dafeiyu-tools"
BREEZE_ROOT = TOOLS / "breeze-test"
SAMPLE_RATE = 24000
CACHE_VERSION = "4"  # 改动合成方式时加一，旧缓存自动失效（2：前后加留白和淡出；3：换行处固定停顿；4：制表符=结巴短停顿）
# 前后留白：QQ 播放时开头结尾容易被吃掉一点，Breeze 有时最后一个字刚出口就结束，听着像被掐断
LEAD_SILENCE, TAIL_SILENCE, FADE_OUT = 0.12, 0.35, 0.03  # 秒
DASH_PAUSE = float(os.environ.get("DAFEIYU_DASH_PAUSE", "0.8"))  # 破折号（插件传来的换行）处的停顿长度，秒
STUTTER_PAUSE = float(os.environ.get("DAFEIYU_STUTTER_PAUSE", "0.35"))  # 结巴“谁、谁”（插件传来的制表符）处，秒
PAUSES = {"\n": DASH_PAUSE, "\t": STUTTER_PAUSE}

# 2026-11-09 起默认参考换成芙宁娜（游戏语音，正常说话），不再需要“别唱”。
# 旧的大肥鱼参考（参考片段1，是唱的）若用 --ref-audio 换回去，记得把 KEEP 加回“用自然说话的方式（不要唱），”
KEEP = "保持参考音频的音色，"
# 标签 → 语气指令（字符串，或多条时按文字固定挑一条）。来自 2026-10-29 的 A/B 试听：委屈、惊讶、开心、平静用户满意；
# 2026-11-09 芙宁娜参考试听：害羞原指令太柔，换成 A 嘴硬 / B 戏剧化 / C 慌乱 三条轮换（用户都满意）。
# 试听版在开心句前加了 [笑]、委屈句前加了 [叹气]；插件按句合成，每句都加会反复笑/叹气，所以这里不加。
# 2026-11-10：群聊里最常用开心、害羞，实测 5.0–5.7 字/秒偏快（原声 4.1），去掉这两类里的“语速快”字样。
# 2026-11-10：用户最喜欢的「我喜欢你，这句听清了没有？」是带“语速略快”的 A 版，试听后 A 和 A 略快都保留（四条轮换）。
# 2026-11-10：芙宁娜的傲娇分成两面（用户选的）：【傲娇】= 表层名伶端架子（自我介绍不再“不自信”），
#             【嘴硬】= 里层少女口是心非（原来的傲娇说明）。由大模型按场景选标签。
EMOTIONS = {
    "开心": "开心雀跃的语气，声音明亮，带着笑意。",
    "傲娇": "傲娇的大小姐语气：表面高傲、嘴上不饶人，语气笃定又带点得意，声音明亮，语调上扬，像在故意端架子，句尾干脆，不要软绵绵的。",
    "嘴硬": "嘴硬心软的傲娇语气：表面不耐烦、有点别扭，其实心里很高兴，尾音略带撒娇。",
    "委屈": "委屈的语气，声音低落、带一点哭腔，像忍着眼泪小声抱怨。",
    "平静": "平静温和的语气，语速平稳，放松自然。",
    "害羞": [
        "害羞但嘴硬的语气，像大小姐被说中心事时强装镇定，声音清亮、稍微提高，带点慌乱和结巴，不要压低声音。",
        "害羞但嘴硬的语气，像大小姐被说中心事时强装镇定，声音清亮、稍微提高，语速略快，带点慌乱和结巴，不要压低声音。",
        "戏剧化的害羞：一开始慌张结巴，后半句假装生气地抱怨，声音明亮有力，尾音带一点撒娇。",
        "被夸得手足无措的语气，又急又羞，声音偏高、带点颤，像在努力掩饰脸红。",
    ],
    "惊讶": "非常惊讶的语气，语调明显上扬，带着难以置信的感觉。",
}
ALIASES = {
    "happy": "开心", "tsundere": "傲娇", "sad": "委屈", "calm": "平静",
    "shy": "害羞", "surprised": "惊讶", "surprise": "惊讶", "stubborn": "嘴硬",
}
NO_EMOTION = {"none", "无", "原声", "reference"}
TAG_RE = re.compile(r"【([^】]{1,6})】")
# IndexTTS2 的情绪向量写法（如 "angry:0.5,happy:0.5"），Breeze 不认识
VECTOR_RE = re.compile(r"^[a-z]+(:[0-9.]+)?(,[a-z]+(:[0-9.]+)?)*$")


def parse_args():
    p = argparse.ArgumentParser(description="大肥鱼说话服务（Breeze TTS 2）")
    env = os.environ.get
    p.add_argument("--host", default=env("DAFEIYU_TTS_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(env("DAFEIYU_TTS_PORT", "9880")))
    p.add_argument("--model", default=env("DAFEIYU_BREEZE_MODEL", str(BREEZE_ROOT / "models" / "Breeze-TTS-2-mlx")))
    p.add_argument("--ref-audio", default=env("DAFEIYU_BREEZE_REF", str(TOOLS / "voices" / "芙宁娜_参考.wav")))
    p.add_argument("--ref-text", default=None, help="参考音频的逐字稿；默认读同名 .txt")
    p.add_argument("--cache-dir", default=env("DAFEIYU_BREEZE_CACHE", str(TOOLS / "cache" / "tts_breeze")))
    p.add_argument("--cache-max-mb", type=int, default=int(env("DAFEIYU_TTS_CACHE_MB", "300")))
    p.add_argument("--idle-timeout", type=int, default=int(env("DAFEIYU_TTS_IDLE", "900")),
                   help="闲置多少秒后自动退出，0 表示不退出（默认 900 = 15 分钟）")
    p.add_argument("--max-queue", type=int, default=int(env("DAFEIYU_TTS_QUEUE", "6")),
                   help="同时排队的句子上限，超过返回 503")
    p.add_argument("--max-chars", type=int, default=300)
    p.add_argument("--cfg-scale", type=float, default=float(env("DAFEIYU_BREEZE_CFG", "4")),
                   help="语气指令的力度，官方建议 4")
    p.add_argument("--no-level", action="store_true", help="不做音量均衡（输出模型原始音量）")
    p.add_argument("--cache-limit-gb", type=float, default=2.0,
                   help="MLX 留着备用的显存上限；不设的话会一路涨到 20 GB 以上")
    p.add_argument("--no-caffeinate", action="store_true", help="不阻止电脑休眠")
    return p.parse_args()


ARGS = parse_args()
CACHE_DIR = Path(ARGS.cache_dir)
RUN_DIR = TOOLS / "run"
PID_FILE = RUN_DIR / "tts_server.pid"  # 和 IndexTTS2 版共用：两者同一时间只开一个

logging.basicConfig(level=logging.INFO, format="%(asctime)s [说话服务·Breeze] %(message)s", datefmt="%m-%d %H:%M:%S")
log = logging.getLogger("breeze_tts_server")


class State:
    status = "loading"  # loading / ready / error
    error = ""
    load_seconds = 0.0
    pending = 0
    last_active = time.monotonic()
    started = time.time()


STATE = State()
MODEL = None
REF_TEXT = ""
# MLX 的数组和线程绑定，所以加载和合成都放在同一个线程里，顺便实现“一次只合成一句”
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
INFLIGHT: dict = {}
SERVER = None


# ---------------------------------------------------------------- 情绪与文字 ----
def guess_label(text: str) -> str:
    if "哼" in text:
        return "傲娇"
    if "呜" in text:
        return "委屈"
    if "嘿嘿" in text or "哈哈" in text:
        return "开心"
    return "平静"


def resolve(text: str, emotion: Optional[str]):
    """返回 (清理后的文字, 显示用标签, 语气指令或 None)。"""
    m = TAG_RE.match(text.strip())
    tag = m.group(1) if m else None
    text = TAG_RE.sub("", text).strip()

    e = (emotion or "auto").strip()
    if e.lower() == "auto":
        e = tag if tag in EMOTIONS else guess_label(text)
    e = ALIASES.get(e.lower(), e)

    if e.lower() in NO_EMOTION:
        return text, "none", None
    if e in EMOTIONS:
        ins = EMOTIONS[e]
        if isinstance(ins, list):
            # 同一句话每次一样（缓存有效），不同句子有变化
            ins = ins[int(hashlib.md5(text.encode()).hexdigest(), 16) % len(ins)]
        return text, e, KEEP + ins
    if VECTOR_RE.match(e.lower()):
        raise ValueError(f"Breeze 不认识 IndexTTS2 的情绪写法：{e}。可用：{'/'.join(EMOTIONS)}、auto、none，或一句中文语气描述")
    return text, "自定义", KEEP + e


# ---------------------------------------------------------------- 合成（工作线程） ----
def _load_model():
    global MODEL
    t = time.perf_counter()
    try:
        import mlx.core as mx
        from mlx_audio.tts import load

        mx.set_cache_limit(int(ARGS.cache_limit_gb * 2**30))
        MODEL = load(ARGS.model)
        STATE.load_seconds = time.perf_counter() - t
        STATE.status = "ready"
        log.info("模型加载完成，用时 %.1f 秒", STATE.load_seconds)
    except Exception as e:  # noqa: BLE001
        STATE.status, STATE.error = "error", f"{type(e).__name__}: {e}"
        log.exception("模型加载失败")


def _synth(text, instruct, temperature, seed, out_path: Path) -> float:
    import mlx.core as mx
    import numpy as np
    import soundfile as sf

    if MODEL is None:
        raise RuntimeError(f"模型没有加载成功：{STATE.error}")
    t = time.perf_counter()
    # 文字里的换行 / 制表符 = 插件标出的“硬停顿”（破折号 / 结巴）：Breeze 按它们分段合成，段与段之间插固定长度的静音。
    # （实测 Breeze 对破折号、顿号几乎不停，对“……”停 0.3–0.85 秒不稳定，插静音才能保证停够。）
    seps = re.findall(r"[\n\t]", text)
    pieces = [s for s in re.split(r"[\n\t]", text)]
    if any(not p.strip() for p in pieces):  # 有空段时 Breeze 会跳过它，对不上号就都按破折号停
        seps = []
    segs, last = [], None
    for r in MODEL.generate(text=text, instruct=instruct, ref_audio=ARGS.ref_audio, ref_text=REF_TEXT,
                            cfg_scale=ARGS.cfg_scale, temperature=temperature, seed=seed, split_pattern=r"[\n\t]"):
        seg = getattr(r, "segment_idx", 0)
        a = np.array(r.audio, dtype=np.float32)
        if segs and seg == last:
            segs[-1] = np.concatenate([segs[-1], a])
        else:
            segs.append(a)
        last = seg
    mx.clear_cache()
    chunks = []
    for i, a in enumerate(segs):
        if len(segs) > 1:  # 衔接处先去掉各段自带的长短不一的空白，再插固定长度的停顿
            a = _trim_edges(a, start=i > 0, end=i < len(segs) - 1)
        if chunks:
            pause = PAUSES.get(seps[i - 1], DASH_PAUSE) if len(seps) == len(segs) - 1 else DASH_PAUSE
            chunks.append(np.zeros(int(SAMPLE_RATE * pause), np.float32))
        chunks.append(a)
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if not ARGS.no_level:
        audio = level_loudness(audio, SAMPLE_RATE)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio *= 0.99 / peak
    n = min(int(SAMPLE_RATE * FADE_OUT), audio.size)
    if n:
        audio[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    audio = np.concatenate([np.zeros(int(SAMPLE_RATE * LEAD_SILENCE), np.float32), audio.astype(np.float32),
                            np.zeros(int(SAMPLE_RATE * TAIL_SILENCE), np.float32)])
    tmp = out_path.with_suffix(".tmp")
    sf.write(str(tmp), audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    os.replace(tmp, out_path)
    return time.perf_counter() - t


def _trim_edges(a, start: bool, end: bool, margin: float = 0.04):
    """去掉一段音频开头/结尾的空白（留 40 毫秒），结尾淡出 20 毫秒防止咔哒声。"""
    import numpy as np

    fr = int(SAMPLE_RATE * 0.01)
    if a.size < fr * 3:
        return a
    rms = np.sqrt(np.mean(a[: a.size // fr * fr].reshape(-1, fr) ** 2, axis=1))
    voiced = np.where(rms > rms.max() * 0.03)[0]
    if voiced.size == 0:
        return a
    m = int(margin * SAMPLE_RATE)
    lo = max(0, voiced[0] * fr - m) if start else 0
    hi = min(a.size, (voiced[-1] + 1) * fr + m) if end else a.size
    a = a[lo:hi].copy()
    if end:
        k = min(int(SAMPLE_RATE * 0.02), a.size)
        a[-k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
    return a


def _prune_cache():
    files = sorted(CACHE_DIR.glob("*.wav"), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    limit = ARGS.cache_max_mb * 1024 * 1024
    while files and total > limit:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink(missing_ok=True)


async def synthesize(text, emotion, temperature, seed):
    from fastapi import HTTPException

    try:
        text, label, instruct = resolve(text or "", emotion)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not text:
        raise HTTPException(400, "文字是空的")
    if len(text) > ARGS.max_chars:
        raise HTTPException(400, f"文字太长（{len(text)} 字，上限 {ARGS.max_chars}）")
    if STATE.status == "error":
        raise HTTPException(503, f"模型没有加载成功：{STATE.error}")

    key_src = json.dumps(
        [CACHE_VERSION, Path(ARGS.ref_audio).name, REF_TEXT, text, instruct, round(temperature, 3), seed,
         ARGS.cfg_scale, not ARGS.no_level],
        ensure_ascii=False,
    )
    key = hashlib.sha1(key_src.encode()).hexdigest()
    path = CACHE_DIR / f"{key}.wav"
    STATE.last_active = time.monotonic()

    if path.exists():
        os.utime(path)
        return path, label, True, 0.0

    fut = INFLIGHT.get(key)
    if fut is None:
        if STATE.pending >= ARGS.max_queue:
            raise HTTPException(503, f"排队的句子太多了（{STATE.pending}），稍后再试")
        STATE.pending += 1
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(EXECUTOR, _synth, text, instruct, temperature, seed, path)
        INFLIGHT[key] = fut

        def _done(_f, key=key):
            STATE.pending -= 1
            STATE.last_active = time.monotonic()
            INFLIGHT.pop(key, None)

        fut.add_done_callback(_done)

    try:
        secs = await asyncio.shield(fut)
    except Exception as e:  # noqa: BLE001
        log.exception("合成失败")
        raise HTTPException(500, f"合成失败：{type(e).__name__}: {e}")
    _prune_cache()
    log.info("合成 %.1f 秒｜%s｜%s", secs, label, text[:40])
    return path, label, False, secs


# ---------------------------------------------------------------- 后台任务 ----
async def idle_watchdog():
    while True:
        await asyncio.sleep(15)
        if ARGS.idle_timeout <= 0 or STATE.pending > 0 or STATE.status == "loading":
            continue
        idle = time.monotonic() - STATE.last_active
        if idle > ARGS.idle_timeout:
            log.info("闲置 %d 秒，自动退出", int(idle))
            SERVER.should_exit = True
            return


def start_caffeinate():
    if ARGS.no_caffeinate or sys.platform != "darwin":
        return
    try:
        # -i 阻止闲置休眠；-w 跟随本进程，本进程退出它也退出
        subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    except OSError as e:
        log.warning("caffeinate 启动失败：%s", e)


@asynccontextmanager
async def lifespan(app):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    start_caffeinate()
    STATE.last_active = time.monotonic()
    log.info("启动：端口 %d，参考 %s，cfg %.1f，闲置 %d 秒退出",
             ARGS.port, Path(ARGS.ref_audio).name, ARGS.cfg_scale, ARGS.idle_timeout)
    asyncio.get_running_loop().run_in_executor(EXECUTOR, _load_model)
    watchdog = asyncio.create_task(idle_watchdog())
    try:
        yield
    finally:
        watchdog.cancel()
        try:
            if PID_FILE.read_text().strip() == str(os.getpid()):
                PID_FILE.unlink()
        except OSError:
            pass
        log.info("已退出")


# ---------------------------------------------------------------- HTTP ----
def build_app():
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    app = FastAPI(title="大肥鱼说话服务（Breeze）", lifespan=lifespan)

    class TTSRequest(BaseModel):
        text: str
        emotion: Optional[str] = "auto"
        emo_alpha: Optional[float] = None  # IndexTTS2 用的，这里不起作用
        temperature: float = 0.9
        seed: int = 42
        speed: float = 1.0  # Breeze 不支持调速，收下忽略

    class SpeechRequest(BaseModel):
        input: str
        voice: Optional[str] = "auto"
        model: Optional[str] = None
        response_format: Optional[str] = "wav"
        speed: Optional[float] = 1.0

    def wav_response(path, label, cached, secs):
        return FileResponse(
            path,
            media_type="audio/wav",
            headers={
                "X-Cache": "hit" if cached else "miss",
                "X-Emotion": quote(label),
                "X-Gen-Seconds": f"{secs:.2f}",
            },
        )

    @app.get("/health")
    async def health():
        return {
            "status": STATE.status,
            "engine": "breeze",
            "error": STATE.error,
            "pending": STATE.pending,
            "idle_seconds": int(time.monotonic() - STATE.last_active),
            "idle_timeout": ARGS.idle_timeout,
            "model_load_seconds": round(STATE.load_seconds, 1),
            "uptime_seconds": int(time.time() - STATE.started),
            "speaker": Path(ARGS.ref_audio).name,
            "cfg_scale": ARGS.cfg_scale,
            "level_loudness": not ARGS.no_level,
            "emotions": list(EMOTIONS),
        }

    @app.post("/tts")
    async def tts(req: TTSRequest):
        r = await synthesize(req.text, req.emotion, req.temperature, req.seed)
        return wav_response(*r)

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        r = await synthesize(req.input, req.voice, 0.9, 42)
        return wav_response(*r)

    @app.post("/shutdown")
    async def shutdown():
        log.info("收到退出请求")
        SERVER.should_exit = True
        return {"ok": True}

    return app


def main():
    import uvicorn

    global SERVER, REF_TEXT
    ref = Path(ARGS.ref_audio)
    if not ref.exists():
        sys.exit(f"找不到参考音频：{ref}")
    REF_TEXT = (ARGS.ref_text or "").strip()
    if not REF_TEXT:
        txt = ref.with_suffix(".txt")
        if not txt.exists():
            sys.exit(f"Breeze 需要参考音频的逐字稿：请写在 {txt}")
        REF_TEXT = txt.read_text(encoding="utf-8").strip()
    # 先确认端口空着，免得第二个实例加载模型、改写 pid 文件后才发现绑定失败
    with socket.socket() as s:
        if s.connect_ex((ARGS.host, ARGS.port)) == 0:
            sys.exit(f"端口 {ARGS.port} 已被占用：说话服务（IndexTTS2 或 Breeze）可能已经在运行")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = uvicorn.Config(build_app(), host=ARGS.host, port=ARGS.port, log_level="warning", access_log=False)
    SERVER = uvicorn.Server(config)
    SERVER.run()
    # 工作线程可能还在合成，不等它，直接结束进程
    os._exit(0)


if __name__ == "__main__":
    main()
