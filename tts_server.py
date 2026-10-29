"""大肥鱼说话服务：IndexTTS2（mlx-indextts）只加载一次，通过 HTTP 给 AstrBot 插件调用。

运行（必须用 mlx-indextts 的 venv）：
  ~/dafeiyu-tools/mlx-indextts/.venv/bin/python ~/dafeiyu-tools/tts_server.py

接口（只监听 127.0.0.1）：
  GET  /health            状态 loading / ready / error、排队数、闲置时间
  POST /tts               {"text", "emotion", "emo_alpha", "temperature", "seed", "speed"} → wav
  POST /v1/audio/speech   OpenAI 兼容：{"input", "voice"}，voice = 情绪名或 auto → wav
  POST /shutdown          立即退出（给插件的“/声音 关”和停止脚本用）

emotion 可以写：开心/傲娇/委屈/平静/害羞/惊讶（或对应英文）、auto（看文字猜）、
none（只用参考音频的语气），或原始写法如 "angry:0.5,happy:0.5"。
文字开头的【标签】会被去掉；emotion 为 auto 时优先用这个标签。
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
MI_ROOT = TOOLS / "mlx-indextts"
SAMPLE_RATE = 22050
CACHE_VERSION = "2"  # 改动合成方式时加一，旧缓存自动失效

# 标签 → [(情绪参数, 默认强度), ...]，都是用户试听后定的。
# 有多个配比时按文字固定挑一个：同一句话每次一样（缓存有效），不同句子有变化
EMOTIONS = {
    "开心": [("happy", 0.8)],
    "傲娇": [("happy:0.45,surprised:0.25,angry:0.15,disgusted:0.15", 0.8)],
    "委屈": [("sad", 0.8)],
    "平静": [("calm", 0.6)],
    "害羞": [("happy:0.4,calm:0.4,afraid:0.2", 0.8), ("happy:0.5,calm:0.3,sad:0.2", 0.8)],
    "惊讶": [("surprised", 0.8)],
}
ALIASES = {
    "happy": "开心", "tsundere": "傲娇", "sad": "委屈", "calm": "平静",
    "shy": "害羞", "surprised": "惊讶", "surprise": "惊讶",
}
BASIC_EMOTIONS = {"happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"}
NO_EMOTION = {"none", "无", "原声", "reference"}
TAG_RE = re.compile(r"【([^】]{1,6})】")


def parse_args():
    p = argparse.ArgumentParser(description="大肥鱼说话服务")
    env = os.environ.get
    p.add_argument("--host", default=env("DAFEIYU_TTS_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(env("DAFEIYU_TTS_PORT", "9880")))
    p.add_argument("--model", default=env("DAFEIYU_TTS_MODEL", str(MI_ROOT / "models" / "IndexTTS-2-MLX")))
    p.add_argument("--speaker", default=env("DAFEIYU_TTS_SPEAKER", str(TOOLS / "voices" / "speaker_片段1.npz")))
    p.add_argument("--cache-dir", default=env("DAFEIYU_TTS_CACHE", str(TOOLS / "cache" / "tts")))
    p.add_argument("--cache-max-mb", type=int, default=int(env("DAFEIYU_TTS_CACHE_MB", "300")))
    p.add_argument("--idle-timeout", type=int, default=int(env("DAFEIYU_TTS_IDLE", "900")),
                   help="闲置多少秒后自动退出，0 表示不退出（默认 900 = 15 分钟）")
    p.add_argument("--max-queue", type=int, default=int(env("DAFEIYU_TTS_QUEUE", "6")),
                   help="同时排队的句子上限，超过返回 503")
    p.add_argument("--max-chars", type=int, default=300)
    p.add_argument("--diffusion-steps", type=int, default=int(env("DAFEIYU_TTS_STEPS", "15")),
                   help="声音还原的步数：25 是原版默认，15 快约 20%%，用户试听听不出区别")
    p.add_argument("--no-level", action="store_true", help="不做音量均衡（输出模型原始音量）")
    p.add_argument("--memory-limit-gb", type=float, default=12.0)
    p.add_argument("--no-caffeinate", action="store_true", help="不阻止电脑休眠")
    return p.parse_args()


ARGS = parse_args()
CACHE_DIR = Path(ARGS.cache_dir)
RUN_DIR = TOOLS / "run"
PID_FILE = RUN_DIR / "tts_server.pid"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [说话服务] %(message)s", datefmt="%m-%d %H:%M:%S")
log = logging.getLogger("tts_server")


class State:
    status = "loading"  # loading / ready / error
    error = ""
    load_seconds = 0.0
    pending = 0
    last_active = time.monotonic()
    started = time.time()


STATE = State()
TTS = None
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


def resolve(text: str, emotion: Optional[str], emo_alpha: Optional[float]):
    """返回 (清理后的文字, 显示用标签, 情绪参数或 None, 强度)。"""
    m = TAG_RE.match(text.strip())
    tag = m.group(1) if m else None
    text = TAG_RE.sub("", text).strip()

    e = (emotion or "auto").strip()
    if e.lower() == "auto":
        e = tag if tag in EMOTIONS else guess_label(text)
    e = ALIASES.get(e.lower(), e)

    if e.lower() in NO_EMOTION:
        return text, "none", None, 0.0
    if e in EMOTIONS:
        variants = EMOTIONS[e]
        emo, alpha = variants[int(hashlib.md5(text.encode()).hexdigest(), 16) % len(variants)]
        label = e
    else:
        # 原始写法，例如 "angry:0.5,happy:0.5"
        names = [part.split(":", 1)[0].strip().lower() for part in e.split(",") if part.strip()]
        if not names or any(n not in BASIC_EMOTIONS for n in names):
            raise ValueError(
                f"不认识的情绪：{e}。可用：{'/'.join(EMOTIONS)}、auto、none，"
                f"或 {'/'.join(sorted(BASIC_EMOTIONS))} 的组合"
            )
        emo, alpha, label = e, 0.8, e
    if emo_alpha is not None:
        alpha = max(0.0, min(1.0, float(emo_alpha)))
    return text, label, emo, alpha


# ---------------------------------------------------------------- 合成（工作线程） ----
def _load_model():
    global TTS
    t = time.perf_counter()
    try:
        os.chdir(MI_ROOT)
        from mlx_indextts.generate_v2 import IndexTTSv2

        TTS = IndexTTSv2(model_dir=ARGS.model, memory_limit_gb=ARGS.memory_limit_gb)
        STATE.load_seconds = time.perf_counter() - t
        STATE.status = "ready"
        log.info("模型加载完成，用时 %.1f 秒", STATE.load_seconds)
    except Exception as e:  # noqa: BLE001
        STATE.status, STATE.error = "error", f"{type(e).__name__}: {e}"
        log.exception("模型加载失败")


def _synth(text, emo, alpha, temperature, seed, speed, out_path: Path) -> float:
    import numpy as np
    import soundfile as sf

    if TTS is None:
        raise RuntimeError(f"模型没有加载成功：{STATE.error}")
    t = time.perf_counter()
    audio = TTS.generate(
        text=text,
        reference_audio=ARGS.speaker,
        emotion=emo,
        emo_alpha=alpha,
        temperature=temperature,
        seed=seed,
        speed=speed,
        diffusion_steps=ARGS.diffusion_steps,
    )
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not ARGS.no_level:
        audio = level_loudness(audio, SAMPLE_RATE)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio *= 0.99 / peak
    tmp = out_path.with_suffix(".tmp")
    sf.write(str(tmp), audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    os.replace(tmp, out_path)
    return time.perf_counter() - t


def _prune_cache():
    files = sorted(CACHE_DIR.glob("*.wav"), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    limit = ARGS.cache_max_mb * 1024 * 1024
    while files and total > limit:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink(missing_ok=True)


async def synthesize(text, emotion, emo_alpha, temperature, seed, speed):
    from fastapi import HTTPException

    try:
        text, label, emo, alpha = resolve(text or "", emotion, emo_alpha)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not text:
        raise HTTPException(400, "文字是空的")
    if len(text) > ARGS.max_chars:
        raise HTTPException(400, f"文字太长（{len(text)} 字，上限 {ARGS.max_chars}）")
    if STATE.status == "error":
        raise HTTPException(503, f"模型没有加载成功：{STATE.error}")

    key_src = json.dumps(
        [CACHE_VERSION, Path(ARGS.speaker).name, text, emo, round(alpha, 3),
         round(temperature, 3), seed, round(speed, 3), ARGS.diffusion_steps, not ARGS.no_level],
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
        fut = loop.run_in_executor(EXECUTOR, _synth, text, emo, alpha, temperature, seed, speed, path)
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
    log.info("启动：端口 %d，音色 %s，闲置 %d 秒退出", ARGS.port, Path(ARGS.speaker).name, ARGS.idle_timeout)
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

    app = FastAPI(title="大肥鱼说话服务", lifespan=lifespan)

    class TTSRequest(BaseModel):
        text: str
        emotion: Optional[str] = "auto"
        emo_alpha: Optional[float] = None
        temperature: float = 1.0
        seed: int = 42
        speed: float = 1.0

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
            "error": STATE.error,
            "pending": STATE.pending,
            "idle_seconds": int(time.monotonic() - STATE.last_active),
            "idle_timeout": ARGS.idle_timeout,
            "model_load_seconds": round(STATE.load_seconds, 1),
            "uptime_seconds": int(time.time() - STATE.started),
            "speaker": Path(ARGS.speaker).name,
            "diffusion_steps": ARGS.diffusion_steps,
            "level_loudness": not ARGS.no_level,
            "emotions": list(EMOTIONS),
        }

    @app.post("/tts")
    async def tts(req: TTSRequest):
        r = await synthesize(req.text, req.emotion, req.emo_alpha, req.temperature, req.seed, req.speed)
        return wav_response(*r)

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        speed = req.speed if req.speed and 0.5 <= req.speed <= 2.0 else 1.0
        r = await synthesize(req.input, req.voice, None, 1.0, 42, speed)
        return wav_response(*r)

    @app.post("/shutdown")
    async def shutdown():
        log.info("收到退出请求")
        SERVER.should_exit = True
        return {"ok": True}

    return app


def main():
    import uvicorn

    if not Path(ARGS.speaker).exists():
        sys.exit(f"找不到音色文件：{ARGS.speaker}")
    # 先确认端口空着，免得第二个实例加载模型、改写 pid 文件后才发现绑定失败
    with socket.socket() as s:
        if s.connect_ex((ARGS.host, ARGS.port)) == 0:
            sys.exit(f"端口 {ARGS.port} 已被占用：说话服务可能已经在运行")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    global SERVER
    config = uvicorn.Config(build_app(), host=ARGS.host, port=ARGS.port, log_level="warning", access_log=False)
    SERVER = uvicorn.Server(config)
    SERVER.run()
    # 工作线程可能还在合成，不等它，直接结束进程
    os._exit(0)


if __name__ == "__main__":
    main()
