"""大肥鱼唱歌服务：点歌 → 找音源 → 截片段 → 分离人声 → 去和声 → 难度检查 → 换音色 → 清唱 wav。

两个歌手（2026-11-10 起，雨爱已去掉）：
  dafeiyu = 大肥鱼：Seed-VC + voices/歌声参考_23秒.wav（在本进程里）
  furina  = 芙宁娜：社区芙宁娜 RVC 模型（rvc_worker.py 子进程，Applio 的 venv）
点歌时可以指定 voice；不指定就用 sing_voice.conf 里的默认歌手（dafeiyu-voice.sh voice 改，默认大肥鱼）。
分离在 sep_worker.py 子进程里：Kim 版 Mel-Band Roformer 分人声 → karaoke 模型去和声；
去和声后主唱几乎没声音（模型把主唱当成了和声）就退回全部人声；分离进程出问题时用 demucs 兜底。

运行（必须用 seed-vc 的 venv）：
  ~/dafeiyu-tools/seed-vc/.venv/bin/python ~/dafeiyu-tools/sing_server.py

接口（只监听 127.0.0.1）：
  POST /sing              {"song", "artist", "start", "duration", "voice", "probe", "learn", "learn_daily_limit"}
                          → 任务（命中缓存直接 done）。库里没有且 learn=true 时，先去学歌（默认走 go-music-dl 多源）
                          probe=true：只看库里有没有、能不能唱，返回 found，不接单（插件先问对方想听谁唱）
  GET  /sing/{job_id}     任务状态：queued / running(找歌/截取/分离/检查难度/转换) / done / failed + 原因
  GET  /songs             歌曲库里能唱的歌
  GET  /health            状态、排队情况
  POST /admin/clear_cache 清掉转换好的缓存（{"all": true} 连分离好的人声一起清）
  POST /admin/blacklist   {"song", "reason"} 拉黑
  POST /shutdown          立即退出

歌曲库（默认 ~/dafeiyu-tools/歌曲库）：
  清唱/ 原曲/      文件名“歌手-歌名.m4a/mp3/wav/flac”，同名时优先用清唱
  片段.txt         每行“歌手-歌名 起点”，如“邓紫棋-泡沫 2:37”
  白名单.txt       每行“歌手-歌名 [升降调]”：不做难度检查直接唱；写了升降调就按它来
  黑名单.txt       每行“歌手-歌名 | 原因”（原因可省略）：直接拒绝
  缓存/            转换结果；缓存/分离/ 是整首分离好的人声
  待导入/          把下载的 .ncm/mp3/flac 丢进来，点歌或看歌单时自动解密、改名进 原曲/
  学歌记录.txt     每次学歌记一行，用来算每天的上限
  想学的歌.txt     群友点过但库里没有的歌（“歌名 | 次数 | 最近时间”），照着去下载或教她
"""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # 必须在导入 torch 之前
os.environ.setdefault("HF_HUB_OFFLINE", "1")

TOOLS = Path.home() / "dafeiyu-tools"
CACHE_VERSION = "2"  # 改动转换方式时加一，旧缓存自动失效（2 = Roformer 分离 + 去和声）
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".flac", ".aac", ".ogg"}


def parse_args():
    p = argparse.ArgumentParser(description="大肥鱼唱歌服务")
    env = os.environ.get
    p.add_argument("--host", default=env("DAFEIYU_SING_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(env("DAFEIYU_SING_PORT", "9881")))
    p.add_argument("--library", default=env("DAFEIYU_SING_LIBRARY", str(TOOLS / "歌曲库")))
    # 大肥鱼（Seed-VC）的参考歌声。2026-11-05 试过雨爱，2026-11-10 用户决定只留大肥鱼和芙宁娜
    p.add_argument("--ref", default=env("DAFEIYU_SING_REF", str(TOOLS / "voices" / "歌声参考_23秒.wav")))
    p.add_argument("--voice-file", default=env("DAFEIYU_SING_VOICE_FILE", str(TOOLS / "sing_voice.conf")),
                   help="默认歌手记在这个文件里（furina / dafeiyu），点歌没指定歌手时用它")
    # 芙宁娜 RVC 没有参考歌声，硬唱时按她说话的中位音高（语音包约 301 Hz，挑的参考 308 Hz）拉过去
    p.add_argument("--furina-median", type=float, default=305.0)
    p.add_argument("--no-karaoke", action="store_true", default=env("DAFEIYU_SING_KARAOKE", "1") == "0",
                   help="不去和声（只分人声）")
    p.add_argument("--lead-min-db", type=float, default=6.0,
                   help="去和声后主唱比全部人声小这么多分贝以上，就当去和声失败，退回全部人声")
    # 用户试听选了“参考 23 秒 + 40 步”：长音气息更稳；参考缩到 12 秒能快 2.5 倍但有轻微电音
    p.add_argument("--ref-seconds", type=float, default=float(env("DAFEIYU_SING_REF_SECONDS", "25")))
    p.add_argument("--diffusion-steps", type=int, default=int(env("DAFEIYU_SING_STEPS", "40")))
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--idle-timeout", type=int, default=int(env("DAFEIYU_SING_IDLE", "300")),
                   help="闲置多少秒后自动退出，0 表示不退出（默认 300 = 5 分钟）")
    p.add_argument("--learn-source", choices=["musicdl", "bili"], default=env("DAFEIYU_LEARN_SOURCE", "musicdl"),
                   help="学歌去哪：musicdl=本机 go-music-dl（多源），bili=B站")
    p.add_argument("--max-queue", type=int, default=3, help="排队加正在做的歌，最多几首")
    p.add_argument("--default-duration", type=float, default=30.0)
    p.add_argument("--max-duration", type=float, default=60.0)
    p.add_argument("--cache-max-gb", type=float, default=2.0)
    # 挑歌阈值（原唱人声中位音高，Hz）：范围内直接唱；偏高一点降 3 个半音；
    # 再高或再低：默认升降调拉到她的音域硬唱（任务里带 stretch，插件先说一句可能唱不好），--no-stretch 时拒绝
    p.add_argument("--ok-low", type=float, default=280)
    p.add_argument("--ok-high", type=float, default=370)
    p.add_argument("--shift-high", type=float, default=430)
    p.add_argument("--shift-semitones", type=int, default=-3)
    p.add_argument("--no-stretch", action="store_true",
                   default=os.environ.get("DAFEIYU_SING_STRETCH", "1") == "0",
                   help="超出音域的歌直接拒绝（默认会升降调试着唱）")
    p.add_argument("--max-shift", type=int, default=12, help="硬唱时最多升降几个半音")
    p.add_argument("--no-caffeinate", action="store_true")
    return p.parse_args()


ARGS = parse_args()
LIB = Path(ARGS.library)
CACHE = LIB / "缓存"
SEP_CACHE = CACHE / "分离"
RUN_DIR = TOOLS / "run"
PID_FILE = RUN_DIR / "sing_server.pid"


def voice_tag(ref: str) -> str:
    """缓存文件名里的音色标记：换了参考歌声，旧音色唱的缓存就不会被拿出来。
    原来那段参考不加标记，换回去时它以前的缓存还能用。"""
    if Path(ref).name == "歌声参考_23秒.wav":
        return ""
    import hashlib
    return "_" + hashlib.md5(Path(ref).read_bytes()).hexdigest()[:6]


VOICE_TAG = ""  # main() 里按 --ref 算好（大肥鱼）
RVC_TAG = ""  # main() 里按芙宁娜 RVC 模型文件算好

# 两个歌手。label 给插件和日志看；aliases 是点歌时可能的写法
VOICES = {
    "dafeiyu": {"label": "大肥鱼", "engine": "Seed-VC"},
    "furina": {"label": "芙宁娜", "engine": "RVC"},
}
VOICE_ALIASES = {"dafeiyu": "dafeiyu", "大肥鱼": "dafeiyu", "old": "dafeiyu", "原音色": "dafeiyu",
                 "furina": "furina", "芙宁娜": "furina"}


def default_voice() -> str:
    """sing_voice.conf 里的默认歌手。以前的 old（原音色）就是大肥鱼；new（雨爱）已去掉，也按大肥鱼。"""
    try:
        v = Path(ARGS.voice_file).read_text(encoding="utf-8").strip()
    except OSError:
        v = ""
    return "furina" if v == "furina" else "dafeiyu"


def pick_voice(v) -> str:
    return VOICE_ALIASES.get(str(v or "").strip().lower()) or default_voice()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [唱歌服务] %(message)s", datefmt="%m-%d %H:%M:%S")
log = logging.getLogger("sing_server")


class State:
    status = "loading"  # loading / ready / error
    error = ""
    load_seconds = 0.0
    last_active = time.monotonic()
    started = time.time()
    # 用实际耗时滚动更新，用来估算等待时间：大肥鱼 30 秒片段约 3.5 分钟，芙宁娜（RVC）不到 1 分钟
    avg_job_seconds = {"dafeiyu": 220.0, "furina": 50.0}
    learn_blocked_until = 0.0


STATE = State()
ENGINE = None
SEP = None  # 人声分离子进程（sing_workers.Worker）
RVC = None  # 芙宁娜 RVC 子进程
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sing")  # 模型和任务都在这一个线程里
JOBS: dict = {}
ACTIVE: dict = {}  # 缓存键 → 未完成的 job_id，同一首歌重复点只做一次
SERVER = None


# ---------------------------------------------------------------- 歌曲库 ----
def norm(s: str) -> str:
    return re.sub(r"[\s\-_·・.,，。!！?？'\"“”‘’()（）\[\]【】《》<>]+", "", s or "").lower()


def split_stem(stem: str):
    artist, _, title = stem.partition("-")
    return (artist.strip(), title.strip()) if title else ("", stem.strip())


def parse_time(v: Union[str, float, int, None]) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("：", ":")
    if ":" in s:
        m, _, sec = s.partition(":")
        return int(m) * 60 + float(sec)
    return float(s)


LIST_TAIL = {
    "片段.txt": re.compile(r"^(.+?)\s+(\d+[:：]\d+(?:\.\d+)?|\d+(?:\.\d+)?)$"),  # 歌手-歌名 1:47
    "白名单.txt": re.compile(r"^(.+?)\s+([+-]?\d+)$"),  # 歌手-歌名 -3（半音可省略）
    "黑名单.txt": re.compile(r"^(.+?)\s*[|｜]\s*(.*)$"),  # 歌手-歌名 | 原因（原因可省略）
}


def read_list(name: str) -> dict:
    """读名单，返回 {规范化歌名: (原始歌名, 附加内容)}。歌名里可以有空格。"""
    path = LIB / name
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = LIST_TAIL[name].match(line)
        stem, rest = (m.group(1).strip(), m.group(2).strip()) if m else (line, "")
        out[norm(stem)] = (stem, rest)
    return out


def list_key(stem: str, table: dict):
    """名单里写“歌手-歌名”或只写“歌名”都算。"""
    artist, title = split_stem(stem)
    for k in (norm(stem), norm(title)):
        if k in table:
            return table[k]
    return None


def library_songs():
    songs = {}
    for kind in ("清唱", "原曲"):
        d = LIB / kind
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in AUDIO_EXT and not f.name.startswith("."):
                songs.setdefault(f.stem, {})[kind] = f
    return songs


# 歌名里的版本说明：“雨爱(司风版)”“xx Live”是另一首，不能拿库里的原版顶替
VERSION_RE = re.compile(r"[（(【\[][^)）】\]]*[)）】\]]|版|live|remix|cover|翻唱|伴奏|dj", re.I)


def find_song(song: str, artist: str = ""):
    """返回 (歌手-歌名, 种类, 路径)；先精确匹配歌名，再包含匹配；同名优先清唱。
    写了歌手就必须对得上（对不上算没有，点歌时由 similar_songs 去问“是不是库里那首”）；
    没写歌手、点的歌名又带版本说明而库里的没有时，不算包含匹配。"""
    q_title, q_artist = norm(song), norm(artist)
    if not q_title:
        return None
    q_version = bool(VERSION_RE.search(song))
    exact, fuzzy = [], []
    for stem, kinds in library_songs().items():
        a, t = split_stem(stem)
        na, nt = norm(a), norm(t)
        if q_artist and q_artist not in na and na not in q_artist:
            continue
        if nt == q_title or norm(stem) == q_title:
            exact.append((stem, kinds))
        elif len(q_title) >= 2 and q_title in nt:
            fuzzy.append((stem, kinds))
        elif len(nt) >= 2 and nt in q_title and not (q_version and not q_artist and not VERSION_RE.search(t)):
            # 写了歌手且对得上时，学来的“时年先生-雨爱”就是点的“雨爱(司风版)”（入库时去掉了括号）
            fuzzy.append((stem, kinds))
    for stem, kinds in exact + fuzzy:
        kind = "清唱" if "清唱" in kinds else "原曲"
        return stem, kind, kinds[kind]
    return None


def similar_songs(song: str, artist: str = "", limit: int = 3) -> list:
    """歌单里歌名只差一两个字的歌（打错、漏字），按相似度排；黑名单里的不算。"""
    from difflib import SequenceMatcher

    q = norm(song)
    if len(q) < 2:
        return []
    black = read_list("黑名单.txt")
    scored = []
    for stem in library_songs():
        if list_key(stem, black):
            continue
        a, t = split_stem(stem)
        nt = norm(t)
        ratio = SequenceMatcher(None, q, nt).ratio()
        if abs(len(nt) - len(q)) <= 3 and ratio >= 0.7:
            if artist and norm(artist) not in norm(a):
                ratio -= 0.1  # 写了歌手但对不上，排后面
            scored.append((ratio, stem))
    return [s for _, s in sorted(scored, reverse=True)[:limit]]


# 学下来发现唱不了的这些原因，就把下载的音频删掉、拉黑，免得占地方或重复下载
UNSINGABLE = ("too_high", "too_low", "no_vocals", "too_short")


def drop_learned(job):
    """学歌下载的歌唱不了：删音频和分离缓存，拉进黑名单。只动学歌下载的文件。"""
    path = Path(job.get("_learned_file") or "")
    stem = job.get("song") or path.stem
    removed = []
    for f in (path, SEP_CACHE / f"{stem}.flac"):
        if f.name and f.exists():
            f.unlink()
            removed.append(f.name)
    with open(LIB / "黑名单.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{stem} | 学过：{job.get('message', '唱不了')}\n")
    job["deleted"] = True
    log.info("学歌唱不了，已删除 %s 并拉黑：%s", "、".join(removed) or "（没有文件）", stem)


# ---------------------------------------------------------------- 任务 ----
LEARN_LOG = "学歌记录.txt"
LEARN_EXTRA_SECONDS = 150  # 学歌比普通点歌多出来的：下载 + 整首分离
LEARN_BLOCK_SECONDS = 3600  # B站返回 412 后暂停学歌的时间


def learned_today() -> int:
    path = LIB / LEARN_LOG
    if not path.exists():
        return 0
    today = time.strftime("%Y-%m-%d")
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(today))


def log_learn(query: str, result: str):
    with open(LIB / LEARN_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M')} | {query} | {result}\n")


WISH_FILE = "想学的歌.txt"


def record_wish(song: str, artist: str = ""):
    """群友点了库里没有的歌：记进 想学的歌.txt（同一首累加次数），给管理员照着去下载。"""
    name = (f"{artist.strip()}-{song.strip()}" if artist and artist.strip() else song.strip())[:60]
    if not name:
        return
    path = LIB / WISH_FILE
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3 and parts[1].isdigit():
                rows[parts[0]] = [int(parts[1]), parts[2]]
    def words(s):  # “晴天 周杰伦”和“周杰伦-晴天”算同一首
        return frozenset(norm(w) for w in re.split(r"[\s\-]+", s) if norm(w))

    key = next((k for k in rows if words(k) == words(name)), name)
    count = rows.pop(key, [0, ""])[0] + 1
    if "-" in name and "-" not in key:
        key = name  # 有“歌手-歌名”写法就用它
    rows[key] = [count, time.strftime("%Y-%m-%d %H:%M")]
    lines = [f"{k} | {c} | {t}" for k, (c, t) in sorted(rows.items(), key=lambda kv: (-kv[1][0], kv[1][1]))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_inbox():
    try:
        from ncm_import import import_pending

        import_pending(LIB, log=log.info)
    except Exception:  # noqa: BLE001
        log.exception("处理待导入文件夹出错")


def new_job(stem, kind, path, start, duration, key, voice):
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {
        "job_id": jid, "song": stem, "source": kind, "status": "queued", "stage": "排队",
        "start": start, "duration": duration, "reason": "", "message": "",
        "file": "", "semitone": None, "pitch": None, "stretch": None,
        "voice": voice, "voice_label": VOICES[voice]["label"], "separation": None, "start_note": None,
        "created": time.time(), "started": None, "finished": None,
        "_path": str(path), "_key": key,
    }
    return JOBS[jid]


def public(job: dict) -> dict:
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    out["learning"] = bool(job.get("_learn"))
    extra = LEARN_EXTRA_SECONDS if job.get("_learn") else 0
    avg = STATE.avg_job_seconds
    if job["status"] == "queued":
        ahead = [j for j in JOBS.values() if j["status"] in ("queued", "running") and j["created"] < job["created"]]
        out["position"] = len(ahead) + 1
        out["eta_seconds"] = int(sum(avg[j["voice"]] for j in ahead) + avg[job["voice"]] + extra)
    elif job["status"] == "running":
        out["eta_seconds"] = max(5, int(avg[job["voice"]] + extra - (time.time() - job["started"])))
    return out


def fail(job, reason, message):
    job.update(status="failed", reason=reason, message=message, finished=time.time())
    log.info("失败｜%s｜%s", job["song"], message)


def cache_name(stem, start, duration, voice):
    start_txt = "chorus" if start is None else f"{start:.1f}"  # 没写起点 = 自动挑副歌（11-10 以前是 auto=人声最响）
    safe = re.sub(r'[\\/:*?"<>|]', "_", stem)
    # 换了转换设置、歌手或分离方式，旧缓存自动不用
    if voice == "furina":
        setting = f"rvc{RVC_TAG}"
    else:
        setting = f"r{ARGS.ref_seconds:g}s{ARGS.diffusion_steps}{VOICE_TAG}"
    sep = "" if not ARGS.no_karaoke else "nk"
    return f"{safe}@{start_txt}+{duration:.0f}_v{CACHE_VERSION}{sep}{setting}.wav"


def known_start(stem):
    hit = list_key(stem, read_list("片段.txt"))
    if hit:
        try:
            return parse_time(hit[1])
        except ValueError:
            log.warning("片段.txt 里的起点看不懂：%s", hit)
    return None


def toward_ref(med, voice):
    """把中位音高 med 拉到这个歌手的中位音高附近要升降几个半音。"""
    import math
    target = ARGS.furina_median if voice == "furina" else ENGINE.ref_median_hz
    s = round(12 * math.log2(target / med))
    return max(-ARGS.max_shift, min(ARGS.max_shift, s))


def decide_semitone(stem, pitch, voice):
    """返回 (半音, None, 硬唱原因或 None) 或 (None, (原因, 说明), None)。"""
    wl = list_key(stem, read_list("白名单.txt"))
    if wl and wl[1]:
        try:
            return int(wl[1]), None, None
        except ValueError:
            pass
    med = pitch["median_hz"]
    if pitch["voiced_ratio"] < 0.2 or med <= 0:
        return None, ("no_vocals", "这一段几乎没有人声"), None
    if ARGS.ok_low <= med <= ARGS.ok_high:
        return 0, None, None
    if ARGS.ok_high < med <= ARGS.shift_high:
        return ARGS.shift_semitones, None, None
    if wl:  # 白名单但音高超出范围：照唱，靠近她的音域
        return toward_ref(med, voice), None, None
    kind = "too_high" if med > ARGS.shift_high else "too_low"
    if not ARGS.no_stretch:
        return toward_ref(med, voice), None, kind
    if kind == "too_high":
        return None, ("too_high", f"原唱太高了（中位音高约 {med:.0f} Hz），她的嗓子够不着"), None
    return None, ("too_low", f"原唱太低了（中位音高约 {med:.0f} Hz），像是男声歌，她唱不出味道"), None


def learn_step(job) -> bool:
    """学歌：按 --learn-source 去 go-music-dl（多源）或 B站下载音频到 原曲/，把任务改成普通点歌。
    失败时直接把任务标成失败。"""
    from bili import LearnError

    info = job["_learn"]
    query = f"{info['title']} {info['artist']}".strip()
    use_bili = ARGS.learn_source == "bili"
    job["stage"] = "学歌（B站搜索下载）" if use_bili else "学歌（多源搜索下载）"
    try:
        if use_bili:
            from bili import Bili
            path, video = Bili().learn(info["title"], info["artist"], LIB / "原曲", log=log.info)
            found = f"B站 {video['bvid']}《{video['title']}》"
        else:
            from musicdl import MusicDL, SOURCE_NAMES
            md = MusicDL()
            if info.get("pick"):  # 点歌的人确认过的版本，照着下，不换
                path, song = md.download_pick(info["pick"], info["title"], info["artist"], LIB / "原曲", log=log.info)
            else:
                path, song = md.learn(info["title"], info["artist"], LIB / "原曲", log=log.info)
            found = f"{SOURCE_NAMES.get(song.get('source'), song.get('source'))} {song['id']}《{song['name']}》{song['artist']}"
    except LearnError as e:
        log_learn(query, f"失败：{e}")
        record_wish(info["title"], info["artist"])  # 学不成就记进想学清单，等管理员手动加
        if e.reason == "blocked":  # B站拒绝了：暂停学歌，不重试
            STATE.learn_blocked_until = time.time() + LEARN_BLOCK_SECONDS
            log.warning("B站拒绝访问，%d 分钟内不再学歌", LEARN_BLOCK_SECONDS // 60)
        fail(job, f"learn_{e.reason}", str(e))
        return False
    log_learn(query, f"{path.stem}（{found}）")
    stem = path.stem
    job["_learned_file"] = str(path)
    if list_key(stem, read_list("黑名单.txt")):
        fail(job, "blacklisted", f"《{split_stem(stem)[1]}》在黑名单里")
        return False
    start = job["start"] if job["start"] is not None else known_start(stem)
    job.update(song=stem, source="原曲", _path=str(path), start=start,
               _key=cache_name(stem, start, job["duration"], job["voice"]))
    return True


JOB_TMP = RUN_DIR / "sing_tmp"  # 和子进程交换音频用的临时文件


def _db(x) -> float:
    import numpy as np
    return float(10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))


def separate_vocals(stereo, job):
    """双声道原曲 → 双声道人声（Roformer，还带和声）。分离进程出问题时用 demucs 兜底（单声道复制成双声道）。"""
    import numpy as np
    import soundfile as sf
    from sing_engine import SR

    JOB_TMP.mkdir(parents=True, exist_ok=True)
    src, dst = JOB_TMP / f"{job['job_id']}_mix.wav", JOB_TMP / f"{job['job_id']}_vocals.wav"
    try:
        sf.write(str(src), stereo, SR, subtype="FLOAT")
        SEP.call({"op": "vocals", "input": str(src), "output": str(dst)}, timeout=120 + 2 * len(stereo) / SR)
        out, sr = sf.read(str(dst), dtype="float32", always_2d=True)
        if sr != SR:
            raise RuntimeError(f"分离结果采样率是 {sr}")
        job["separation"] = "Roformer 人声"
        return out[:, :2] if out.shape[1] >= 2 else np.repeat(out, 2, axis=1)
    except Exception as e:  # noqa: BLE001
        log.warning("分离进程出错（%s），这次用 demucs 兜底", e)
        job["separation"] = "demucs 兜底"
        v = ENGINE.separate(stereo)
        return np.stack([v, v], axis=1)
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


def lead_vocals(segment, job):
    """双声道人声 → 单声道主唱（去和声）。去和声后主唱小了太多（模型把主唱当成了和声，
    2026-11-10 唯一 0:15 就这样），或者去和声出错，就退回全部人声。"""
    import soundfile as sf
    from sing_engine import SR

    everything = segment.mean(axis=1)
    if ARGS.no_karaoke:
        return everything
    tag = job["job_id"]
    src, lead, back = (JOB_TMP / f"{tag}_{n}.wav" for n in ("seg", "lead", "backing"))
    try:
        sf.write(str(src), segment, SR, subtype="FLOAT")
        SEP.call({"op": "lead", "input": str(src), "output": str(lead), "backing": str(back)},
                 timeout=120 + 2 * len(segment) / SR)
        main_v = sf.read(str(lead), dtype="float32", always_2d=True)[0].mean(axis=1)
        side_v = sf.read(str(back), dtype="float32", always_2d=True)[0].mean(axis=1)
        n = min(len(main_v), len(everything))
        main_v = main_v[:n]
        lost = _db(everything[:n]) - _db(main_v)
        if lost > ARGS.lead_min_db:
            log.info("去和声后主唱小了 %.1f dB，像是把主唱当成了和声，退回全部人声", lost)
            job["separation"] += f"，去和声失败（主唱小了 {lost:.0f} dB），用全部人声"
            return everything
        job["separation"] += f"，去和声（和声比主唱低 {_db(main_v) - _db(side_v[:n]):.0f} dB）"
        return main_v
    except Exception as e:  # noqa: BLE001
        log.warning("去和声出错（%s），用全部人声", e)
        job["separation"] += "，去和声出错，用全部人声"
        return everything
    finally:
        for f in (src, lead, back):
            f.unlink(missing_ok=True)


def rvc_convert(vocals, semitone, tag):
    """单声道主唱（44.1 kHz）→ 芙宁娜 RVC → 44.1 kHz。"""
    import soundfile as sf
    import soxr
    from sing_engine import SR

    JOB_TMP.mkdir(parents=True, exist_ok=True)
    src, dst = JOB_TMP / f"{tag}_rvc_in.wav", JOB_TMP / f"{tag}_rvc_out.wav"
    try:
        sf.write(str(src), vocals, SR, subtype="FLOAT")
        RVC.call({"input": str(src), "output": str(dst), "semitone": int(semitone)},
                 timeout=120 + len(vocals) / SR)
        out, sr = sf.read(str(dst), dtype="float32", always_2d=True)
        out = out.mean(axis=1)
        return soxr.resample(out, sr, SR) if sr != SR else out
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


def auto_start(stem, whole, dur, job):
    """没指定起点时唱哪段：先按歌词找副歌、挑最好唱的那遍（chorus.py）；找不到歌词就挑人声最响的一段。"""
    import numpy as np
    from sing_engine import SR, loudest_window

    job["stage"] = "挑副歌"
    try:
        import chorus
        import torchaudio

        torch = ENGINE.torch

        def f0_track(mono):
            with torch.no_grad():
                w16 = torchaudio.functional.resample(
                    torch.from_numpy(np.ascontiguousarray(mono)).float()[None], SR, 16000)
                return ENGINE.f0_fn(w16[0], thred=0.03)

        got = chorus.pick_start(stem, whole, f0_track, LIB / "歌词", length=dur, log=log.info)
        if got:
            job["start_note"] = got[1]
            return got[0]
    except Exception as e:  # noqa: BLE001
        log.warning("挑副歌出错（%s），改挑人声最响的一段", e)
    job["start_note"] = "没找到歌词，挑的人声最响的一段"
    return loudest_window(whole.mean(axis=1), dur)


def run_job(job):
    """在工作线程里跑：（学歌 →）截取 → 分离 → 去和声 → 检查难度 → 转换 → 写缓存。"""
    import numpy as np
    import soundfile as sf
    from sing_engine import SR, decode, finish, loudest_window, probe_duration

    t0 = time.time()
    job.update(status="running", started=t0)
    try:
        if ENGINE is None:
            raise RuntimeError(f"模型没有加载成功：{STATE.error}")
        if job.get("_learn") and not learn_step(job):
            return
        path, kind, stem = job["_path"], job["source"], job["song"]
        dur = job["duration"]
        start = job["start"]

        voice, tag = job["voice"], job["job_id"]
        job["stage"] = "截取"
        if kind == "清唱":  # 本来就是干声，不分离
            if start is None:
                whole = decode(path, channels=1)[:, 0]
                start = loudest_window(whole, dur)
                vocals = whole[int(start * SR):int((start + dur) * SR)]
            else:
                vocals = decode(path, start=start, duration=dur, channels=1)[:, 0]
            job["separation"] = "清唱源，不用分离"
        else:
            full = SEP_CACHE / f"{stem}.flac"  # 整首分离好的双声道人声（还带和声）
            whole = None
            if full.exists():
                whole, _ = sf.read(str(full), dtype="float32", always_2d=True)
                if whole.shape[1] != 2:  # 2026-11-10 以前 demucs 存的单声道，重新分
                    whole = None
            if start is None or whole is not None:
                if whole is None:
                    job["stage"] = "分离（整首）"
                    log.info("整首分离：%s（%.0f 秒）", stem, probe_duration(path))
                    whole = separate_vocals(decode(path), job)
                    SEP_CACHE.mkdir(parents=True, exist_ok=True)
                    sf.write(str(full), whole, SR, format="FLAC")
                os.utime(full)
                job["separation"] = job["separation"] or "Roformer 人声（整首，已缓存）"
                if start is None:
                    start = auto_start(stem, whole, dur, job)
                segment = whole[int(start * SR):int((start + dur) * SR)]
            else:
                job["stage"] = "分离"
                segment = separate_vocals(decode(path, start=start, duration=dur), job)
            job["stage"] = "去和声"
            vocals = lead_vocals(segment, job)
        job["start"] = round(float(start), 1)
        if len(vocals) < SR * 3:
            return fail(job, "too_short", "截出来的片段太短了，起点可能超过了歌曲长度")

        job["stage"] = "检查难度"
        pitch = ENGINE.pitch_stats(vocals)
        job["pitch"] = {k: round(v, 2 if k == "voiced_ratio" else 0) for k, v in pitch.items()}
        semitone, problem, stretch = decide_semitone(stem, pitch, voice)
        if problem:
            return fail(job, *problem)
        job["semitone"] = semitone
        if stretch:  # 插件轮询时看到就先说一句“有点高/低，可能唱不好”
            job["stretch"] = stretch
            log.info("硬唱｜%s｜%s，中位 %.0f Hz，升降 %+d", stem, stretch, pitch["median_hz"], semitone)

        job["stage"] = f"转换（{VOICES[voice]['label']}）"
        out = finish(rvc_convert(vocals, semitone, tag) if voice == "furina" else ENGINE.convert(vocals, semitone))
        if not np.isfinite(out).all():
            raise RuntimeError("转换结果里有无效数值")
        dest = CACHE / job["_key"]
        tmp = dest.with_suffix(".tmp")
        sf.write(str(tmp), out, SR, subtype="PCM_16", format="WAV")
        os.replace(tmp, dest)
        (dest.with_suffix(".json")).write_text(json.dumps(
            {"song": stem, "source": kind, "start": job["start"], "duration": dur, "voice": voice,
             "voice_label": VOICES[voice]["label"], "separation": job["separation"], "start_note": job.get("start_note"),
             "semitone": semitone, "pitch": job["pitch"], "stretch": stretch}, ensure_ascii=False))
        prune_cache()
        elapsed = time.time() - t0
        if not job.get("_learn"):  # 学歌那几分钟不算进平均，免得把普通点歌的估计拉长
            STATE.avg_job_seconds[voice] = 0.7 * STATE.avg_job_seconds[voice] + 0.3 * elapsed
        job.update(status="done", stage="完成", file=str(dest), finished=time.time())
        log.info("完成 %.0f 秒｜%s｜%s｜起点 %.0f｜升降 %+d｜中位 %.0f Hz｜%s", elapsed, stem,
                 VOICES[voice]["label"], start, semitone, pitch["median_hz"], job["separation"])
    except Exception as e:  # noqa: BLE001
        log.exception("任务出错")
        fail(job, "error", f"{type(e).__name__}: {e}")
    finally:
        if job.get("_learned_file") and job["status"] == "failed" and job["reason"] in UNSINGABLE:
            try:
                drop_learned(job)
            except OSError:
                log.exception("删除学歌下载的音频出错")
        STATE.last_active = time.monotonic()  # 闲置从做完算起，免得刚做完还没被取走就退出


def prune_cache():
    files = [f for f in CACHE.rglob("*") if f.is_file() and f.suffix in (".wav", ".flac")]
    files.sort(key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    limit = ARGS.cache_max_gb * 1024 ** 3
    while files and total > limit:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink(missing_ok=True)
        f.with_suffix(".json").unlink(missing_ok=True)


def cached_result(key, touch=True):
    f = CACHE / key
    if not f.exists():
        return None
    if touch:
        os.utime(f)
    meta = {}
    try:
        meta = json.loads(f.with_suffix(".json").read_text())
    except (OSError, ValueError):
        pass
    return f, meta


# ---------------------------------------------------------------- 后台 ----
def _load_engine():
    global ENGINE
    t = time.perf_counter()
    try:
        from sing_engine import SingEngine

        ENGINE = SingEngine(ARGS.ref, diffusion_steps=ARGS.diffusion_steps, ref_seconds=ARGS.ref_seconds,
                            fp16=ARGS.fp16)
        STATE.load_seconds = time.perf_counter() - t
        STATE.status = "ready"
        log.info("模型加载完成，用时 %.1f 秒（参考中位音高 %.0f Hz）", STATE.load_seconds, ENGINE.ref_median_hz)
    except Exception as e:  # noqa: BLE001
        STATE.status, STATE.error = "error", f"{type(e).__name__}: {e}"
        log.exception("模型加载失败")


def _warm_workers():
    """后台线程：先把分离进程和芙宁娜 RVC 进程拉起来、加载好模型（各约 5–10 秒），第一首歌就不用等。
    失败也不要紧：真正用到时还会再试，分离还有 demucs 兜底。"""
    import threading

    def warm(w):
        if not w.available():
            log.warning("%s 进程的环境或脚本不在，用到时会失败", w.name)
            return
        try:
            w.warm()
        except Exception as e:  # noqa: BLE001
            log.warning("%s 进程没拉起来：%s", w.name, e)

    for w in (SEP, RVC):
        threading.Thread(target=warm, args=(w,), daemon=True).start()


def busy() -> bool:
    return any(j["status"] in ("queued", "running") for j in JOBS.values())


async def idle_watchdog():
    while True:
        await asyncio.sleep(15)
        if ARGS.idle_timeout <= 0 or busy() or STATE.status == "loading":
            continue
        idle = time.monotonic() - STATE.last_active
        if idle > ARGS.idle_timeout:
            log.info("闲置 %d 秒，自动退出", int(idle))
            SERVER.should_exit = True
            return


@asynccontextmanager
async def lifespan(app):
    for d in (LIB / "清唱", LIB / "原曲", LIB / "待导入", CACHE, RUN_DIR):
        d.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    if not ARGS.no_caffeinate and sys.platform == "darwin":
        subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    try:
        os.nice(5)  # 唱歌很吃 CPU，让聊天和其他程序优先
    except OSError:
        pass
    STATE.last_active = time.monotonic()
    log.info("启动：端口 %d，歌曲库 %s，闲置 %d 秒退出，默认歌手 %s", ARGS.port, LIB, ARGS.idle_timeout,
             VOICES[default_voice()]["label"])
    _warm_workers()
    asyncio.get_running_loop().run_in_executor(EXECUTOR, _load_engine)
    watchdog = asyncio.create_task(idle_watchdog())
    try:
        yield
    finally:
        watchdog.cancel()
        for w in (SEP, RVC):
            w.stop()
        try:
            if PID_FILE.read_text().strip() == str(os.getpid()):
                PID_FILE.unlink()
        except OSError:
            pass
        log.info("已退出")


# ---------------------------------------------------------------- HTTP ----
def build_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="大肥鱼唱歌服务", lifespan=lifespan)

    class SingRequest(BaseModel):
        song: str
        artist: Optional[str] = ""
        start: Optional[Union[str, float]] = None
        duration: Optional[float] = None
        learn: bool = False  # 库里没有时去学（插件按“谁能学歌”的设置决定）
        learn_daily_limit: int = 5
        suggest: bool = True  # 库里没有时先看歌单里有没有只差一两个字的歌（did_you_mean）
        confirmed: bool = False  # 学歌：False 时只搜索不下载，返回 confirm 让点歌的人确认版本
        pick: Optional[dict] = None  # 学歌：点歌的人确认过的那一首（confirm 返回的 candidate）
        voice: Optional[str] = None  # 谁来唱：furina / dafeiyu（中文名也认）；不写用默认歌手
        probe: bool = False  # 只看能不能唱，不接单：库里有就返回 found

    class BlacklistRequest(BaseModel):
        song: str
        reason: Optional[str] = ""

    class WishRequest(BaseModel):
        song: str
        artist: Optional[str] = ""

    class ClearRequest(BaseModel):
        all: bool = False

    @app.post("/sing")
    async def sing(req: SingRequest):
        STATE.last_active = time.monotonic()
        if STATE.status == "error":
            return {"status": "failed", "reason": "error", "message": f"唱歌模型没有加载成功：{STATE.error}"}
        try:
            start = parse_time(req.start)
        except ValueError:
            raise HTTPException(400, f"看不懂的起点：{req.start}（写 1:47 或秒数）")
        dur = min(ARGS.max_duration, max(5.0, req.duration or ARGS.default_duration))

        await asyncio.to_thread(import_inbox)
        found = find_song(req.song, req.artist or "")
        if not found:
            who = f"{req.artist}的" if req.artist else ""
            if req.suggest and (near := similar_songs(req.song, req.artist or "")):
                return {"status": "failed", "reason": "did_you_mean", "song": req.song, "suggestions": near,
                        "message": f"歌曲库里没有{who}《{req.song}》，但有相近的《{split_stem(near[0])[1]}》"}
            if not req.learn:
                return {"status": "failed", "reason": "not_found", "song": req.song,
                        "message": f"歌曲库里没有{who}《{req.song}》"}
            if not req.confirmed and ARGS.learn_source == "musicdl":
                return await asyncio.to_thread(preview_learning, req)
            return start_learning(req, dur, start)
        stem, kind, path = found
        bl = list_key(stem, read_list("黑名单.txt"))
        if bl:
            return {"status": "failed", "reason": "blacklisted", "song": stem,
                    "message": f"《{split_stem(stem)[1]}》在黑名单里" + (f"：{bl[1]}" if bl[1] else "")}
        if start is None:
            start = known_start(stem)

        if req.probe:  # 插件先问“想听谁唱”，这里只告诉它这首能唱、哪个歌手已经有现成的
            ready = [v for v in VOICES if cached_result(cache_name(stem, start, dur, v), touch=False)]
            return {"status": "found", "song": stem, "source": kind, "cached_voices": ready,
                    "default_voice": default_voice()}
        voice = pick_voice(req.voice)
        key = cache_name(stem, start, dur, voice)
        hit = cached_result(key)
        if hit:
            f, meta = hit
            log.info("缓存命中｜%s｜%s", stem, VOICES[voice]["label"])
            return {"status": "done", "cached": True, "song": stem, "file": str(f), **meta,
                    "voice": voice, "voice_label": VOICES[voice]["label"]}
        if key in ACTIVE and JOBS[ACTIVE[key]]["status"] in ("queued", "running"):
            return public(JOBS[ACTIVE[key]])
        pending = sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))
        if pending >= ARGS.max_queue:
            return {"status": "failed", "reason": "queue_full", "song": stem,
                    "message": f"已经有 {pending} 首在排队了，等会儿再点"}

        job = new_job(stem, kind, path, start, dur, key, voice)
        ACTIVE[key] = job["job_id"]
        log.info("接单｜%s｜%s（%s，起点 %s）", stem, VOICES[voice]["label"], kind,
                 "自动" if start is None else f"{start:.0f}s")
        asyncio.get_running_loop().run_in_executor(EXECUTOR, run_job, job)
        return public(job)

    def learn_refusal(title):
        """学歌前的通用检查（暂停中、今天到上限）。能学返回 None。"""
        if time.time() < STATE.learn_blocked_until:
            mins = int((STATE.learn_blocked_until - time.time()) // 60) + 1
            return {"status": "failed", "reason": "learn_blocked", "song": title,
                    "message": f"暂时不能学新歌，{mins} 分钟后再试"}
        return None

    def preview_learning(req):
        """只搜索不下载：把要学的版本告诉插件，让点歌的人确认（在工作线程外跑，搜索约几秒）。"""
        from bili import LearnError
        from musicdl import MusicDL, SOURCE_NAMES

        title, artist = req.song.strip(), (req.artist or "").strip()
        if refusal := learn_refusal(title):
            return refusal
        if learned_today() >= max(0, req.learn_daily_limit):
            return {"status": "failed", "reason": "learn_limit", "song": title,
                    "message": f"今天已经学了 {req.learn_daily_limit} 首新歌，到上限了"}
        md = MusicDL()
        try:
            song, title, artist = md.choose(title, artist)
        except LearnError as e:
            return {"status": "failed", "reason": f"learn_{e.reason}", "song": title, "message": str(e)}
        if not song:
            record_wish(req.song, req.artist or "")
            return {"status": "failed", "reason": "learn_not_found", "song": title,
                    "message": f"没找到像样的《{title}》（只有翻唱、伴奏、Live 之类，或者没有这首）"}
        stem = md.stem_for(song, title, artist)
        if bl := list_key(stem, read_list("黑名单.txt")):
            return {"status": "failed", "reason": "blacklisted", "song": stem,
                    "message": f"《{split_stem(stem)[1]}》在黑名单里" + (f"：{bl[1]}" if bl[1] else "")}
        if find_song(split_stem(stem)[1], split_stem(stem)[0]):  # 搜到的其实就是库里那首
            return {"status": "failed", "reason": "did_you_mean", "song": req.song, "suggestions": [stem],
                    "message": f"歌单里已经有《{split_stem(stem)[1]}》"}
        log.info("学歌预览｜%s → %s %s《%s》%s", req.song, SOURCE_NAMES.get(song["source"], song["source"]),
                 song["id"], song["name"], song["artist"])
        return {"status": "confirm", "song": stem, "title": title, "artist": artist,
                "source_name": SOURCE_NAMES.get(song["source"], song["source"]), "candidate": song,
                "message": f"找到{SOURCE_NAMES.get(song['source'], song['source'])}的《{song['name']}》- {song['artist']}"}

    def start_learning(req, dur, start):
        title, artist = req.song.strip(), (req.artist or "").strip()
        key = f"learn:{norm(title)}{norm(artist)}"
        if key in ACTIVE and JOBS[ACTIVE[key]]["status"] in ("queued", "running"):
            return public(JOBS[ACTIVE[key]])
        if time.time() < STATE.learn_blocked_until:
            mins = int((STATE.learn_blocked_until - time.time()) // 60) + 1
            return {"status": "failed", "reason": "learn_blocked", "song": title,
                    "message": f"B站暂时拒绝访问，{mins} 分钟后才能再学新歌"}
        busy_learning = sum(1 for j in JOBS.values() if j.get("_learn") and j["status"] in ("queued", "running"))
        if learned_today() + busy_learning >= max(0, req.learn_daily_limit):
            return {"status": "failed", "reason": "learn_limit", "song": title,
                    "message": f"今天已经学了 {req.learn_daily_limit} 首新歌，到上限了"}
        pending = sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))
        if pending >= ARGS.max_queue:
            return {"status": "failed", "reason": "queue_full", "song": title,
                    "message": f"已经有 {pending} 首在排队了，等会儿再点"}
        display = f"{artist}-{title}" if artist else title
        job = new_job(display, "学歌", "", start, dur, None, pick_voice(req.voice))
        job["_learn"] = {"title": title, "artist": artist, "pick": req.pick}
        job["stage"] = "排队（要先学歌）"
        ACTIVE[key] = job["job_id"]
        log.info("接单（学歌）｜%s", display)
        asyncio.get_running_loop().run_in_executor(EXECUTOR, run_job, job)
        return public(job)

    @app.post("/wish")
    async def wish(req: WishRequest):
        """插件确认这首歌最终没找到（各种拆法都试过）后调用，记进想学清单。"""
        record_wish(req.song, req.artist or "")
        return {"ok": True}

    @app.get("/sing/{job_id}")
    async def job_status(job_id: str):
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "没有这个任务（服务重启过的话任务会清空）")
        STATE.last_active = time.monotonic()
        return public(job)

    @app.get("/songs")
    async def songs():
        await asyncio.to_thread(import_inbox)
        starts, wl, bl = read_list("片段.txt"), read_list("白名单.txt"), read_list("黑名单.txt")
        out = []
        for stem, kinds in library_songs().items():
            artist, title = split_stem(stem)
            st = list_key(stem, starts)
            out.append({
                "song": stem, "artist": artist, "title": title, "sources": sorted(kinds),
                "start": st[1] if st else None,
                "whitelisted": bool(list_key(stem, wl)), "blacklisted": bool(list_key(stem, bl)),
            })
        return {"songs": out}

    @app.get("/health")
    async def health():
        return {
            "status": STATE.status, "error": STATE.error,
            "queue": sum(1 for j in JOBS.values() if j["status"] == "queued"),
            "running": next((public(j) for j in JOBS.values() if j["status"] == "running"), None),
            "idle_seconds": int(time.monotonic() - STATE.last_active), "idle_timeout": ARGS.idle_timeout,
            "model_load_seconds": round(STATE.load_seconds, 1),
            "avg_job_seconds": {v: round(s) for v, s in STATE.avg_job_seconds.items()},
            "uptime_seconds": int(time.time() - STATE.started),
            "library": str(LIB), "diffusion_steps": ARGS.diffusion_steps, "ref_seconds": ARGS.ref_seconds,
            "stretch": not ARGS.no_stretch, "ref": Path(ARGS.ref).name,
            "voices": {v: info["label"] for v, info in VOICES.items()}, "default_voice": default_voice(),
            "karaoke": not ARGS.no_karaoke,
            "workers": {w.name: w.alive() for w in (SEP, RVC)},
        }

    @app.post("/admin/clear_cache")
    async def clear_cache(req: ClearRequest = ClearRequest()):
        n = 0
        for f in list(CACHE.glob("*.wav")) + list(CACHE.glob("*.json")) + (
                list(SEP_CACHE.glob("*.flac")) if req.all else []):
            f.unlink(missing_ok=True)
            n += 1
        log.info("清缓存：删了 %d 个文件", n)
        return {"ok": True, "deleted": n}

    @app.post("/admin/blacklist")
    async def blacklist(req: BlacklistRequest):
        found = find_song(req.song)
        stem = found[0] if found else req.song.strip()
        with open(LIB / "黑名单.txt", "a", encoding="utf-8") as fh:
            fh.write(f"{stem} | {req.reason or '管理员拉黑'}\n")
        log.info("拉黑：%s", stem)
        return {"ok": True, "song": stem}

    @app.post("/shutdown")
    async def shutdown():
        log.info("收到退出请求")
        SERVER.should_exit = True
        return {"ok": True}

    return app


def main():
    import socket

    import uvicorn

    if not Path(ARGS.ref).exists():
        sys.exit(f"找不到参考歌声：{ARGS.ref}")
    global VOICE_TAG, RVC_TAG, SEP, RVC
    VOICE_TAG = voice_tag(ARGS.ref)
    with socket.socket() as s:
        if s.connect_ex((ARGS.host, ARGS.port)) == 0:
            sys.exit(f"端口 {ARGS.port} 已被占用：唱歌服务可能已经在运行")
    sys.path.insert(0, str(TOOLS))
    import hashlib

    import sing_workers

    SEP, RVC = sing_workers.separator(log.info), sing_workers.rvc(log.info)
    pth = TOOLS / "rvc-test" / "model" / "芙宁娜.pth"
    RVC_TAG = hashlib.md5(pth.read_bytes()).hexdigest()[:6] if pth.exists() else "none"
    global SERVER
    SERVER = uvicorn.Server(uvicorn.Config(build_app(), host=ARGS.host, port=ARGS.port,
                                           log_level="warning", access_log=False))
    SERVER.run()
    os._exit(0)  # 工作线程可能还在转换，不等它


if __name__ == "__main__":
    main()
