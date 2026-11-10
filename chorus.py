"""自动挑唱哪 30 秒：先用歌词找副歌（重复出现的段落），再在几遍副歌里挑最好唱的那遍。

用在唱歌服务里“点歌没写起点、片段.txt 里也没有”的时候（手写的起点永远优先）：
  1. 歌词：经本机 go-music-dl（/music/lyric）按“歌名 + 时长”找原版的逐行时间歌词，存到 歌曲库/歌词/
     （翻唱、Live 的时间轴对不上，所以时长要和库里的文件差 4 秒以内）
  2. 对齐：歌词时间轴和库里的文件可能差几秒，用人声的响度起伏对一下
  3. 副歌：连续 3 行以上、每行都在歌里出现 2 次以上的段落
  4. 好唱：每遍副歌的 30 秒里，和声多不多（人声的左右声道差）、高音多不多、转音密不密，
     再稍微偏向前面那遍（最后一遍副歌常常升调、和声也多）
找不到歌词或者看不出副歌，就返回 None，由唱歌服务退回“人声最响的一段”。
网易云自带的“副歌时间”接口要模拟客户端加密，不用（2026-11-10 定的）。
"""

import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

SR = 44100
SOURCES = ("netease", "qq", "kugou", "kuwo")  # 歌词齐全、时间轴靠谱的几个源
META_RE = re.compile(r"(作词|作曲|编曲|制作|监制|混音|母带|和声|吉他|贝斯|鼓|弦乐|录音|出品|发行|OP|SP|词|曲)\s*[:：]")
TIME_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
BAD_VERSION = ("live", "现场", "演唱会", "音乐会", "音乐节", "伴奏", "dj", "remix", "cover", "翻唱", "原唱",
               "片段", "女声版", "男声版", "加速", "降调", "升调")


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_·・.,，。!！?？'\"“”‘’()（）\[\]【】《》<>~～…]+", "", s or "").lower()


# ---------------------------------------------------------------- 歌词 ----
def fetch_lrc(stem: str, duration: float, lyric_dir: Path, log=print, max_age_days: float = 30):
    """返回 LRC 文本或 None。结果（包括“没找到”）存在 歌曲库/歌词/ 里，不重复去找。"""
    lyric_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[\\/:*?"<>|]', "_", stem)
    hit, miss = lyric_dir / f"{safe}.lrc", lyric_dir / f"{safe}.没找到"
    if hit.exists():
        return hit.read_text(encoding="utf-8")
    if miss.exists() and time.time() - miss.stat().st_mtime < max_age_days * 86400:
        return None
    try:
        import requests
        from musicdl import MusicDL
    except ImportError:
        return None
    artist, _, title = stem.partition("-")
    if not title:
        artist, title = "", stem
    nt, na = _norm(re.sub(r"[（(\[【].*?[)）\]】]", "", title)), _norm(artist)
    md = MusicDL()
    s = requests.Session()
    s.trust_env = False
    tried = []
    pool = []  # 所有源的候选先收齐，再统一挑：歌手对得上的优先（不在第一个源就定下来）
    for source in SOURCES:
        try:
            cands = md._search_one(f"{title} {artist}".strip(), source)
        except Exception:  # noqa: BLE001
            continue
        for c in cands:
            name = _norm(re.sub(r"[（(\[【].*?[)）\]】]", "", c["name"]))
            if name != nt or not c.get("duration") or abs(c["duration"] - duration) > 4:
                continue  # 歌名不一样、或者时长对不上（翻唱、Live、剪辑版）
            low = c["name"].lower() + " " + c["artist"].lower()
            if any(w in low for w in BAD_VERSION):
                continue  # 翻唱、现场版：时长一样，时间轴也对不上
            # 歌手名要整个对上（“王菲儿”不是王菲）；多个歌手用顿号、逗号、& 分开
            names = {_norm(x) for x in re.split(r"[、,，/&;；]| feat\.? | ft\.? ", c["artist"])}
            same_artist = bool(na) and na in names
            pool.append((same_artist, -abs(c["duration"] - duration), source, c))
    if any(x[0] for x in pool):
        pool = [x for x in pool if x[0]]  # 有歌手对得上的就只用它们
    else:
        pool = [x for x in pool if x[1] >= -2]  # 歌手名写法不同（惠特妮/Whitney）：时长要更准
    pool.sort(key=lambda x: (-(x[0] * 2 + x[1] * 0.1), SOURCES.index(x[2])))
    for same, _, source, c in pool[:4]:
        try:
            r = s.get(md.base + "/lyric", params={"id": c["id"], "source": source, "mode": "line"}, timeout=15)
            text = r.text if r.status_code == 200 else ""
        except Exception:  # noqa: BLE001
            text = ""
        tried.append(f"{source}:{c['name']}-{c['artist']}")
        if len(parse_lrc(text)) >= 8:
            hit.write_text(f"[ti:{c['name']}]\n[ar:{c['artist']}]\n[by:{source} {c['id']}]\n" + text, encoding="utf-8")
            log(f"歌词：{stem} ← {source} 《{c['name']}》{c['artist']}（{c['duration']:.0f} 秒）")
            return hit.read_text(encoding="utf-8")
    miss.write_text("试过：" + "；".join(tried) + "\n", encoding="utf-8")
    log(f"歌词：{stem} 没找到时长对得上的原版歌词（试过 {len(tried)} 个）")
    return None


def parse_lrc(text: str) -> list:
    """→ [(开始秒, 结束秒或 None, 这一行)]，去掉作词作曲这类信息行和空行。"""
    out = []
    for raw in (text or "").splitlines():
        stamps = [(int(m), float(s)) for m, s in TIME_RE.findall(raw)]
        line = TIME_RE.sub("", raw).strip()
        if not stamps or not line or META_RE.search(line) or re.search(r"[:：]", line):
            continue
        start = stamps[0][0] * 60 + stamps[0][1]
        end = stamps[-1][0] * 60 + stamps[-1][1] if len(stamps) > 1 else None  # go-music-dl 的“行”模式带结束时间
        out.append((start, end, line))
    out.sort()
    for i, (s, e, l) in enumerate(out):  # 没写结束时间的，用下一行开始（最多 8 秒）
        if e is None or e <= s:
            nxt = out[i + 1][0] if i + 1 < len(out) else s + 6
            out[i] = (s, min(nxt, s + 8), l)
    return out


def find_choruses(lines: list) -> list:
    """副歌：连续 3 行以上、每行在全歌出现 2 次以上（允许中间夹 1 行不重复的）。
    返回 [(开始秒, 结束秒, 平均重复次数)]，按时间排。"""
    keys = [_norm(l) for _, _, l in lines]
    count = Counter(k for k in keys if len(k) >= 2)
    rep = [count.get(k, 0) >= 2 for k in keys]
    runs, i = [], 0
    while i < len(lines):
        if not rep[i]:
            i += 1
            continue
        j, gap = i, 0
        while j + 1 < len(lines) and (rep[j + 1] or (gap == 0 and j + 2 < len(lines) and rep[j + 2])):
            if not rep[j + 1]:
                gap += 1
            j += 1
        n = sum(rep[i:j + 1])
        if n >= 3 and lines[j][1] - lines[i][0] >= 10:
            runs.append((lines[i][0], lines[j][1], float(np.mean([count[keys[k]] for k in range(i, j + 1) if rep[k]]))))
        i = j + 1
    return runs


# ---------------------------------------------------------------- 对齐与打分 ----
def align_offset(lines: list, vocals_mono: np.ndarray, max_shift: float = 6.0) -> float:
    """歌词时间 + 返回值 = 库里文件的时间。用 10 Hz 的人声响度和“歌词正在唱”的 0/1 序列做相关。"""
    hop = SR // 10
    n = len(vocals_mono) // hop
    if n < 50:
        return 0.0
    env = np.sqrt(np.mean(vocals_mono[: n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1))
    env = (env > np.percentile(env, 60)).astype(np.float64)  # 有人在唱 = 比较响的那 40%
    sing = np.zeros(n)
    for s, e, _ in lines:
        sing[int(s * 10):int(e * 10)] = 1
    env, sing = env - env.mean(), sing - sing.mean()
    best, best_c = 0.0, -1e9
    for k in range(-int(max_shift * 10), int(max_shift * 10) + 1):
        # 歌词第 t 帧对文件第 t+k 帧
        a, b = (sing[:n - k], env[k:]) if k >= 0 else (sing[-k:], env[:n + k])
        c = float(np.dot(a, b)) / max(1, len(a))
        if c > best_c:
            best, best_c = k / 10, c
    return best


def _tracks(whole_stereo: np.ndarray, f0_track):
    """整首算一次：每 0.1 秒的人声响度（dB）、左右差（和声多少）；每 10 毫秒的音高。"""
    hop = SR // 10
    n = len(whole_stereo) // hop
    mid = whole_stereo[: n * hop].mean(axis=1).astype(np.float64)
    side = ((whole_stereo[: n * hop, 0] - whole_stereo[: n * hop, 1]) / 2).astype(np.float64)
    pm = np.mean(mid.reshape(n, hop) ** 2, axis=1)
    ps = np.mean(side.reshape(n, hop) ** 2, axis=1)
    db = 10 * np.log10(pm + 1e-10)
    f0 = np.asarray(f0_track(whole_stereo.mean(axis=1)))
    return db, pm, ps, f0


def window_stats(t0: float, length: float, db, pm, ps, f0) -> dict:
    """从 t0 起 length 秒的难度：和声（左右差）、有声比例、音域、转音密度、响度、起点处是不是“变响了”。"""
    a, b = int(t0 * 10), int((t0 + length) * 10)
    loud = db[a:b]
    voiced_env = loud > np.percentile(db, 40)
    fa, fb = int(t0 * 100), int((t0 + length) * 100)
    f = f0[fa:fb]
    voiced = f > 1
    semi = 12 * np.log2(np.where(voiced, f, 1) / 440.0)
    both = voiced[1:] & voiced[:-1]
    jumps = np.abs(np.diff(semi))[both]
    vs = semi[voiced]
    before = db[max(0, a - 40):a]
    after = db[a:a + 40]
    return {
        "width": float(ps[a:b].sum() / (pm[a:b].sum() + 1e-12)),
        "voiced": float(voiced.mean()) if voiced.size else 0.0,
        "range": float(np.percentile(vs, 95) - np.percentile(vs, 5)) if vs.size > 20 else 0.0,
        "agility": float(np.mean(jumps > 0.6)) if jumps.size else 0.0,  # 每 10 毫秒变化超过 0.6 个半音的比例
        "loud": float(np.mean(loud[voiced_env])) if voiced_env.any() else -99.0,
        "rise": float(np.mean(after) - np.mean(before)) if before.size and after.size else 0.0,
    }


def pick_start(stem: str, whole_stereo: np.ndarray, f0_track, lyric_dir: Path, length: float = 30.0,
               log=print):
    """返回 (起点秒, 说明) 或 None。whole_stereo：整首分离好的双声道人声（44.1 kHz）。
    候选起点 = 每一行重复出现的歌词（副歌、副歌前的过渡句都会重复，所以逐行比）；
    打分看：30 秒里重复的歌词占多少、整体响不响、起点处是不是变响了（副歌一般这样进来）、好不好唱、靠不靠前。"""
    total = len(whole_stereo) / SR
    text = fetch_lrc(stem, total, lyric_dir, log=log)
    lines = parse_lrc(text) if text else []
    if len(lines) < 8:
        return None
    keys = [_norm(l) for _, _, l in lines]
    count = Counter(k for k in keys if len(k) >= 2)
    if not find_choruses(lines):
        log(f"副歌：{stem} 歌词里看不出重复的段落")
        return None
    offset = align_offset(lines, whole_stereo.mean(axis=1))
    db, pm, ps, f0 = _tracks(whole_stereo, f0_track)
    song_loud = float(np.median(db[db > np.percentile(db, 40)]))
    prev_of = {}  # 每一行歌词每次出现时，前一行是什么
    for i, k in enumerate(keys):
        prev_of.setdefault(k, set()).add(keys[i - 1] if i else "")

    def section_start(i):
        """这一行是不是“一段的开头”：前一行不重复、几次出现前面接的不一样（重复段从这里开始），
        或者唱之前有换气停顿（前 0.7 秒人声明显变小）。免得从半句开始唱。"""
        if i == 0 or count.get(keys[i - 1], 0) < 2 or len(prev_of[keys[i]]) > 1:
            return True
        t = lines[i][0] + offset
        a, b = int((t - 0.8) * 10), int((t - 0.1) * 10)
        return a >= 0 and b > a and float(np.mean(db[a:b])) < song_loud - 10

    cands = []
    for i, (s, e, line) in enumerate(lines):
        if count.get(keys[i], 0) < 2 or not section_start(i):
            continue
        t0 = s + offset - 0.3
        if t0 < 0 or t0 + length > total:
            continue
        cover = 0.0
        for (s2, e2, _), k2 in zip(lines, keys):
            ov = min(e2 + offset, t0 + length) - max(s2 + offset, t0)
            if ov > 0:
                cover += ov * min(count.get(k2, 1), 3) / 3
        st = window_stats(t0, length, db, pm, ps, f0)
        score = (1.2 * cover / length + 0.06 * (st["loud"] - song_loud) + 0.05 * max(-6.0, min(8.0, st["rise"]))
                 + 0.6 * st["voiced"] - 3.0 * st["width"] - 0.06 * max(0.0, st["range"] - 10)
                 - 1.2 * st["agility"] - 0.1 * t0 / total)
        cands.append((score, t0, line, st, cover / length))
    if not cands:
        return None
    score, start, line, st, cov = max(cands, key=lambda c: c[0])
    why = (f"从「{line}」开始（歌词对齐偏移 {offset:+.1f} 秒；重复歌词占 {cov:.0%}，响度比全曲 {st['loud'] - song_loud:+.1f} dB，"
           f"起点变响 {st['rise']:+.1f} dB，和声 {st['width']:.2f}，音域 {st['range']:.0f} 半音，转音 {st['agility']:.2f}）")
    log(f"副歌：{stem} 从 {int(start) // 60}:{int(start) % 60:02d} 唱，{why}")
    return round(start, 1), why
