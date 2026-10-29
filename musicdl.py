"""通过本机的 go-music-dl（桌面版，127.0.0.1:37777）学歌。给唱歌服务的“学歌”用。

"""

import html
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from pathlib import Path

import requests

from bili import BAD_WORDS, SAME_KIND, LearnError, _has, _norm

BASE = "http://127.0.0.1:37777/music"
FFPROBE = "/opt/homebrew/bin/ffprobe"

ALLOWED_SOURCES = (
    "netease", "qq", "kugou", "kuwo", "migu", "qianqian",
    "soda", "joox", "apple", "bilibili", "fivesing", "jamendo",
)
SOURCE_NAMES = {
    "netease": "网易云", "qq": "QQ音乐", "kugou": "酷狗", "kuwo": "酷我",
    "migu": "咪咕", "qianqian": "千千", "soda": "汽水音乐", "joox": "JOOX",
    "apple": "Apple Music", "bilibili": "B站", "fivesing": "5sing", "jamendo": "Jamendo",
}
SOURCE_PRIORITY = {name: i for i, name in enumerate(ALLOWED_SOURCES)}
CARD_RE = re.compile(r'<li class="song-card"(.*?)>', re.S)
ATTR_RE = re.compile(r'data-([a-z-]+)="([^"]*)"|data-([a-z-]+)=\'([^\']*)\'')


def _probe_seconds(path: Path) -> float:
    try:
        out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out)
    except (ValueError, subprocess.SubprocessError, OSError):
        return 0.0


def _ext_of(data: bytes, content_type: str) -> str:
    if data[:4] == b"fLaC":
        return "flac"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    if data[4:8] == b"ftyp":
        return "m4a"
    return {"audio/flac": "flac", "audio/x-flac": "flac", "audio/mp4": "m4a"}.get(content_type.split(";")[0], "mp3")


class MusicDL:
    def __init__(self, base: str = BASE, timeout: float = 30, sources=ALLOWED_SOURCES):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.sources = [x for x in sources if x in ALLOWED_SOURCES]
        self.s = requests.Session()
        self.s.trust_env = False  # 本机服务，别走代理

    def _get(self, path: str, params: dict, timeout: float = None, stream: bool = False):
        try:
            return self.s.get(f"{self.base}{path}", params=params, timeout=timeout or self.timeout, stream=stream)
        except requests.ConnectionError:
            raise LearnError("service", "go-music-dl 没开（打开 music-dl-desktop-go 再试）") from None
        except requests.RequestException as e:
            raise LearnError("network", f"go-music-dl 没响应：{type(e).__name__}") from None

    def _search_one(self, keyword: str, source: str) -> list:
        s = requests.Session()
        s.trust_env = False
        try:
            r = s.get(f"{self.base}/search",
                      params={"q": keyword, "type": "song", "sources": source},
                      timeout=min(self.timeout, 15))
            if r.status_code != 200:
                return []
            text = r.text
        except requests.RequestException:
            return []
        out = []
        for m in CARD_RE.finditer(text):
            a = {}
            for k1, v1, k2, v2 in ATTR_RE.findall(m.group(1)):
                a[k1 or k2] = html.unescape(v1 or v2)
            if a.get("source") != source or not a.get("id"):
                continue
            out.append({"id": a["id"], "source": source, "name": a.get("name", ""),
                        "artist": a.get("artist", ""), "album": a.get("album", ""),
                        "extra": a.get("extra", ""),
                        "duration": float(a.get("duration") or 0)})
        return out

    def search(self, keyword: str) -> list:
        """并发查询所有已启用源，合并去重；单个源失败不影响其他源。"""
        sources = list(self.sources)
        if not sources:
            return []
        results = []
        ex = ThreadPoolExecutor(max_workers=min(6, len(sources)))
        futures = {ex.submit(self._search_one, keyword, s): s for s in sources}
        try:
            for fut in as_completed(futures, timeout=20):
                try:
                    results.extend(fut.result())
                except Exception:
                    continue
        except FuturesTimeoutError:
            pass
        finally:
            for fut in futures:
                fut.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
        seen, uniq = set(), []
        for c in results:
            key = (c.get("source"), c.get("id"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        return uniq

    @staticmethod
    def pick(cands: list, title: str, artist: str = ""):
        """歌名对得上（括号里的版本说明另算）；不要伴奏/Live/DJ 之类；歌手对得上的优先。"""
        nt, na = _norm(title), _norm(artist)
        asked = (title + " " + artist).lower()
        for group in SAME_KIND:
            if any(_has(w, asked) for w in group):
                asked += " " + " ".join(group)
        best, best_score = None, 5
        for rank, c in enumerate(cands):
            base_name = re.sub(r"[（(\[【].*?[)）\]】]", "", c["name"])
            if not nt or nt not in _norm(c["name"]):
                continue
            score = 10.0 + (4 if _norm(base_name) == nt else 0)
            if na and na not in _norm(c["artist"]):
                continue  # 网易云上同名翻唱很多，歌手对不上就不要
            low = c["name"].lower()
            note = re.findall(r"[（(\[【](.*?)[)）\]】]", c["name"])
            if any(not any(_has(w, n.lower()) for w in ("live", "现场") if _has(w, asked)) for n in note):
                score -= 6  # 括号里写着“氛围版”“女声版”之类的，多半不是原版
            score -= 10 * sum(1 for w in BAD_WORDS if _has(w, low) and not _has(w, asked))
            if c["duration"] and c["duration"] < 90:
                score -= 10
            score -= rank * 0.1 + SOURCE_PRIORITY.get(c.get("source"), len(ALLOWED_SOURCES)) * 0.3
            if score > best_score:
                best, best_score = c, score
        return best

    def download(self, song: dict, dest_dir: Path, stem: str) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        params = {"id": song["id"], "source": song["source"], "name": song["name"], "artist": song["artist"]}
        if song.get("extra"):
            params["extra"] = song["extra"]
        r = self._get("/download", params, timeout=self.timeout * 6)
        if r.status_code != 200 or len(r.content) < 100_000:
            raise LearnError("not_found", f"{SOURCE_NAMES.get(song['source'], song['source'])}没给这首的音频（{r.status_code}），可能要会员/Cookie 或没有版权")
        ext = _ext_of(r.content[:16], r.headers.get("Content-Type", ""))
        dest, tmp = dest_dir / f"{stem}.{ext}", dest_dir / f".{stem}.{ext}.part"
        tmp.write_bytes(r.content)
        got = _probe_seconds(tmp)
        want = song["duration"]
        if got < 90 or (want and got < want * 0.8):  # 试听片段一般 30~60 秒
            tmp.unlink(missing_ok=True)
            raise LearnError("trial", f"{SOURCE_NAMES.get(song['source'], song['source'])}只给了 {got:.0f} 秒的试听片段（整首 {want:.0f} 秒），要登录会员才能学")
        tmp.replace(dest)
        return dest

    def choose(self, title: str, artist: str = ""):
        """只搜索不下载：返回 (挑中的歌或 None, 歌名, 歌手)，给“下载前先和点歌的人确认”用。"""
        cands = self.search(f"{title} {artist}".strip())
        return self._choose_in(cands, title, artist)

    def _choose_in(self, cands: list, title: str, artist: str = ""):
        """没写歌手又带空格（“后来 刘若英”）时，先试着拆出歌手、要求歌手对得上；
        都拆不出来才把整句当歌名——否则会挑中标题里恰好带着歌手名的翻唱。"""
        words = title.split()
        if not artist and len(words) >= 2:
            for k in range(len(words) - 1, 0, -1):
                t, a = " ".join(words[:k]), " ".join(words[k:])
                song = self.pick(cands, t, a)
                if song:
                    return song, t, a
        return self.pick(cands, title, artist), title, artist

    @staticmethod
    def stem_for(song: dict, title: str, artist: str = "") -> str:
        """入库文件名“歌手-歌名”，和 learn() 里的算法一致。"""
        first_artist = artist if artist and _norm(artist) in _norm(song["artist"]) else \
            re.split(r"[,，、&/]", song["artist"])[0].strip()
        name = re.sub(r"\s*[（(](?!.*(live|现场)).*?[)）]\s*$", "", song["name"], flags=re.I).strip() or title
        return re.sub(r'[\\/:*?"<>|]', "_", f"{first_artist}-{name}" if first_artist else name).strip()

    def download_pick(self, song: dict, title: str, artist: str, dest_dir: Path, log=print) -> tuple:
        """点歌的人确认过的那一首：先下它；那个平台不给（404/试听）时，只换平台下同名同歌手的同一首。"""
        stem = self.stem_for(song, title, artist)
        src = SOURCE_NAMES.get(song["source"], song["source"])
        log(f"学歌：按确认的版本下载 {src} {song['id']}《{song['name']}》{song['artist']}")
        try:
            return self.download(song, dest_dir, stem), song
        except LearnError as e:
            if e.reason not in ("trial", "not_found"):
                raise
            log(f"学歌：{e}，换平台找同一首")
            last = e

        def base(n):
            return _norm(re.sub(r"[（(\[【].*?[)）\]】]", "", n))

        want_name = base(song["name"])
        want_artists = {_norm(a) for a in re.split(r"[,，、&/]", song["artist"]) if a.strip()}
        same = [c for c in self.search(f"{title} {artist}".strip())
                if (c["source"], c["id"]) != (song["source"], song["id"]) and base(c["name"]) == want_name
                and want_artists & {_norm(a) for a in re.split(r"[,，、&/]", c["artist"]) if a.strip()}
                and _norm(c["name"]) == _norm(song["name"])]  # 括号里的版本说明也要一样
        same.sort(key=lambda c: SOURCE_PRIORITY.get(c.get("source"), len(ALLOWED_SOURCES)))
        for c in same:
            log(f"学歌：改从{SOURCE_NAMES.get(c['source'], c['source'])}下同一首《{c['name']}》{c['artist']}")
            try:
                return self.download(c, dest_dir, stem), c
            except LearnError as e:
                if e.reason not in ("trial", "not_found"):
                    raise
                last = e
        raise last

    def learn(self, title: str, artist: str, dest_dir: Path, log=print) -> tuple:
        """搜索 → 挑版本 → 下载到 dest_dir/歌手-歌名.扩展名。返回 (文件路径, 选中的歌)。"""
        cands = self.search(f"{title} {artist}".strip())
        song, title, artist = self._choose_in(cands, title, artist)  # “/唱 晴天 周杰伦”：先拆出歌手
        if not song and not artist:
            raise LearnError("not_found", f"学新歌要带上歌手名，比如“{title} 歌手”，免得学成别人的翻唱")
        if not song:
            raise LearnError("not_found", f"没找到像样的《{title}》（只有翻唱、伴奏、Live 之类，或者没有这首）")
        # 先按整体最佳试一次，再按每个源各取一首最佳，逐源回退；避免在同一个源上反复撞 404。
        ordered = []
        first = self.pick(cands, title, artist)
        if first:
            ordered.append(first)
        for src_key in ALLOWED_SOURCES:
            sub = [c for c in cands if c.get("source") == src_key]
            if not sub:
                continue
            best = self.pick(sub, title, artist)
            if best and all((best["source"], best["id"]) != (x["source"], x["id"]) for x in ordered):
                ordered.append(best)

        last = None
        tried = []
        for song in ordered:
            src = SOURCE_NAMES.get(song["source"], song["source"])
            tried.append(src)
            log(f"学歌：选中{src} {song['id']}《{song['name']}》{song['artist']}，{song['duration']:.0f} 秒")
            first_artist = artist if artist and _norm(artist) in _norm(song["artist"]) else \
                re.split(r"[,，、&/]", song["artist"])[0].strip()
            name = re.sub(r"\s*[（(](?!.*(live|现场)).*?[)）]\s*$", "", song["name"], flags=re.I).strip() or title
            stem = re.sub(r'[\\/:*?"<>|]', "_", f"{first_artist}-{name}" if first_artist else name).strip()
            try:
                return self.download(song, dest_dir, stem), song
            except LearnError as e:
                if e.reason not in ("trial", "not_found"):
                    raise
                log(f"学歌：{e}")
                last = e
        if last:
            raise last
        raise LearnError("not_found", f"所有源都没找到能下载的《{title}》（试过：{'、'.join(tried)}）")


if __name__ == "__main__":
    import sys

    md = MusicDL()
    q = " ".join(sys.argv[1:]) or "后来 刘若英"
    res = md.search(q)
    for c in res[:10]:
        print(c["id"], c["name"], "|", c["artist"], "|", int(c["duration"]), "秒")
    t, _, a = q.partition(" ")
    print("挑中：", md.pick(res, t, a))
