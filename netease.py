"""网易云音乐：搜歌、取下载地址、下载。给唱歌服务的“学歌”用。

非官方接口（和 lx-music-source / pyncm 用的同一套 eapi）。为了少触发风控：
  - 只发两种请求：搜索、取播放地址；不调用登录、签到等接口
  - 两次请求之间随机停几秒；一次只学一首（唱歌服务的工作线程本来就是单线程）
  - 学过的歌永久存进歌曲库，不重复下载
Cookie（MUSIC_U）只在内存里用，不写日志、不落盘。
"""

import hashlib
import json
import random
import re
import time

import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

EAPI_KEY = b"e82ckenh8dichen8"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/91.0.4472.164 NeteaseMusicDesktop/2.10.2.200154 Safari/537.36")
# 搜到这些版本时扣分，除非点歌的人自己写了
VERSION_WORDS = ("live", "伴奏", "instrumental", "dj", "remix", "cover", "翻自", "片段", "铃声",
                 "钢琴", "吉他", "纯音乐", "karaoke", "inst", "demo", "现场", "加速", "降调", "升调", "0.8x", "1.2x")


class NeteaseError(Exception):
    """reason: not_found / need_vip / bad_cookie / network"""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_·・.,，。!！?？'\"“”‘’()（）\[\]【】《》<>/&+]+", "", s or "").lower()


def _eapi_params(path: str, payload: dict) -> dict:
    text = json.dumps(payload, separators=(",", ":"))
    digest = hashlib.md5(f"nobody{path}use{text}md5forencrypt".encode()).hexdigest()
    data = f"{path}-36cd479b6b5-{text}-36cd479b6b5-{digest}".encode()
    pad = 16 - len(data) % 16
    enc = Cipher(algorithms.AES(EAPI_KEY), modes.ECB()).encryptor()
    return {"params": (enc.update(data + bytes([pad]) * pad) + enc.finalize()).hex().upper()}


class Netease:
    def __init__(self, music_u: str = "", timeout: float = 15, pause=(2.0, 5.0)):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Referer": "https://music.163.com/"})
        cookie = "os=pc; appver=2.10.2.200154"
        if music_u:
            music_u = music_u.strip()
            if music_u.startswith("MUSIC_U="):
                music_u = music_u[len("MUSIC_U="):]
            cookie += f"; MUSIC_U={music_u}"
        self.s.headers["Cookie"] = cookie
        self.has_login = bool(music_u)
        self.timeout = timeout
        self.pause = pause

    def _eapi(self, path: str, payload: dict) -> dict:
        try:
            r = self.s.post(f"https://interface3.music.163.com/eapi{path[4:]}", data=_eapi_params(path, payload),
                            timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            raise NeteaseError("network", f"连不上网易云：{type(e).__name__}") from None

    def search(self, keyword: str, limit: int = 10) -> list:
        d = self._eapi("/api/cloudsearch/pc", {"s": keyword, "type": 1, "limit": limit, "offset": 0, "total": True})
        songs = (d.get("result") or {}).get("songs") or []
        return [{
            "id": s["id"],
            "name": s.get("name", ""),
            "artists": [a.get("name", "") for a in s.get("ar") or s.get("artists") or []],
            "album": (s.get("al") or s.get("album") or {}).get("name", ""),
            "duration": (s.get("dt") or s.get("duration") or 0) / 1000,
            "fee": s.get("fee"),
        } for s in songs]

    @staticmethod
    def pick(cands: list, title: str, artist: str = ""):
        """挑最像原版的一首：歌名对得上、歌手对得上、不是 Live/伴奏/翻唱。"""
        nt, na = _norm(title), _norm(artist)
        query = nt + na
        best, best_score = None, 4
        for rank, c in enumerate(cands):
            cn = _norm(c["name"])
            if not cn:
                continue
            if cn == nt:
                score = 10
            elif len(cn) >= 2 and cn in query:
                score = 7  # 点歌时把歌手写进了歌名，如“晴天周杰伦”
            elif len(nt) >= 2 and nt in cn:
                score = 3
            else:
                continue
            arts = [_norm(a) for a in c["artists"] if a]
            if any(a and (a in query or (na and na in a)) for a in arts):
                score += 5
            elif na:
                score -= 3  # 指定了歌手但对不上
            low = c["name"].lower() + " " + c["album"].lower()
            if any(w in low and w not in (title + artist).lower() for w in VERSION_WORDS):
                score -= 8
            if c["duration"] and c["duration"] < 90:
                score -= 5
            score -= rank * 0.3  # 网易云自己的排序也算一点
            if score > best_score:
                best, best_score = c, score
        return best

    def audio_url(self, song_id: int, br: int = 320000) -> dict:
        d = self._eapi("/api/song/enhance/player/url", {"ids": f"[{song_id}]", "br": br})
        if d.get("code") == 301:
            raise NeteaseError("bad_cookie", "网易云账号没登录上（Cookie 可能过期了）")
        item = (d.get("data") or [{}])[0]
        if not item.get("url"):
            raise NeteaseError("need_vip", "这首拿不到完整音频（可能要会员、版权受限或 Cookie 失效）")
        if item.get("freeTrialInfo"):
            raise NeteaseError("need_vip", "只拿到试听片段（会员歌，Cookie 可能失效了）")
        return {"url": item["url"], "type": (item.get("type") or "mp3").lower(), "br": item.get("br"),
                "size": item.get("size")}

    def download(self, url: str, dest) -> None:
        try:
            with self.s.get(url, stream=True, timeout=self.timeout * 4) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
                tmp.replace(dest)
        except requests.RequestException as e:
            raise NeteaseError("network", f"下载失败：{type(e).__name__}") from None

    def learn(self, title: str, artist: str, dest_dir, log=print) -> tuple:
        """搜索 → 挑版本 → 取地址 → 下载到 dest_dir/歌手-歌名.mp3。返回 (文件路径, 选中的歌)。"""
        keyword = f"{title} {artist}".strip()
        cands = self.search(keyword)
        song = self.pick(cands, title, artist)
        if not song:
            raise NeteaseError("not_found", f"网易云上没搜到《{title}》")
        log(f"学歌：选中 {song['name']} - {'/'.join(song['artists'])}（{song['album']}，id {song['id']}）")
        time.sleep(random.uniform(*self.pause))
        info = self.audio_url(song["id"])
        name = re.sub(r'[\\/:*?"<>|]', "_", f"{song['artists'][0] if song['artists'] else '未知'}-{song['name']}")
        dest = dest_dir / f"{name}.{info['type'] if info['type'] in ('mp3', 'flac', 'm4a') else 'mp3'}"
        self.download(info["url"], dest)
        return dest, song
