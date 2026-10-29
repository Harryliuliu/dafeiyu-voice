"""B站：搜歌、挑最像原版的视频、下载音频。给唱歌服务的“学歌”用。

不登录，也不伪装成浏览器：B站对脚本访问会返回 412（限流/机器人检测），
遇到就立刻停下、报告 blocked，由唱歌服务暂停学歌一段时间，绝不重试或绕过。
"""

import hashlib
import html
import math
import random
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

UA = "dafeiyu-sing/0.1 (+personal bot; python-requests)"
# 标题或分区里出现这些多半不是原唱原版（除非点歌的人自己写了）
BAD_WORDS = ("翻唱", "cover", "翻自", "live", "现场", "演唱会", "伴奏", "纯音乐", "钢琴", "吉他", "古筝", "二胡",
             "口琴", "架子鼓", "唢呐", "八音盒", "尤克里里", "dj", "remix", "混音", "串烧", "合集", "混剪", "剪辑",
             "教学", "教程", "学唱", "男声版", "女声版", "戏腔", "改编", "抖音", "加速", "降调", "升调", "倍速",
             "0.8x", "1.2x", "片段", "铃声", "鬼畜", "ai", "干声", "清唱", "reaction", "反应", "伴唱", "跟唱")
SAME_KIND = (("live", "现场", "演唱会"), ("翻唱", "cover", "翻自"), ("伴奏", "纯音乐"), ("清唱", "干声"))
GOOD_WORDS = ("官方", "mv", "原唱", "无损", "hi-res", "hires", "录音棚", "动态歌词", "歌词", "高音质", "完整版", "音频")
BAD_TYPES = ("翻唱", "演奏", "鬼畜", "舞蹈", "VOCALOID", "人力")
# 播放接口的 wbi 参数签名，和 yt-dlp 里的 getMixinKey 一致
MIXIN_TAB = [46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49, 33, 9, 42, 19, 29, 28,
             14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21,
             56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52]


class LearnError(Exception):
    """reason: not_found / network / blocked（B站 412：调用方要停一阵，不能重试）"""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_·・.,，。!！?？'\"“”‘’()（）\[\]【】《》<>/&+|｜:：~]+", "", s or "").lower()


def _has(word: str, text: str) -> bool:
    if word.isascii():  # 英文按整词算，免得 "ai" 撞上 "again"
        return re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text) is not None
    return word in text


def _seconds(d) -> float:
    if isinstance(d, (int, float)):
        return float(d)
    total = 0
    for p in [int(p) for p in str(d or "0").split(":") if p.isdigit()]:
        total = total * 60 + p
    return float(total)


class Bili:
    def __init__(self, timeout: float = 15, pause=(1.0, 3.0)):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Referer": "https://www.bilibili.com/"})
        self.timeout = timeout
        self.pause = pause

    def _get_json(self, url: str, params: dict) -> dict:
        try:
            r = self.s.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as e:
            raise LearnError("network", f"连不上 B站：{type(e).__name__}") from None
        if r.status_code == 412:
            raise LearnError("blocked", "B站暂时拒绝了访问（412），要过一阵再学")
        try:
            r.raise_for_status()
            d = r.json()
        except (requests.RequestException, ValueError):
            raise LearnError("network", f"B站返回异常（{r.status_code}）") from None
        if d.get("code") in (-412, -352):
            raise LearnError("blocked", f"B站暂时拒绝了访问（{d.get('code')}），要过一阵再学")
        return d

    def search(self, keyword: str) -> list:
        d = self._get_json("https://api.bilibili.com/x/web-interface/search/type", {
            "search_type": "video", "keyword": keyword, "page": 1, "order": "totalrank"})
        if d.get("code") != 0:
            raise LearnError("network", f"B站搜索没成功（{d.get('code')}：{d.get('message')}）")
        out = []
        for v in (d.get("data") or {}).get("result") or []:
            out.append({
                "bvid": v.get("bvid", ""),
                "title": html.unescape(re.sub(r"<[^>]+>", "", v.get("title", ""))),
                "author": v.get("author", ""),
                "play": int(v.get("play") or 0) if str(v.get("play") or "0").isdigit() else 0,
                "duration": _seconds(v.get("duration")),
                "typename": v.get("typename", ""),
            })
        return out

    @staticmethod
    def pick(cands: list, title: str, artist: str = ""):
        """标题必须含歌名；歌手对得上、播放量高、不是翻唱/Live/伴奏/合集的优先。"""
        nt, na = _norm(title), _norm(artist)
        asked = (title + " " + artist).lower()
        for group in SAME_KIND:  # 点名要 Live，就连“现场”“演唱会”也不扣分
            if any(_has(w, asked) for w in group):
                asked += " " + " ".join(group)
        best, best_score = None, 5
        for rank, c in enumerate(cands):
            t = _norm(c["title"])
            if not nt or nt not in t:
                continue
            score = 10.0
            if na:
                score += 6 if (na in t or na in _norm(c["author"])) else -4
            low = (c["title"] + " " + c["typename"]).lower()
            score -= 10 * sum(1 for w in BAD_WORDS if _has(w, low) and not _has(w, asked))
            score += min(4, 2 * sum(1 for w in GOOD_WORDS if _has(w, low)))
            if any(bt.lower() in c["typename"].lower() for bt in BAD_TYPES):
                score -= 8
            if c["duration"] and c["duration"] < 90:
                score -= 10
            elif c["duration"] > 480:
                score -= 6  # 太长多半是合集或直播回放
            score += max(-1.0, min(4.0, math.log10(c["play"] + 1) - 3))  # 1 万播放 +1，100 万 +3
            score -= rank * 0.3
            if score > best_score:
                best, best_score = c, score
        return best

    def _signed(self, params: dict) -> dict:
        nav = self._get_json("https://api.bilibili.com/x/web-interface/nav", {})
        img = (nav.get("data") or {}).get("wbi_img") or {}
        lookup = "".join(u.rpartition("/")[2].partition(".")[0] for u in (img.get("img_url", ""), img.get("sub_url", "")))
        if len(lookup) < 64:
            raise LearnError("network", "B站签名信息拿不到")
        key = "".join(lookup[i] for i in MIXIN_TAB)[:32]
        params = dict(params, wts=round(time.time()))
        params = {k: "".join(ch for ch in str(v) if ch not in "!'()*") for k, v in sorted(params.items())}
        params["w_rid"] = hashlib.md5(f"{urlencode(params)}{key}".encode()).hexdigest()
        return params

    def download(self, bvid: str, dest_dir: Path, stem: str) -> Path:
        view = self._get_json("https://api.bilibili.com/x/web-interface/view", {"bvid": bvid})
        if view.get("code") != 0:
            raise LearnError("network", f"B站视频信息拿不到（{view.get('code')}：{view.get('message')}）")
        cid = view["data"]["cid"]
        time.sleep(random.uniform(*self.pause))
        play = self._get_json("https://api.bilibili.com/x/player/wbi/playurl",
                              self._signed({"bvid": bvid, "cid": cid, "fnval": 16, "fnver": 0, "fourk": 0}))
        if play.get("code") != 0:
            raise LearnError("network", f"B站播放地址拿不到（{play.get('code')}：{play.get('message')}）")
        audios = ((play.get("data") or {}).get("dash") or {}).get("audio") or []
        if not audios:
            raise LearnError("network", "这个视频没有单独的音频流")
        best = max(audios, key=lambda a: a.get("bandwidth") or 0)
        url = best.get("baseUrl") or best.get("base_url")
        dest, tmp = dest_dir / f"{stem}.m4a", dest_dir / f"{stem}.m4a.part"
        try:
            with self.s.get(url, stream=True, timeout=self.timeout * 4) as r:
                if r.status_code == 412:
                    raise LearnError("blocked", "B站暂时拒绝了访问（412），要过一阵再学")
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
        except requests.RequestException as e:
            tmp.unlink(missing_ok=True)
            raise LearnError("network", f"B站音频下载失败：{type(e).__name__}") from None
        tmp.replace(dest)
        return dest

    def learn(self, title: str, artist: str, dest_dir: Path, log=print) -> tuple:
        """搜索 → 挑版本 → 下载到 dest_dir/歌手-歌名.m4a。返回 (文件路径, 选中的视频)。"""
        cands = self.search(f"{title} {artist}".strip())
        video = self.pick(cands, title, artist)
        words = title.split()
        if not video and not artist and len(words) >= 2:
            # “/唱 晴天 周杰伦”：不知道哪段是歌手，从后往前试着拆
            for k in range(len(words) - 1, 0, -1):
                t, a = " ".join(words[:k]), " ".join(words[k:])
                video = self.pick(cands, t, a)
                if video:
                    title, artist = t, a
                    break
        if not video:
            raise LearnError("not_found", f"B站上没找到像样的《{title}》（只有翻唱、Live 之类）")
        log(f"学歌：选中 B站 {video['bvid']}《{video['title']}》UP：{video['author']}，"
            f"播放 {video['play']}，{video['duration']:.0f} 秒")
        time.sleep(random.uniform(*self.pause))
        stem = re.sub(r'[\\/:*?"<>|]', "_", f"{artist}-{title}" if artist else title).strip()
        return self.download(video["bvid"], dest_dir, stem), video
