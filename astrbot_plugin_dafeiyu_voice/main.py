"""大肥鱼的声音：带情绪的语音回复 + 清唱点歌。

依赖本机两个服务（~/dafeiyu-tools）：说话 127.0.0.1:9880、唱歌 127.0.0.1:9881。
服务没开时调用 dafeiyu-voice.sh 按需拉起，闲置后服务自己退出。

语音回复的流程（两个 on_decorating_result 钩子）：
  1. 最先执行：从文字里去掉【语气标签】，记下情绪，群友永远看不到标签
  2. 最后执行：决定这条要不要发语音（关键词/概率/长度/每分钟上限），
     要发就逐句合成、合成一句发一句，然后把文字从结果里拿掉（双发模式除外）
"""

import asyncio
import random
import re
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import aiohttp

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import TextPart
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from astrbot.core.utils.session_waiter import SessionController, SessionFilter, session_waiter

from .textproc import clean_for_tts, extract_tag, has_keyword, spoken_len, split_sentences, strip_tags

EMOTION_EXTRA = "dafeiyu_emotion"
MAX_SENTENCES = 6  # 说话服务最多同时排 6 句
TIME_RE = re.compile(r"^\d{1,2}[:：]\d{1,2}$|^\d+(?:秒|s)?$")
# 对方点名要语音时，只在这一轮告诉大模型：不然它会照常识回答“我不会发语音”，再被念出来
VOICE_HINT = ("（系统提示，不要复述：你这条回复会自动用你自己的声音念出来，以语音发给对方。"
              "直接说你想说的话，不要说自己不会或不能发语音，也不要提语音条、文字转语音这些。"
              "这次只说一两句，{n} 个字以内。）")
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".flac", ".aac", ".ogg"}
YES_RE = re.compile(r"^(是|是的|对|对的|对呀|对啊|嗯|嗯嗯|好|好的|行|可以|没错|就是|就这首|yes|y|ok)[!！~。.呀啊的]*$", re.I)
NO_RE = re.compile(r"^(不是|不对|不|不要|否|算了|取消|没|no|n)[!！~。.呀啊的]*$", re.I)
ASK_TIMEOUT = 60
UNSINGABLE = ("too_high", "too_low", "no_vocals", "too_short")


class SenderFilter(SessionFilter):
    """等回复时只认点歌的人本人（AstrBot 默认按整个群算，群里谁说“是”都会被当成答复）。"""

    def filter(self, event: AstrMessageEvent) -> str:
        return f"{event.unified_msg_origin}#{event.get_sender_id()}"


def song_title(stem: str) -> str:
    return stem.partition("-")[2] or stem


def song_artist(stem: str) -> str:
    return stem.partition("-")[0] if "-" in stem else ""


def mmss(seconds) -> str:
    s = int(seconds or 0)
    return f"{s // 60} 分 {s % 60:02d} 秒" if s else "时长未知"


class DafeiyuVoice(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._chats: dict | None = None  # 每个会话的 /语音 设置，存在插件 KV 里
        self._voice_times = defaultdict(deque)  # 会话 → 最近发语音的时间
        self._sing_user_last: dict = {}  # (会话, 用户) → 上次点歌时间
        self._sing_chat_times = defaultdict(deque)  # 会话 → 最近一小时点歌时间
        self._start_locks = {"tts": asyncio.Lock(), "sing": asyncio.Lock()}
        self._tasks: set = set()
        self._tmp = Path(get_astrbot_temp_path()) / "dafeiyu_voice"
        self._tmp.mkdir(parents=True, exist_ok=True)

    async def terminate(self):
        for t in list(self._tasks):
            t.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------ 工具函数 ----
    def cfg(self, key, default=None):
        return self.config.get(key, default)

    def refuse(self, reason: str, **fmt) -> str:
        texts = self.cfg("refuse_texts", {}) or {}
        builtin = {  # 配置里没有这几条时用
            "no_vocals": "这首好像没有人声（可能是纯音乐或伴奏版），我没法唱",
            "too_short": "这首截出来太短了，我没法唱",
        }
        if reason in builtin and not texts.get(reason):
            return builtin[reason]
        key = {"queue_full": "busy"}.get(reason, reason)
        if key.startswith("learn_") and key not in ("learn_limit", "learn_blocked"):
            key = "learn_failed"
        text = texts.get(key) or texts.get("error") or "呜…没唱出来"
        try:
            return text.format(**fmt)
        except (KeyError, IndexError):
            return text

    def _spawn(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def url(self, kind: str) -> str:
        key = "tts_url" if kind == "tts" else "sing_url"
        default = "http://127.0.0.1:9880" if kind == "tts" else "http://127.0.0.1:9881"
        return (self.cfg(key) or default).rstrip("/")

    async def health(self, kind: str):
        try:
            s = await self.http()
            async with s.get(f"{self.url(kind)}/health", timeout=aiohttp.ClientTimeout(total=3)) as r:
                return await r.json() if r.status == 200 else None
        except Exception:
            return None

    async def ensure(self, kind: str, timeout: float = 120) -> bool:
        """服务能连上（就绪或正在加载模型）就返回 True；没开且允许自动拉起时调用启停脚本。"""
        h = await self.health(kind)
        if h and h.get("status") in ("ready", "loading"):
            return True
        if not self.cfg("auto_start", True):
            return False
        async with self._start_locks[kind]:
            h = await self.health(kind)
            if h and h.get("status") in ("ready", "loading"):
                return True
            script = str(Path(self.cfg("control_script") or "~/dafeiyu-tools/dafeiyu-voice.sh").expanduser())
            logger.info(f"[大肥鱼声音] 拉起{'说话' if kind == 'tts' else '唱歌'}服务")
            try:
                proc = await asyncio.create_subprocess_exec(
                    "/bin/bash", script, "start", kind,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    start_new_session=True,  # 服务和 AstrBot 分开，AstrBot 重启不会带走它
                )
            except OSError as e:
                logger.error(f"[大肥鱼声音] 启停脚本运行失败：{e}")
                return False
            self._spawn(proc.wait())
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                h = await self.health(kind)
                if h and h.get("status") in ("ready", "loading"):
                    return True
                if h and h.get("status") == "error":
                    logger.error(f"[大肥鱼声音] 服务加载失败：{h.get('error')}")
                    return False
                await asyncio.sleep(1)
        logger.error("[大肥鱼声音] 等服务启动超时")
        return False

    async def chat_settings(self, umo: str) -> dict:
        if self._chats is None:
            self._chats = await self.get_kv_data("chats", {}) or {}
        return self._chats.get(umo, {})

    async def save_chat_settings(self, umo: str, **changes):
        cur = dict(await self.chat_settings(umo))
        cur.update(changes)
        self._chats[umo] = cur
        await self.put_kv_data("chats", self._chats)

    # ------------------------------------------------------------------ 说话 ----
    async def synth(self, text: str, emotion: str) -> Path:
        s = await self.http()
        async with s.post(f"{self.url('tts')}/tts", json={"text": text, "emotion": emotion},
                          timeout=aiohttp.ClientTimeout(total=180)) as r:
            if r.status != 200:
                raise RuntimeError(f"说话服务 {r.status}: {(await r.text())[:200]}")
            data = await r.read()
        path = self._tmp / f"{uuid.uuid4().hex}.wav"
        path.write_bytes(data)
        return path

    def _voice_rate_ok(self, umo: str) -> bool:
        q, now = self._voice_times[umo], time.time()
        while q and now - q[0] > 60:
            q.popleft()
        return len(q) < int(self.cfg("voice_per_minute", 3))

    async def _speak(self, event: AstrMessageEvent, sentences: list, emotion: str) -> bool:
        """合成一句发一句。一句都没发出去返回 False（让原文字照常发）。"""
        tasks = [asyncio.create_task(self.synth(s, emotion)) for s in sentences]
        sent = 0
        try:
            for i, task in enumerate(tasks):
                try:
                    path = await task
                except Exception as e:
                    logger.warning(f"[大肥鱼声音] 合成失败：{e}")
                    if sent == 0:
                        return False
                    await event.send(MessageChain(chain=[Comp.Plain("".join(sentences[i:]))]))
                    return True
                if sent:
                    await asyncio.sleep(random.uniform(0.6, 1.4))
                try:
                    await event.send(MessageChain(chain=[Comp.Record(file=str(path), url=str(path))]))
                except Exception as e:
                    logger.warning(f"[大肥鱼声音] 语音发送失败：{e}")
                    if sent == 0:
                        return False
                    await event.send(MessageChain(chain=[Comp.Plain("".join(sentences[i:]))]))
                    return True
                finally:
                    path.unlink(missing_ok=True)
                sent += 1
            return True
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    @filter.on_llm_request()
    async def tell_llm_it_speaks(self, event: AstrMessageEvent, req: ProviderRequest):
        """对方要语音时，告诉大模型这条会被念出来、说短一点。只在本轮生效，不写进聊天记录。"""
        if not self.cfg("voice_enable", True):
            return
        if not has_keyword(event.get_message_str(), self.cfg("force_keywords", []) or []):
            return
        umo = event.unified_msg_origin
        if not (await self.chat_settings(umo)).get("voice", True) or not self._voice_rate_ok(umo):
            return
        n = int(self.cfg("voice_hint_chars", 30))
        req.extra_user_content_parts.append(TextPart(text=VOICE_HINT.format(n=n)).mark_as_temp())

    @filter.on_decorating_result(priority=100)
    async def strip_emotion_tag(self, event: AstrMessageEvent):
        """最先执行：去掉【语气标签】并记下情绪，后面的插件和群友都看不到标签。"""
        result = event.get_result()
        if not result or not result.chain:
            return
        label = None
        for comp in result.chain:
            if isinstance(comp, Comp.Plain) and comp.text:
                label = label or extract_tag(comp.text)
                cleaned = strip_tags(comp.text)
                if cleaned != comp.text:
                    comp.text = cleaned.lstrip()
        if label:
            event.set_extra(EMOTION_EXTRA, label)

    @filter.on_decorating_result(priority=-100)
    async def voice_reply(self, event: AstrMessageEvent):
        """最后执行：按规则把大模型回复换成语音。"""
        try:
            result = event.get_result()
            if not result or not result.chain or not result.is_llm_result():
                return
            if not self.cfg("voice_enable", True):
                return
            umo = event.unified_msg_origin
            chat = await self.chat_settings(umo)
            if not chat.get("voice", True):
                return
            plains = [c for c in result.chain if isinstance(c, Comp.Plain) and c.text and c.text.strip()]
            if not plains:
                return
            text = "\n".join(c.text for c in plains)
            n = spoken_len(clean_for_tts(text))
            if n < 2:
                return
            forced = has_keyword(event.get_message_str(), self.cfg("force_keywords", []) or [])
            if n > int(self.cfg("force_max_chars" if forced else "max_voice_chars", 60)):
                return
            prob = float(chat.get("prob", self.cfg("voice_probability", 0.3)))
            if not forced and random.random() >= prob:
                return
            if not self._voice_rate_ok(umo):
                return
            sentences = split_sentences(text, max_len=int(self.cfg("sentence_max_chars", 40)))
            if not sentences or len(sentences) > MAX_SENTENCES:
                return
            if not await self.ensure("tts"):
                return
            emotion = event.get_extra(EMOTION_EXTRA) or "auto"
            if not await self._speak(event, sentences, emotion):
                return
            self._voice_times[umo].append(time.time())
            if not self.cfg("dual_output", False):
                result.chain = [c for c in result.chain if not isinstance(c, Comp.Plain)]
        except Exception:
            logger.exception("[大肥鱼声音] 语音回复出错，改发文字")

    # ------------------------------------------------------------------ 唱歌 ----
    def library(self) -> Path:
        return Path(self.cfg("song_library") or "~/dafeiyu-tools/歌曲库").expanduser()

    def local_songs(self) -> list:
        """歌曲库里的歌（去掉黑名单），不需要开唱歌服务。"""
        lib = self.library()
        stems = set()
        for kind in ("清唱", "原曲"):
            d = lib / kind
            if d.is_dir():
                stems.update(f.stem for f in d.iterdir() if f.suffix.lower() in AUDIO_EXT and not f.name.startswith("."))
        black = set()
        bl = lib / "黑名单.txt"
        if bl.exists():
            for line in bl.read_text(encoding="utf-8").splitlines():
                name = line.split("|")[0].strip()
                if name and not name.startswith("#"):
                    black.add(name)
        return sorted(s for s in stems if s not in black and s.partition("-")[2] not in black)

    def _sing_limit(self, event: AstrMessageEvent):
        """返回 None 表示可以点；否则返回 (原因代码, 给群友的话, 给大模型的说明)。"""
        if event.is_admin():
            return None
        umo, now = event.unified_msg_origin, time.time()
        cd = int(self.cfg("sing_cooldown_seconds", 120))
        last = self._sing_user_last.get((umo, event.get_sender_id()))
        if last and now - last < cd:
            wait = int(cd - (now - last))
            return "cooldown", self.refuse("cooldown", wait=wait), f"这个人刚点过歌，还要等 {wait} 秒才能再点"
        q = self._sing_chat_times[umo]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= int(self.cfg("sing_per_hour", 10)):
            return "hourly", self.refuse("hourly"), "这个群一小时内唱太多了，嗓子要休息"
        return None

    async def _wish(self, song: str, artist: str = ""):
        """这首最终没找到：记进 歌曲库/想学的歌.txt，管理员照着去下载或教她。"""
        try:
            s = await self.http()
            async with s.post(f"{self.url('sing')}/wish", json={"song": song, "artist": artist or ""},
                              timeout=aiohttp.ClientTimeout(total=10)):
                pass
        except Exception as e:
            logger.warning(f"[大肥鱼声音] 记想学清单失败：{e}")

    def _can_learn(self, event: AstrMessageEvent) -> bool:
        if not self.cfg("learn_enable", False):
            return False
        return event.is_admin() or not self.cfg("learn_admin_only", False)

    async def _request_song(self, event: AstrMessageEvent, song: str, artist: str = "", start=None,
                            learn: bool = False, suggest: bool = True, confirmed: bool = False,
                            pick: dict | None = None) -> dict:
        """向唱歌服务点歌。返回服务的回应，另加 refuse（给群友的话）；接单成功会在后台等结果并发送。
        learn=True 时，库里没有就去学（经本机 go-music-dl 多源：网易云/QQ/酷狗/酷我/咪咕/千千/汽水/JOOX/Apple/B站/5sing/Jamendo）。
        suggest=True 时，库里没有先返回 did_you_mean（歌单里只差一两个字的歌）；
        learn 且 confirmed=False 时只搜索不下载，返回 confirm（要学的版本），确认后带 confirmed+pick 再请求。"""
        if not (await self.chat_settings(event.unified_msg_origin)).get("sing", True):
            return {"status": "failed", "reason": "disabled", "refuse": self.refuse("disabled"),
                    "message": "这个群没有开放唱歌"}
        limited = self._sing_limit(event)
        if limited:
            reason, refuse, why = limited
            return {"status": "failed", "reason": reason, "refuse": refuse, "message": why}
        if not await self.ensure("sing"):
            return {"status": "failed", "reason": "service", "refuse": self.refuse("service")}
        body = {"song": song, "artist": artist or "", "duration": int(self.cfg("sing_duration", 30)),
                "learn": learn, "learn_daily_limit": int(self.cfg("learn_daily_limit", 5)),
                "suggest": suggest, "confirmed": confirmed, "pick": pick}
        if start:
            body["start"] = start
        try:
            s = await self.http()  # 学歌预览要先搜索，给足时间
            async with s.post(f"{self.url('sing')}/sing", json=body, timeout=aiohttp.ClientTimeout(total=90)) as r:
                resp = await r.json()
                if r.status != 200:
                    raise RuntimeError(resp)
        except Exception as e:
            logger.error(f"[大肥鱼声音] 点歌请求失败：{e}")
            return {"status": "failed", "reason": "service", "refuse": self.refuse("service")}

        if resp.get("status") == "failed":
            resp["refuse"] = self.refuse(resp.get("reason", "error"))
            return resp
        if resp.get("status") == "confirm":  # 还没下载，等点歌的人确认
            return resp
        umo = event.unified_msg_origin
        self._sing_user_last[(umo, event.get_sender_id())] = time.time()
        self._sing_chat_times[umo].append(time.time())
        self._spawn(self._deliver(umo, resp))
        return resp

    async def _ask_yes_no(self, event: AstrMessageEvent, question: str):
        """问点歌的人一个是/不是的问题，等他本人回复（别人说话不算）。返回 True/False，超时返回 None。"""
        await event.send(event.plain_result(question))
        answer = {}

        @session_waiter(timeout=ASK_TIMEOUT, record_history_chains=False)
        async def waiter(controller: SessionController, ev: AstrMessageEvent):
            text = re.sub(r"\s+", "", ev.message_str or "")
            if YES_RE.match(text):
                answer["v"] = True
                controller.stop()
            elif NO_RE.match(text):
                answer["v"] = False
                controller.stop()
            else:
                await ev.send(ev.plain_result("回「是」或者「不是」就好~"))
                controller.keep(timeout=ASK_TIMEOUT, reset_timeout=True)

        try:
            await waiter(event, SenderFilter())
        except TimeoutError:
            return None
        return answer.get("v")

    def _confirm_question(self, resp: dict) -> str:
        c = resp.get("candidate") or {}
        return (f"这首我还没学过~ 我在{resp.get('source_name', '')}找到的是《{c.get('name')}》- {c.get('artist')}"
                f"（{mmss(c.get('duration'))}），是这首吗？回「是」我就去学（大概 6 分钟），回「不是」就先不学")

    async def _deliver(self, umo: str, job: dict):
        """后台：等唱歌服务做完，发“开嗓”提示和清唱；做不了就发拒绝的话。"""
        try:
            deadline = time.monotonic() + 30 * 60
            misses = 0
            while job.get("status") not in ("done", "failed"):
                await asyncio.sleep(5)
                try:
                    s = await self.http()
                    async with s.get(f"{self.url('sing')}/sing/{job['job_id']}",
                                     timeout=aiohttp.ClientTimeout(total=10)) as r:
                        job = await r.json()
                    misses = 0
                except Exception:
                    misses += 1
                    if misses >= 6:  # 半分钟连不上：服务挂了
                        raise RuntimeError("唱歌服务没有响应")
                if time.monotonic() > deadline:
                    raise RuntimeError("等太久了")
            if job["status"] == "failed":
                logger.info(f"[大肥鱼声音] 唱不了 {job.get('song')}：{job.get('message')}")
                text = self.refuse(job.get("reason", "error"))
                if job.get("learning") and job.get("reason") in UNSINGABLE:
                    text += f"\n（《{song_title(job.get('song', ''))}》学下来发现唱不了，已经把它删掉啦，不占地方）"
                await self.context.send_message(umo, MessageChain(chain=[Comp.Plain(text)]))
                return
            tip = self.cfg("sing_start_tip", "咳咳，开嗓了~")
            if tip:
                await self.context.send_message(umo, MessageChain(chain=[Comp.Plain(tip)]))
                await asyncio.sleep(1.5)
            f = job["file"]
            await self.context.send_message(umo, MessageChain(chain=[Comp.Record(file=f, url=f)]))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[大肥鱼声音] 发送清唱失败：{e}")
            try:
                await self.context.send_message(umo, MessageChain(chain=[Comp.Plain(self.refuse("error"))]))
            except Exception:
                pass

    def _eta_text(self, job: dict) -> str:
        secs = int(job.get("eta_seconds") or 200)
        return f"{max(1, round(secs / 60))} 分钟"

    @filter.llm_tool(name="sing_song")
    async def sing_song(self, event: AstrMessageEvent, song_name: str, artist: str = "", confirmed=False) -> str:
        """让大肥鱼用自己的声音清唱一段歌。有人让你唱歌、点歌时调用。调用后立刻返回，清唱做好后会自动发到群里。
        工具可能先让你向对方确认歌名或版本；对方确认后再调用一次并带 confirmed=true。

        Args:
            song_name(string): 歌名，只写歌名本身，不要书名号
            artist(string): 原唱歌手；不确定就留空
            confirmed(boolean): 只有对方已经明确确认“就是这首”之后才填 true；第一次调用不要填
        """
        name = song_name.strip("《》 ")
        confirmed = str(confirmed).strip().lower() in ("true", "1", "yes", "是")
        resp = await self._request_song(event, name, artist or "", learn=self._can_learn(event),
                                        suggest=not confirmed, confirmed=confirmed)
        if resp.get("reason") == "did_you_mean":
            near = resp["suggestions"][0]
            t, a = song_title(near), song_artist(near)
            return (f"歌单里没有《{name}》，但有很像的《{t}》（{a or '歌手未知'}），对方可能打错或记错了。"
                    f"先用一句话问对方是不是想听《{t}》，现在不要说已经开唱。对方说是，就用 song_name=“{t}”、artist=“{a}” 再调用；"
                    f"对方说就是要《{name}》，再调用时 song_name 不变并带 confirmed=true。")
        if resp.get("status") == "confirm":
            c = resp.get("candidate") or {}
            return (f"《{name}》还没学过。准备学的版本是{resp.get('source_name', '')}的《{c.get('name')}》- {c.get('artist')}"
                    f"（{mmss(c.get('duration'))}），学一首大约 6 分钟。先用一句话问对方是不是这首，现在不要说已经在学；"
                    f"对方确认后再调用 sing_song，song_name、artist 不变，并带 confirmed=true。")
        if resp.get("reason") == "not_found":
            await self._wish(song_name.strip("《》 "), artist or "")
        status = resp.get("status")
        if status == "done":
            return f"《{song_name}》之前唱过，马上就会发出去。用一句话告诉对方马上开唱，不要把歌词打出来。"
        if status in ("queued", "running") and resp.get("learning"):
            return (f"《{song_name}》还没学过，已经开始现学，大约 {self._eta_text(resp)}后自动发出清唱"
                    "（学的时候可能发现太高唱不了，到时会另外说）。用一句话告诉对方你要先去练一下、让对方等等，不要把歌词打出来。")
        if status in ("queued", "running"):
            return (f"已经开始准备《{song_name}》，大约 {self._eta_text(resp)}后自动发出清唱。"
                    "用一句话让对方稍等，不要把歌词打出来。")
        songs = "、".join(s.partition("-")[2] or s for s in self.local_songs()[:8])
        why = {
            "disabled": "这个群没有开放唱歌",
            "not_found": "歌单里还没有这首歌，已经记进想学清单，等主人教你",
            "learn_limit": "今天学新歌学到上限了",
            "learn_blocked": "暂时学不了新歌，已经记进想学清单",
            "blacklisted": "这首唱不好，被拉黑了",
            "queue_full": "排队的歌太多了",
            "service": "唱歌服务现在打不开",
        }.get(resp.get("reason"), resp.get("message") or "唱不了")
        hint = (f"可以推荐歌单里的歌：{songs}。"
                if songs and resp.get("reason") in ("not_found", "blacklisted", "learn_limit", "learn_blocked") else "")
        return f"唱不了：{why}。用撒娇的语气拒绝对方，{hint}不要把歌词打出来。"

    @filter.command("唱")
    async def sing_cmd(self, event: AstrMessageEvent):
        """点歌：/唱 歌名 [歌手] [起点如 1:47]"""
        parts = re.sub(r"\s+", " ", event.get_message_str().strip()).split(" ")[1:]
        start = None
        if parts and TIME_RE.match(parts[-1]):
            start = parts.pop().replace("：", ":").rstrip("秒s")
        if not parts:
            yield event.plain_result("要唱什么呀？发 /唱 歌名 就好，比如 /唱 多远都要在一起")
            return
        # 歌名可能带空格：先整句当歌名找；找不到再把最后一段当歌手
        song, artist = " ".join(parts), ""
        if "-" in song and len(parts) == 1:
            artist, _, song = song.partition("-")
        resp = await self._request_song(event, song, artist, start, suggest=False)
        if resp.get("reason") == "not_found" and len(parts) >= 2:
            resp = await self._request_song(event, " ".join(parts[:-1]), parts[-1], start, suggest=False)
        if resp.get("reason") == "not_found":  # 再问一次，这次让服务找歌单里只差一两个字的歌
            resp = await self._request_song(event, song, artist, start)

        if resp.get("reason") == "did_you_mean":  # 多半是打错了：先问是不是歌单里那首
            near = resp["suggestions"][0]
            who = f"（{song_artist(near)}）" if song_artist(near) else ""
            then = "我再去找这首学" if self._can_learn(event) else "我把它记进想学清单"
            ok = await self._ask_yes_no(event, f"歌单里没有《{song}》，你是想听《{song_title(near)}》{who}吗？"
                                               f"回「是」我就唱这首，回「不是」{then}")
            if ok is None:
                yield event.plain_result("好久没等到回复，那先不唱啦~")
                return
            if ok:
                resp = await self._request_song(event, song_title(near), song_artist(near), start, suggest=False)
            else:
                resp = {"status": "failed", "reason": "not_found", "refuse": self.refuse("not_found")}

        if resp.get("reason") == "not_found" and self._can_learn(event):
            # 库里没有：先只搜索，把要学的版本给点歌的人确认，确认了才下载
            resp = await self._request_song(event, song, artist, start, learn=True, suggest=False)
            if resp.get("status") == "confirm":
                ok = await self._ask_yes_no(event, self._confirm_question(resp))
                if not ok:
                    await self._wish(song, artist)
                    yield event.plain_result(
                        "好~ 那先不学了。换个写法再点吧，带上歌手名会准很多，比如 /唱 多远都要在一起 邓紫棋"
                        if ok is False else "好久没等到回复，那先不学啦~")
                    return
                resp = await self._request_song(event, resp["title"], resp["artist"], start, learn=True,
                                                suggest=False, confirmed=True, pick=resp["candidate"])
        if resp.get("reason") == "not_found":
            await self._wish(song, artist)
        status = resp.get("status")
        if status == "done":
            return  # 后台马上发
        if status in ("queued", "running") and resp.get("learning"):
            yield event.plain_result(f"这首我还没学过，先去学一下~ 大概 {self._eta_text(resp)}后开唱")
            return
        if status in ("queued", "running"):
            ahead = int(resp.get("position") or 1) - 1
            queue = f"前面还有 {ahead} 首，" if ahead > 0 else ""
            yield event.plain_result(f"收到~ {queue}让我酝酿一下，大概 {self._eta_text(resp)}后开唱")
            return
        yield event.plain_result(resp.get("refuse") or self.refuse("error"))

    @filter.command("歌单")
    async def song_list(self, event: AstrMessageEvent):
        """看看大肥鱼会唱哪些歌"""
        if await self.health("sing"):  # 服务开着就问它（顺便把 待导入/ 里的新歌整理进来）
            try:
                s = await self.http()
                async with s.get(f"{self.url('sing')}/songs", timeout=aiohttp.ClientTimeout(total=60)) as r:
                    data = await r.json()
                songs = sorted(x["song"] for x in data.get("songs", []) if not x.get("blacklisted"))
            except Exception:
                songs = self.local_songs()
        else:
            songs = self.local_songs()
        if not songs:
            yield event.plain_result("我的歌单还是空的呢…")
            return
        lines = [f"{i}. {s}" for i, s in enumerate(songs, 1)]
        yield event.plain_result("我会唱这些~ 发 /唱 歌名 点歌\n" + "\n".join(lines))

    @filter.command_group("唱歌")
    def sing_group(self):
        """唱歌相关：状态、清缓存、拉黑"""

    @sing_group.command("状态")
    async def sing_status(self, event: AstrMessageEvent):
        """看唱歌服务在忙什么"""
        here = "" if (await self.chat_settings(event.unified_msg_origin)).get("sing", True) else "（本群已关闭唱歌）"
        h = await self.health("sing")
        if not h:
            yield event.plain_result(f"唱歌服务没开（有人点歌时会自动打开）{here}")
            return
        if h.get("status") == "loading":
            yield event.plain_result(f"唱歌服务正在启动…{here}")
            return
        run = h.get("running")
        now = f"正在做《{run['song']}》（{run['stage']}）" if run else "现在空闲"
        yield event.plain_result(f"{now}，还有 {h.get('queue', 0)} 首在排队{here}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @sing_group.command("开")
    async def sing_on(self, event: AstrMessageEvent):
        """本群开放唱歌（管理员）"""
        await self.save_chat_settings(event.unified_msg_origin, sing=True)
        yield event.plain_result("本群可以点歌了")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @sing_group.command("关")
    async def sing_off(self, event: AstrMessageEvent):
        """本群不唱歌（管理员）"""
        await self.save_chat_settings(event.unified_msg_origin, sing=False)
        yield event.plain_result("本群不唱歌了（语音回复另用 /语音 关）")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @sing_group.command("清缓存")
    async def sing_clear(self, event: AstrMessageEvent):
        """清掉做好的清唱缓存（管理员）"""
        if await self.health("sing"):
            s = await self.http()
            async with s.post(f"{self.url('sing')}/admin/clear_cache", json={},
                              timeout=aiohttp.ClientTimeout(total=30)) as r:
                n = (await r.json()).get("deleted", 0)
        else:
            n = 0
            for f in list((self.library() / "缓存").glob("*.wav")) + list((self.library() / "缓存").glob("*.json")):
                f.unlink(missing_ok=True)
                n += 1
        yield event.plain_result(f"清掉了 {n} 个缓存文件")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @sing_group.command("拉黑")
    async def sing_blacklist(self, event: AstrMessageEvent):
        """不再唱某首歌（管理员）：/唱歌 拉黑 歌名"""
        name = re.sub(r"^\S+\s+拉黑\s*", "", re.sub(r"\s+", " ", event.get_message_str().strip()))
        if not name:
            yield event.plain_result("要拉黑哪首？/唱歌 拉黑 歌名")
            return
        matches = [s for s in self.local_songs() if name in s]
        stem = matches[0] if len(matches) == 1 else name
        with open(self.library() / "黑名单.txt", "a", encoding="utf-8") as fh:
            fh.write(f"{stem} | 管理员拉黑\n")
        yield event.plain_result(f"好，以后不唱《{stem}》了")

    @filter.command_group("学歌")
    def learn_group(self):
        """点到没学过的歌时去学（管理员）：所有人、仅管理员、关、状态"""

    def _set_learn(self, enable: bool, admin_only: bool):
        self.config["learn_enable"] = enable
        self.config["learn_admin_only"] = admin_only
        if hasattr(self.config, "save_config"):
            self.config.save_config()  # 和网页上的插件配置是同一份

    @filter.permission_type(filter.PermissionType.ADMIN)
    @learn_group.command("所有人")
    async def learn_everyone(self, event: AstrMessageEvent):
        """谁点到没学过的歌都去学"""
        self._set_learn(True, False)
        yield event.plain_result(f"好~ 以后谁点到我没学过的歌，我都去学（每天最多 {self.cfg('learn_daily_limit', 5)} 首）")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @learn_group.command("仅管理员")
    async def learn_admin(self, event: AstrMessageEvent):
        """只有管理员点歌时才学新歌"""
        self._set_learn(True, True)
        yield event.plain_result("好~ 以后只有管理员能让我学新歌，其他人只能点我已经会的")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @learn_group.command("关")
    async def learn_off(self, event: AstrMessageEvent):
        """不再学新歌"""
        self._set_learn(False, bool(self.cfg("learn_admin_only", False)))
        yield event.plain_result("好~ 不学新歌了，只唱已经会的")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @learn_group.command("想学")
    async def learn_wishlist(self, event: AstrMessageEvent):
        """群友点过、但还没学会的歌（按点的次数排）"""
        f = self.library() / "想学的歌.txt"
        rows = [x for x in (f.read_text(encoding="utf-8").splitlines() if f.exists() else []) if x.strip()]
        rows = [r for r in rows if r.split("|")[0].strip() not in self.local_songs()][:15]
        if not rows:
            yield event.plain_result("想学清单是空的~")
            return
        body = "\n".join(f"{i}. {r.split('|')[0].strip()}（点了 {r.split('|')[1].strip()} 次）" for i, r in enumerate(rows, 1))
        yield event.plain_result("大家想听这些：\n" + body + "\n带歌手名发 /唱 歌名 歌手 让我学；或下载后放进 歌曲库/待导入 自动入库")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @learn_group.command("状态")
    async def learn_status(self, event: AstrMessageEvent):
        """学歌设置和今天学了几首"""
        if not self.cfg("learn_enable", False):
            mode = "关闭"
        else:
            mode = "仅管理员" if self.cfg("learn_admin_only", False) else "所有人"
        log_file = self.library() / "学歌记录.txt"
        lines = log_file.read_text(encoding="utf-8").splitlines() if log_file.exists() else []
        today = time.strftime("%Y-%m-%d")
        n = sum(1 for x in lines if x.startswith(today))
        recent = "\n".join(lines[-5:]) or "（还没学过）"
        yield event.plain_result(f"学新歌：{mode}，今天 {n}/{self.cfg('learn_daily_limit', 5)} 首\n最近：\n{recent}")

    # ------------------------------------------------------------------ 管理 ----
    @filter.command_group("语音")
    def voice_group(self):
        """本群语音回复设置（管理员）：开、关、概率、状态"""

    @filter.permission_type(filter.PermissionType.ADMIN)
    @voice_group.command("开")
    async def voice_on(self, event: AstrMessageEvent):
        """本群开启语音回复"""
        await self.save_chat_settings(event.unified_msg_origin, voice=True)
        yield event.plain_result("本群语音回复已开启")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @voice_group.command("关")
    async def voice_off(self, event: AstrMessageEvent):
        """本群关闭语音回复"""
        await self.save_chat_settings(event.unified_msg_origin, voice=False)
        yield event.plain_result("本群语音回复已关闭（点歌不受影响）")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @voice_group.command("概率")
    async def voice_prob(self, event: AstrMessageEvent, prob: float):
        """设置本群随机发语音的概率：/语音 概率 0.5"""
        prob = max(0.0, min(1.0, prob))
        await self.save_chat_settings(event.unified_msg_origin, prob=prob)
        yield event.plain_result(f"本群随机发语音的概率改成 {prob:g} 了")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @voice_group.command("状态")
    async def voice_status(self, event: AstrMessageEvent):
        """看本群语音设置和服务状态"""
        chat = await self.chat_settings(event.unified_msg_origin)
        on = chat.get("voice", True) and self.cfg("voice_enable", True)
        prob = chat.get("prob", self.cfg("voice_probability", 0.3))
        h = await self.health("tts")
        svc = "运行中" if h else "没开（需要时自动打开）"
        yield event.plain_result(
            f"本群语音回复：{'开' if on else '关'}，随机概率 {prob:g}，"
            f"每分钟最多 {self.cfg('voice_per_minute', 3)} 条，超过 {self.cfg('max_voice_chars', 60)} 字发文字\n"
            f"说话服务：{svc}")

    @filter.command_group("声音")
    def service_group(self):
        """手动开关说话和唱歌服务（管理员）：开、关、状态"""

    @filter.permission_type(filter.PermissionType.ADMIN)
    @service_group.command("开")
    async def service_on(self, event: AstrMessageEvent):
        """打开说话和唱歌服务"""
        ok_tts, ok_sing = await self.ensure("tts"), await self.ensure("sing")
        yield event.plain_result(f"说话服务：{'已打开' if ok_tts else '打开失败'}；唱歌服务：{'已打开' if ok_sing else '打开失败'}"
                                 "（闲置一段时间会自动关）")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @service_group.command("关")
    async def service_off(self, event: AstrMessageEvent):
        """关掉说话和唱歌服务"""
        script = str(Path(self.cfg("control_script") or "~/dafeiyu-tools/dafeiyu-voice.sh").expanduser())
        proc = await asyncio.create_subprocess_exec("/bin/bash", script, "stop", "all",
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        yield event.plain_result(out.decode(errors="ignore").strip() or "已关闭")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @service_group.command("状态")
    async def service_status(self, event: AstrMessageEvent):
        """看两个服务开没开"""
        def desc(h):
            if not h:
                return "没开"
            return {"ready": "运行中", "loading": "启动中", "error": f"出错：{h.get('error')}"}.get(h.get("status"), "未知")
        yield event.plain_result(f"说话服务：{desc(await self.health('tts'))}；唱歌服务：{desc(await self.health('sing'))}")
