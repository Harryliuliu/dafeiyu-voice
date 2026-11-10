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
from astrbot.api.event.filter import CustomFilter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import TextPart
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from astrbot.core.utils.session_waiter import SessionController, SessionFilter, session_waiter

from .textproc import clean_for_tts, extract_tag, has_keyword, spoken_len, strip_tags, voice_chunks

EMOTION_EXTRA = "dafeiyu_emotion"
MAX_SENTENCES = 6  # 说话服务最多同时排 6 句
TIME_RE = re.compile(r"^\d{1,2}[:：]\d{1,2}$|^\d+(?:秒|s)?$")
# 对方点名要语音时，只在这一轮告诉大模型：不然它会照常识回答“我不会发语音”，再被念出来。
# 2026-11-10：不再写死“一两句、30 字以内”（会和人格里的长度规则打架、让模型敷衍对方的要求），
# 长短交给人格；voice_hint_chars 设成大于 0 时才加字数提示。
VOICE_HINT = ("（系统提示，不要复述：你这条回复会自动用你自己的声音念出来，以语音发给对方。"
              "直接说你想说的话，不要说自己不会或不能发语音，也不要提语音条、文字转语音这些。"
              "对方点名要你说的内容就照着说完整；不要写动作描写和括号旁白。{limit}）")
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".flac", ".aac", ".ogg"}
YES_RE = re.compile(r"^(是|是的|对|对的|对呀|对啊|嗯|嗯嗯|好|好的|行|可以|没错|就是|就这首|yes|y|ok)[!！~。.呀啊的]*$", re.I)
NO_RE = re.compile(r"^(不是|不对|不|不要|否|算了|取消|没|no|n)[!！~。.呀啊的]*$", re.I)
ASK_TIMEOUT = 60
UNSINGABLE = ("too_high", "too_low", "no_vocals", "too_short")
# 说话模型二选一（dafeiyu-voice.sh engine 切换，记在 tts_engine.conf）
MODEL_NAMES = {"index": "IndexTTS2", "breeze": "Breeze TTS 2"}
MODEL_ALIASES = {"indextts": "index", "indextts2": "index", "breezetts": "breeze", "breezetts2": "breeze"}


def model_label(health: dict) -> str:
    """从说话服务的 /health 看正在跑的是哪个模型（IndexTTS2 版没有 engine 字段）。"""
    return MODEL_NAMES.get(health.get("engine") or "index", "未知模型")


# 唱歌的两个歌手（2026-11-10 起；雨爱已去掉）：芙宁娜本人（RVC）和特邀嘉宾大肥鱼（Seed-VC）。
# 点歌时先问对方想听谁唱；没回答就用默认歌手（/唱歌 音色 改，记在 sing_voice.conf，默认大肥鱼）
SINGERS = {"furina": "芙宁娜", "dafeiyu": "大肥鱼"}
SINGER_EXACT = {"芙宁娜": "furina", "芙芙": "furina", "furina": "furina",
                "大肥鱼": "dafeiyu", "肥鱼": "dafeiyu", "dafeiyu": "dafeiyu"}
# 回答“想听谁唱”时的说法；先看嘉宾，免得“你请嘉宾唱吧”被“你”认成芙宁娜
GUEST_RE = re.compile(r"大肥鱼|肥鱼|鱼|嘉宾|特邀|^2$|后者|后面那个")
SELF_RE = re.compile(r"芙宁娜|芙芙|芙|你|自己|本人|本女士|大导演|水神|亲自|^1$|前者|前面那个")
ANY_RE = re.compile(r"随便|都行|都可以|都好|无所谓|你定|你决定|看你|听你的|你安排")
# 点歌那句话里明确点了名才算“说了谁唱”（“你唱首泡沫”只是在跟她说话，不算选了芙宁娜）
NAMED_FURINA_RE = re.compile(r"芙宁娜|芙芙|本女士")
NAMED_GUEST_RE = re.compile(r"大肥鱼|肥鱼|嘉宾")


def parse_singer(text: str):
    """把“想听谁唱”的回答认成 furina / dafeiyu；认不出返回 None，“随便”返回 "any"。"""
    t = re.sub(r"\s+", "", text or "").lower()
    if not t:
        return None
    if t in SINGER_EXACT:
        return SINGER_EXACT[t]
    if ANY_RE.search(t):
        return "any"
    if GUEST_RE.search(t):
        return "dafeiyu"
    if SELF_RE.search(t):
        return "furina"
    return None


class LearnCmdFilter(CustomFilter):
    """“/学歌 歌名 [歌手] [起点]”：AstrBot 不会把指令组下没注册的子命令交给插件（会落到大模型），
    所以单独接；要求带唤醒词（/ 或 @），并排除 /学歌 自己的子命令。"""

    PATTERN = re.compile(r"^学歌\s+(?!(?:所有人|仅管理员|关|想学|状态)\s*$)\S")

    def filter(self, event: AstrMessageEvent, cfg) -> bool:
        return event.is_at_or_wake_command and bool(self.PATTERN.search(event.get_message_str().strip()))


class SenderFilter(SessionFilter):
    """等回复时只认点歌的人本人（AstrBot 默认按整个群算，群里谁说“是”都会被当成答复）。"""

    def filter(self, event: AstrMessageEvent) -> str:
        return f"{event.unified_msg_origin}#{event.get_sender_id()}"


def song_title(stem: str) -> str:
    return stem.partition("-")[2] or stem


def song_artist(stem: str) -> str:
    return stem.partition("-")[0] if "-" in stem else ""


def norm_name(s: str) -> str:
    """和唱歌服务的 norm 一样：去掉空格和标点、转小写，用来比歌名。"""
    return re.sub(r"[\s\-_·・.,，。!！?？'\"“”‘’()（）\[\]【】《》<>]+", "", s or "").lower()


def cache_prefix(stem: str) -> str:
    """唱歌服务缓存文件名的开头：“歌手-歌名@起点+时长_...wav”。"""
    return re.sub(r'[\\/:*?"<>|]', "_", stem) + "@"


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
        self._singer_asked: dict = {}  # (会话, 用户) → 大模型问过“想听谁唱”的时间
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
            "no_vocals": "这首好像是纯伴奏，没有人声，本女士没法照着唱",
            "too_short": "这一段截出来太短了，连一句都唱不完整",
        }
        if reason in builtin and not texts.get(reason):
            return builtin[reason]
        key = {"queue_full": "busy"}.get(reason, reason)
        if key.startswith("learn_") and key not in ("learn_limit", "learn_blocked"):
            key = "learn_failed"
        text = texts.get(key) or texts.get("error") or "呜……没唱成"
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
        """对方要语音时，告诉大模型这条会被念出来。只在本轮生效，不写进聊天记录。"""
        if not self.cfg("voice_enable", True):
            return
        if not has_keyword(event.get_message_str(), self.cfg("force_keywords", []) or []):
            return
        umo = event.unified_msg_origin
        if not (await self.chat_settings(umo)).get("voice", True) or not self._voice_rate_ok(umo):
            return
        n = int(self.cfg("voice_hint_chars", 0) or 0)
        limit = f"这次控制在 {n} 个字左右。" if n > 0 else ""
        req.extra_user_content_parts.append(TextPart(text=VOICE_HINT.format(limit=limit)).mark_as_temp())

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
            # 整段合成一条语音，超过 sentence_max_chars 才在句子之间拆成几条（以前是一句一条）
            sentences = voice_chunks(text, max_len=int(self.cfg("sentence_max_chars", 120)))
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

    def local_songs(self, include_blacklisted: bool = False) -> list:
        """歌曲库里的歌（默认去掉黑名单），不需要开唱歌服务。"""
        lib = self.library()
        stems = set()
        for kind in ("清唱", "原曲"):
            d = lib / kind
            if d.is_dir():
                stems.update(f.stem for f in d.iterdir() if f.suffix.lower() in AUDIO_EXT and not f.name.startswith("."))
        if include_blacklisted:
            return sorted(stems)
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
                            pick: dict | None = None, voice: str | None = None, probe: bool = False,
                            deliver_delay: float = 0) -> dict:
        """向唱歌服务点歌。返回服务的回应，另加 refuse（给群友的话）；接单成功会在后台等结果并发送。
        learn=True 时，库里没有就去学（经本机 go-music-dl 多源：网易云/QQ/酷狗/酷我/咪咕/千千/汽水/JOOX/Apple/B站/5sing/Jamendo）。
        suggest=True 时，库里没有先返回 did_you_mean（歌单里只差一两个字的歌）；
        learn 且 confirmed=False 时只搜索不下载，返回 confirm（要学的版本），确认后带 confirmed+pick 再请求。
        voice：furina / dafeiyu，谁来唱（不填用默认歌手）；probe=True 只问库里有没有，有就返回 found、不接单
        （先问点歌的人想听谁唱，顺便把唱歌服务拉起来）。"""
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
                "suggest": suggest, "confirmed": confirmed, "pick": pick, "voice": voice, "probe": probe}
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
        if resp.get("status") in ("confirm", "found"):  # 还没下载 / 还没接单，等点歌的人确认或选歌手
            return resp
        umo = event.unified_msg_origin
        self._sing_user_last[(umo, event.get_sender_id())] = time.time()
        self._sing_chat_times[umo].append(time.time())
        self._spawn(self._deliver(umo, resp, deliver_delay))
        return resp

    async def _ask_yes_no(self, event: AstrMessageEvent, question: str):
        """问点歌的人一个是/不是的问题，等他本人回复（别人说话不算）。返回 True/False，超时返回 None。"""
        await event.send(event.plain_result(question))
        answer = {}

        @session_waiter(timeout=ASK_TIMEOUT, record_history_chains=False)
        async def waiter(controller: SessionController, ev: AstrMessageEvent):
            text = re.sub(r"\s+", "", ev.message_str or "")
            if not text:  # QQ 的“正在输入”之类空事件，不是回答，接着等（不动倒计时）
                return
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
        return (f"这首我还没排练过~ 我在{resp.get('source_name', '')}找到的是《{c.get('name')}》- {c.get('artist')}"
                f"（{mmss(c.get('duration'))}），是这首吗？回「是」我就去学（大概 6 分钟），回「不是」就先不学")

    def default_singer(self) -> str:
        """没问 / 没等到回答时谁来唱：启停脚本记在 sing_voice.conf 里，没记就是大肥鱼。"""
        script = Path(self.cfg("control_script") or "~/dafeiyu-tools/dafeiyu-voice.sh").expanduser()
        try:
            name = (script.parent / "sing_voice.conf").read_text().strip()
        except OSError:
            name = ""
        return "furina" if name == "furina" else "dafeiyu"

    async def _choose_singer(self, event: AstrMessageEvent, preset: str | None = None) -> str:
        """问点歌的人想听谁唱（芙宁娜本人 / 特邀嘉宾大肥鱼），只认他本人的回答。
        点歌时已经说了谁唱、或者配置里关掉了“问谁唱”，就不问；没等到回答就用默认歌手。"""
        if preset in SINGERS:
            return preset
        default = self.default_singer()
        if not self.cfg("ask_singer", True):
            return default
        question = self.cfg("singer_question") or "这首想听本女士亲自献唱，还是请特邀嘉宾大肥鱼登台？回「芙宁娜」或「大肥鱼」就好"
        await event.send(event.plain_result(question))
        answer = {}

        @session_waiter(timeout=ASK_TIMEOUT, record_history_chains=False)
        async def waiter(controller: SessionController, ev: AstrMessageEvent):
            # 2026-11-10 私聊：点歌后 1 秒内 QQ 连着来了几个空事件（多半是“正在输入”），
            # 被当成两次答不上来，直接按默认请了大肥鱼。空事件不算回答，接着等（不动倒计时）
            if not (ev.message_str or "").strip():
                return
            who = parse_singer(ev.message_str or "")
            if who:
                answer["v"] = default if who == "any" else who
                controller.stop()
            elif not answer.get("asked_again"):
                answer["asked_again"] = True
                await ev.send(ev.plain_result("回「芙宁娜」或者「大肥鱼」就好~"))
                controller.keep(timeout=ASK_TIMEOUT, reset_timeout=True)
            else:  # 两次都认不出来，就按默认的来，别一直卡着
                answer["v"] = default
                controller.stop()

        try:
            await waiter(event, SenderFilter())
        except TimeoutError:
            pass
        if "v" not in answer:
            who = "那就请特邀嘉宾大肥鱼登台吧" if default == "dafeiyu" else "那本女士就亲自来吧"
            await event.send(event.plain_result(f"好久没等到回复……{who}"))
            return default
        return answer["v"]

    async def _deliver(self, umo: str, job: dict, delay: float = 0):
        """后台：等唱歌服务做完，发“开嗓”提示和清唱；做不了就发拒绝的话。
        原唱太高/太低时服务会升降调硬唱（job 里带 stretch），一看到就先说一句可能唱不好。
        delay：已经做好的歌先等这么多秒再发（大模型点歌时让它的回话先出去）。"""
        if delay and job.get("status") == "done":
            await asyncio.sleep(delay)
        warned = False

        async def warn_stretch():
            nonlocal warned
            if warned or job.get("stretch") not in ("too_high", "too_low"):
                return
            warned = True
            if job["stretch"] == "too_high":
                tip = self.cfg("stretch_tip_high", "这首的调子起得有点高……得降个调来演绎，谁也不许笑场！")
            else:
                tip = self.cfg("stretch_tip_low", "这首调子低得像男声的戏份……升个调试试，唱得不像也要鼓掌哦")
            if tip:
                await self.context.send_message(umo, MessageChain(chain=[Comp.Plain(tip)]))

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
                if job.get("status") != "done":  # 做完那一轮留到开嗓前再说，免得两句挤在一起
                    await warn_stretch()
            if job["status"] == "failed":
                logger.info(f"[大肥鱼声音] 唱不了 {job.get('song')}：{job.get('message')}")
                text = self.refuse(job.get("reason", "error"))
                if job.get("learning") and job.get("reason") in UNSINGABLE:
                    text += f"\n（《{song_title(job.get('song', ''))}》学下来发现唱不了，已经把它删掉啦，不占地方）"
                await self.context.send_message(umo, MessageChain(chain=[Comp.Plain(text)]))
                return
            if not warned and job.get("stretch"):
                await warn_stretch()
                await asyncio.sleep(1.5)
            if job.get("voice") == "dafeiyu":  # 特邀嘉宾大肥鱼唱的
                tip = self.cfg("guest_start_tip", "这首本大导演请来了特邀嘉宾大肥鱼，掌声欢迎！")
            else:
                tip = self.cfg("sing_start_tip", "咳咳……各位观众，好戏开场！")
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
    async def sing_song(self, event: AstrMessageEvent, song_name: str, artist: str = "", start: str = "",
                        singer: str = "", confirmed=False) -> str:
        """让你用自己的声音清唱一段歌，或者请特邀嘉宾大肥鱼来唱。有人让你唱歌、点歌、让你学一首歌时调用。调用后立刻返回，清唱做好后会自动发到群里。
        工具可能先让你向对方确认歌名、版本或者想听谁唱；对方回答后照提示再调用一次。

        Args:
            song_name(string): 歌名，不要书名号；对方说了版本（如“司风版”“Live 版”）就原样写进歌名，例如 雨爱(司风版)
            artist(string): 对方说的歌手或翻唱者；没说就留空
            start(string): 对方说了从哪开始唱就填，如 0:15 或 1:47；没说就留空
            singer(string): 谁来唱：对方明确要你自己唱就填 芙宁娜，要特邀嘉宾大肥鱼唱就填 大肥鱼；对方没说就留空，工具会让你先问
            confirmed(boolean): 只有对方已经明确确认“就是这首”之后才填 true；第一次调用不要填
        """
        name = song_name.strip("《》 ")
        start = (start or "").strip().replace("：", ":").rstrip("秒s") or None
        if start and not TIME_RE.match(start):
            start = None  # 看不懂的起点就不传，免得唱歌服务直接报错
        same = "song_name、artist、start 都不变"
        confirmed = str(confirmed).strip().lower() in ("true", "1", "yes", "是")
        voice = parse_singer(singer) if singer else None
        who_key = (event.unified_msg_origin, event.get_sender_id())
        asked = time.time() - self._singer_asked.get(who_key, 0) < 600  # 10 分钟内问过这个人
        msg = event.message_str or ""
        named = "furina" if NAMED_FURINA_RE.search(msg) else "dafeiyu" if NAMED_GUEST_RE.search(msg) else None
        if voice and not asked and self.cfg("ask_singer", True):
            # 大模型常会自己替对方选（2026-11-10 私聊“唱 泡沫”它直接填了芙宁娜）：
            # 只有对方这句话里点了名，或者是在回答刚才“想听谁唱”的问题，才认它填的
            if named:
                voice = named
            else:
                logger.info(f"[大肥鱼声音] 对方没说谁唱，忽略大模型填的 singer={singer}，先问")
                voice = None
        elif voice is None and named:  # 对方点了名，大模型却没填
            voice = named
        if voice == "any":
            voice = self.default_singer()
        ask = voice is None and self.cfg("ask_singer", True)
        default_label = SINGERS[self.default_singer()]

        def ask_singer_text(extra=""):
            self._singer_asked[who_key] = time.time()
            return (f"先用一句话问对方想听谁来唱《{name}》：你亲自唱，还是请特邀嘉宾大肥鱼登台。"
                    "用你平时的口吻问，比如“这首想听本女士亲自献唱，还是请特邀嘉宾大肥鱼登台？”，"
                    "现在不要说已经开唱，也不要替对方决定。"
                    f"对方选了再调用 sing_song，{same}{extra}，singer 填“芙宁娜”或“大肥鱼”；"
                    f"对方说随便、都行，就填“{default_label}”。")

        if ask and confirmed:  # 对方刚确认了版本（要去学），学之前先问谁唱
            return ask_singer_text("，confirmed=true")
        # 唱过的歌会马上发：等几秒，让她那句“马上开唱”先发出去（语音回复还要合成几秒）
        resp = await self._request_song(event, name, artist or "", start, learn=self._can_learn(event),
                                        suggest=not confirmed, confirmed=confirmed, voice=voice, probe=ask,
                                        deliver_delay=8)
        if resp.get("status") == "found":  # 库里有，先问谁唱
            name = song_title(resp.get("song") or name)
            return ask_singer_text()
        if resp.get("status") in ("done", "queued", "running"):
            self._singer_asked.pop(who_key, None)
        if resp.get("reason") == "did_you_mean":
            near = resp["suggestions"][0]
            t, a = song_title(near), song_artist(near)
            return (f"歌单里没有{artist + '的' if artist else ''}《{name}》，但有很像的《{t}》（{a or '歌手未知'}），"
                    f"可能是同一首，也可能是对方想要的另一个版本。"
                    f"先用一句话问对方是不是想听《{t}》（{a or '歌手未知'}），现在不要说已经开唱。对方说是，就用 song_name=“{t}”、artist=“{a}” 再调用；"
                    f"对方说就是要《{name}》，再调用时{same}，并带 confirmed=true。")
        if resp.get("status") == "confirm":
            c = resp.get("candidate") or {}
            return (f"《{name}》还没学过。准备学的版本是{resp.get('source_name', '')}的《{c.get('name')}》- {c.get('artist')}"
                    f"（{mmss(c.get('duration'))}），学一首大约 6 分钟。先用一句话问对方是不是这首，现在不要说已经在学；"
                    f"对方确认后再调用 sing_song，{same}，并带 confirmed=true。")
        if resp.get("reason") == "not_found":
            await self._wish(song_name.strip("《》 "), artist or "")
        status = resp.get("status")
        guest = resp.get("voice") == "dafeiyu"
        who = "特邀嘉宾大肥鱼来唱，不是你自己唱" if guest else "你亲自唱"
        if status == "done":
            return (f"《{song_name}》之前唱过（{who}），马上就会发出去。用一句话告诉对方马上开唱"
                    f"{'、嘉宾这就登台' if guest else ''}，不要把歌词打出来。")
        if status in ("queued", "running") and resp.get("learning"):
            return (f"《{song_name}》还没学过，已经开始现学，{who}，大约 {self._eta_text(resp)}后自动发出清唱"
                    "（学下来如果发现原唱太高或太低，会升降调试着唱，到时会先说一声）。用一句话告诉对方要先去排练一下、让对方等等，不要把歌词打出来。")
        if status in ("queued", "running"):
            return (f"已经开始准备《{song_name}》，{who}，大约 {self._eta_text(resp)}后自动发出清唱。"
                    f"用一句话让对方稍等{'（嘉宾正在后台准备）' if guest else ''}，不要把歌词打出来。")
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
        return f"唱不了：{why}。用芙宁娜的口吻傲娇地推掉，{hint}不要把歌词打出来。"

    @staticmethod
    def _pop_singer(parts: list):
        """点歌指令最后一段是“芙宁娜/大肥鱼”就当作指定谁唱（/唱 泡沫 芙宁娜）。"""
        if len(parts) >= 2 and parts[-1].strip().lower() in SINGER_EXACT:
            return SINGER_EXACT[parts.pop().strip().lower()]
        return None

    # 注意：@filter 装饰器必须紧贴在指令函数上面，中间不能插别的函数（11-10 插错过一次，/唱 直接报错）
    @filter.command("唱")
    async def sing_cmd(self, event: AstrMessageEvent):
        """点歌：/唱 歌名 [歌手] [起点如 1:47] [芙宁娜|大肥鱼]"""
        parts = re.sub(r"\s+", " ", event.get_message_str().strip()).split(" ")[1:]
        singer = self._pop_singer(parts)
        start = None
        if parts and TIME_RE.match(parts[-1]):
            start = parts.pop().replace("：", ":").rstrip("秒s")
        singer = singer or self._pop_singer(parts)
        if not parts:
            yield event.plain_result("要唱什么呀？发 /唱 歌名 就好，比如 /唱 多远都要在一起")
            return
        # 歌名可能带空格：先整句当歌名找；找不到再把最后一段当歌手。先只问库里有没有（probe），选好谁唱再接单
        song, artist = " ".join(parts), ""
        if "-" in song and len(parts) == 1:
            artist, _, song = song.partition("-")
        resp = await self._request_song(event, song, artist, start, suggest=False, probe=True)
        if resp.get("reason") == "not_found" and len(parts) >= 2:
            resp = await self._request_song(event, " ".join(parts[:-1]), parts[-1], start, suggest=False, probe=True)
        if resp.get("reason") == "not_found":  # 再问一次，这次让服务找歌单里只差一两个字的歌
            resp = await self._request_song(event, song, artist, start, probe=True)
        async for r in self._sing_flow(event, song, artist, start, resp, singer):
            yield r

    @filter.custom_filter(LearnCmdFilter, False)
    async def learn_cmd(self, event: AstrMessageEvent):
        """学一首歌（按学歌设置决定谁能用）：/学歌 歌名 [歌手] [起点如 0:15] [芙宁娜|大肥鱼]"""
        parts = re.sub(r"\s+", " ", event.get_message_str().strip()).split(" ")[1:]
        singer = self._pop_singer(parts)
        start = None
        if parts and TIME_RE.match(parts[-1]):
            start = parts.pop().replace("：", ":").rstrip("秒s")
        singer = singer or self._pop_singer(parts)
        if not parts:
            yield event.plain_result("用法：/学歌 歌名 歌手 起点，比如 /学歌 雨爱(司风版) 时年先生 0:15")
            return
        if not self._can_learn(event):
            yield event.plain_result("现在不能让我学新歌（看 /学歌 状态）；库里有的歌直接 /唱 歌名 就行")
            return
        # 写法固定：最后一段是歌手（只写一段就是歌名）；“歌手-歌名”也认
        song, artist = " ".join(parts), ""
        if len(parts) >= 2:
            song, artist = " ".join(parts[:-1]), parts[-1]
        elif "-" in song:
            artist, _, song = song.partition("-")
        resp = await self._request_song(event, song, artist, start, probe=True)
        async for r in self._sing_flow(event, song, artist, start, resp, singer):
            yield r

    async def _sing_flow(self, event: AstrMessageEvent, song: str, artist: str, start, resp: dict,
                         singer: str | None = None):
        """点歌的后半段：歌单里有相近的先问“是不是这首”，没有就（能学的话）确认版本后去学；
        能唱的话先问想听谁唱（点歌时说了就不问），再接单，最后回话。"""
        if resp.get("reason") == "did_you_mean":  # 打错字，或者库里只有别人唱的同名歌：先问是不是歌单里那首
            near = resp["suggestions"][0]
            who = f"（{song_artist(near)}）" if song_artist(near) else ""
            then = "我再去找这首学" if self._can_learn(event) else "我把它记进想学清单"
            mine = f"{artist}的" if artist else ""
            ok = await self._ask_yes_no(event, f"歌单里没有{mine}《{song}》，你是想听《{song_title(near)}》{who}吗？"
                                               f"回「是」我就唱这首，回「不是」{then}")
            if ok is None:
                yield event.plain_result("好久没等到回复，那先不唱啦~")
                return
            if ok:
                resp = await self._request_song(event, song_title(near), song_artist(near), start, suggest=False,
                                                probe=True)
            else:
                resp = {"status": "failed", "reason": "not_found", "refuse": self.refuse("not_found")}

        if resp.get("status") == "found":  # 库里有：问谁唱，再正式点
            stem = resp.get("song") or song
            voice = await self._choose_singer(event, singer)
            resp = await self._request_song(event, song_title(stem), song_artist(stem), start, suggest=False,
                                            voice=voice)

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
                voice = await self._choose_singer(event, singer)
                resp = await self._request_song(event, resp["title"], resp["artist"], start, learn=True,
                                                suggest=False, confirmed=True, pick=resp["candidate"], voice=voice)
        if resp.get("reason") == "not_found":
            await self._wish(song, artist)
        status = resp.get("status")
        if status == "done":
            # 后台马上发。这里不回话，要显式停掉事件，否则 AstrBot 会把“/唱 xx”再交给大模型，
            # 大模型又调一次 sing_song，结果唱两遍
            event.stop_event()
            return
        guest = resp.get("voice") == "dafeiyu"
        if status in ("queued", "running") and resp.get("learning"):
            who = "再请特邀嘉宾大肥鱼来唱" if guest else "本女士再亲自登台"
            yield event.plain_result(f"这首还没排练过，先去学一下~ 学好了{who}，大概 {self._eta_text(resp)}后开唱")
            return
        if status in ("queued", "running"):
            ahead = int(resp.get("position") or 1) - 1
            queue = f"前面还有 {ahead} 首，" if ahead > 0 else ""
            who = "特邀嘉宾大肥鱼正在后台准备" if guest else "本女士去后台准备一下"
            yield event.plain_result(f"收到~ {queue}{who}，大概 {self._eta_text(resp)}后开唱")
            return
        yield event.plain_result(resp.get("refuse") or self.refuse("error"))

    @filter.command("歌单")
    async def song_list(self, event: AstrMessageEvent):
        """看看会唱哪些歌"""
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
        yield event.plain_result("本女士的剧目单~ 发 /唱 歌名 点歌（末尾加 芙宁娜 或 大肥鱼 可以直接指定谁唱）\n"
                                 + "\n".join(lines))

    @filter.command_group("唱歌")
    def sing_group(self):
        """唱歌相关：状态、音色、片段、清缓存、拉黑"""

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
        now = f"正在做《{run['song']}》（{run.get('voice_label', '')}，{run['stage']}）" if run else "现在空闲"
        default = SINGERS.get(h.get("default_voice") or self.default_singer(), "大肥鱼")
        yield event.plain_result(f"{now}，还有 {h.get('queue', 0)} 首在排队；没选谁唱时默认请{default}{here}")

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
    @sing_group.command("音色")
    async def sing_voice(self, event: AstrMessageEvent, name: str = ""):
        """看 / 改没选谁唱时默认请谁（管理员）：/唱歌 音色 大肥鱼 或 /唱歌 音色 芙宁娜"""
        name = name.strip().lower()
        key = SINGER_EXACT.get(name) or {"原来": "dafeiyu", "原音色": "dafeiyu", "old": "dafeiyu"}.get(name)
        if name in ("新", "新音色", "雨爱", "new"):
            yield event.plain_result("雨爱音色已经去掉了，现在只有芙宁娜和特邀嘉宾大肥鱼")
            return
        if name and not key:
            yield event.plain_result("只能选「芙宁娜」或「大肥鱼」")
            return
        ask = "点歌时会先问对方想听谁唱，没回答就用它" if self.cfg("ask_singer", True) else "点歌时不问，直接用它"
        if not key:
            yield event.plain_result(f"默认歌手是{SINGERS[self.default_singer()]}（{ask}）\n"
                                     "改：/唱歌 音色 芙宁娜 或 /唱歌 音色 大肥鱼")
            return
        script = Path(self.cfg("control_script") or "~/dafeiyu-tools/dafeiyu-voice.sh").expanduser()
        try:
            proc = await asyncio.create_subprocess_exec(
                "/bin/bash", str(script), "voice", key,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        except (asyncio.TimeoutError, OSError) as e:
            yield event.plain_result(f"启停脚本运行失败：{e}")
            return
        yield event.plain_result((out.decode(errors="ignore").strip() or "已切换") + f"（{ask}）")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @sing_group.command("清缓存")
    async def sing_clear(self, event: AstrMessageEvent):
        """清掉做好的清唱缓存（管理员）：/唱歌 清缓存 清全部；/唱歌 清缓存 歌名 只清这一首"""
        name = re.sub(r"^\S+\s+清缓存\s*", "", re.sub(r"\s+", " ", event.get_message_str().strip()))
        if name:
            stem, hint = self._pick_song(name)
            if not stem:
                yield event.plain_result(hint)
                return
            prefix = cache_prefix(stem)
            n = 0
            for f in (self.library() / "缓存").iterdir():
                if f.name.startswith(prefix) and f.suffix in (".wav", ".json"):
                    f.unlink(missing_ok=True)
                    n += 1
            title = song_title(stem)
            if n:
                yield event.plain_result(f"清掉了《{title}》的 {(n + 1) // 2} 份清唱（芙宁娜、大肥鱼的都清了），下次点会重新唱")
            else:
                yield event.plain_result(f"《{title}》还没有做好的清唱，不用清")
            return
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
    @sing_group.command("片段")
    async def sing_segment(self, event: AstrMessageEvent):
        """改某首歌默认唱哪一段（管理员）：/唱歌 片段 歌名 1:47；/唱歌 片段 歌名 自动；/唱歌 片段 歌名 看现在的"""
        parts = re.sub(r"\s+", " ", event.get_message_str().strip()).split(" ")[2:]
        start = None
        if parts and (TIME_RE.match(parts[-1]) or parts[-1] in ("自动", "auto")):
            start = parts.pop().replace("：", ":").rstrip("秒s")
        if not parts:
            yield event.plain_result("用法：/唱歌 片段 歌名 1:47（从 1 分 47 秒起唱）；/唱歌 片段 歌名 自动（按歌词挑最好唱的那遍副歌）")
            return
        stem, hint = self._pick_song(" ".join(parts))
        if not stem:
            yield event.plain_result(hint)
            return
        path = self.library() / "片段.txt"
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        key = (norm_name(stem), norm_name(song_title(stem)))
        keep, old = [], None
        for line in lines:
            m = re.match(r"^(.+?)\s+(\d+[:：]\d+(?:\.\d+)?|\d+(?:\.\d+)?)$", line.strip())
            if m and not line.strip().startswith("#") and norm_name(m.group(1)) in key:
                old = m.group(2)
                continue
            keep.append(line)
        title = song_title(stem)
        if start is None:
            now = f"从 {old} 开始唱" if old else "没指定，自动挑副歌（找不到歌词时挑人声最响的一段）"
            yield event.plain_result(f"《{title}》现在{now}。改的话：/唱歌 片段 {title} 1:47")
            return
        if start not in ("自动", "auto"):
            if ":" not in start:  # 片段.txt 要写成 1:47
                start = f"{int(start) // 60}:{int(start) % 60:02d}"
            keep.append(f"{stem} {start}")
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
        tmp.replace(path)
        now = f"从 {start} 开始唱" if start not in ("自动", "auto") else "自动挑副歌"
        yield event.plain_result(f"好，《{title}》以后{now}（原来：{old or '自动'}）。"
                                 "换了段落要重新唱一遍，第一次点会慢一点")

    def _pick_song(self, name: str):
        """按歌名（可带“歌手-”）在歌曲库里找一首，返回 (歌手-歌名, None) 或 (None, 提示)。"""
        songs, q = self.local_songs(include_blacklisted=True), norm_name(name)
        exact = [s for s in songs if q in (norm_name(s), norm_name(song_title(s)))]
        hits = exact or [s for s in songs if q and q in norm_name(s)]
        if len(hits) == 1:
            return hits[0], None
        if not hits:
            return None, f"歌单里没找到《{name}》，发 /歌单 看看歌名"
        return None, "有好几首对得上：" + "、".join(hits[:6]) + "，写全一点（可以带歌手，如 邓紫棋-唯一）"

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
        svc = f"运行中（{model_label(h)}）" if h else f"没开（需要时自动打开，模型 {MODEL_NAMES[self.selected_model()]}）"
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
    @service_group.command("模型")
    async def service_model(self, event: AstrMessageEvent, name: str = ""):
        """看 / 切换说话用的模型：/声音 模型 index 或 /声音 模型 breeze（二选一，不影响唱歌）"""
        name = MODEL_ALIASES.get(name.strip().lower(), name.strip().lower())
        if name and name not in MODEL_NAMES:
            yield event.plain_result("只能选 index（IndexTTS2，情绪更强烈）或 breeze（Breeze TTS 2，声音更好听）")
            return
        if not name:
            h = await self.health("tts")
            now = f"正在运行 {model_label(h)}" if h else "说话服务没开"
            yield event.plain_result(f"说话模型选的是 {MODEL_NAMES[self.selected_model()]}（{now}）\n"
                                     "切换：/声音 模型 index 或 /声音 模型 breeze")
            return
        if name == self.selected_model():
            yield event.plain_result(f"说话模型本来就是 {MODEL_NAMES[name]}")
            return
        yield event.plain_result(f"正在切换到 {MODEL_NAMES[name]}，说话服务开着的话要重启，稍等……")
        script = str(Path(self.cfg("control_script") or "~/dafeiyu-tools/dafeiyu-voice.sh").expanduser())
        try:
            proc = await asyncio.create_subprocess_exec(
                "/bin/bash", script, "engine", name,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,  # 重启出来的服务和 AstrBot 分开
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            yield event.plain_result("切换等了 5 分钟还没好，用 /声音 状态 看看")
            return
        except OSError as e:
            yield event.plain_result(f"启停脚本运行失败：{e}")
            return
        yield event.plain_result(out.decode(errors="ignore").strip() or "已切换")

    def selected_model(self) -> str:
        """启停脚本记下的说话模型（index / breeze），没记就是 index。"""
        script = Path(self.cfg("control_script") or "~/dafeiyu-tools/dafeiyu-voice.sh").expanduser()
        try:
            name = (script.parent / "tts_engine.conf").read_text().strip()
        except OSError:
            name = ""
        return name if name in MODEL_NAMES else "index"

    @filter.permission_type(filter.PermissionType.ADMIN)
    @service_group.command("状态")
    async def service_status(self, event: AstrMessageEvent):
        """看两个服务开没开"""
        def desc(h):
            if not h:
                return "没开"
            return {"ready": "运行中", "loading": "启动中", "error": f"出错：{h.get('error')}"}.get(h.get("status"), "未知")
        tts = await self.health("tts")
        model = model_label(tts) if tts else f"选的是 {MODEL_NAMES[self.selected_model()]}"
        yield event.plain_result(f"说话服务（{model}）：{desc(tts)}；唱歌服务：{desc(await self.health('sing'))}")
