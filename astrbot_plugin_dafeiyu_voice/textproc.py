"""文字处理：语气标签、送去合成前的清洗、切句。不依赖 AstrBot，方便单独测试。"""

import re

# 2026-11-10 加【嘴硬】：芙宁娜的傲娇分成表层端架子（傲娇）和里层口是心非（嘴硬）
EMOTION_LABELS = ("开心", "傲娇", "嘴硬", "委屈", "害羞", "惊讶", "平静")
_LABELS = "|".join(EMOTION_LABELS)
# 【开心】，也容忍 [开心]、【语气：开心】这类写法
TAG_RE = re.compile(rf"[【\[]\s*(?:语气\s*[:：]\s*)?({_LABELS})\s*[】\]]")

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
# 括号里的动作描写和颜文字整个去掉：（甩尾）(摇尾巴)(｡･ω･｡) *伸懒腰*
# 2026-11-10：上限 30 → 80 字，角色扮演式的长动作描写也不念
BRACKET_RE = re.compile(r"（[^（）]{0,80}）|\([^()]{0,80}\)|【[^【】]{0,80}】|\*[^*\n]{1,40}\*")
# 只保留汉字、字母数字和常用标点，其余（emoji、颜文字零件、假名、韩文等）都删掉
DISALLOWED_RE = re.compile(r"[^一-鿿㐀-䶿A-Za-z0-9\s，。！？、；：,.!?;:%+\-—'\"“”‘’~…]")
PUNCT = "，。！？、；：,.!?;:—'\"“”‘’"
DASH_RE = re.compile(r"[—–─－]+|-{2,}")  # 各种破折号（单个 - 不算，免得把 3-5 这类数字范围改掉）
# 破折号念成“省略号 + 换行”：说话服务在换行处把句子分两段合成，中间插固定长度的停顿（默认 0.8 秒）
DASH_PAUSE = "……\n"
# 结巴“谁、谁”“你，你”：顿号 Breeze 几乎不停，换成制表符，说话服务在那里插短停顿（默认 0.35 秒，比破折号短）
STUTTER_RE = re.compile(r"([一-鿿]{1,2})[、，,]\s*(?=\1)")
STUTTER_PAUSE = "\t"
# 省略号不算句子结束：“那……我喜欢你”要连着念，不然“那”会被切到上一句的末尾（2026-11-10）
SENTENCE_RE = re.compile(r".*?[。？！?!~\n]+|.+$", re.S)
SPOKEN_RE = re.compile(r"[一-鿿㐀-䶿A-Za-z0-9]")


def extract_tag(text: str):
    m = TAG_RE.search(text or "")
    return m.group(1) if m else None


def strip_tags(text: str) -> str:
    return TAG_RE.sub("", text or "")


def spoken_len(text: str) -> int:
    """会被念出来的字数（汉字、字母、数字）。"""
    return len(SPOKEN_RE.findall(text or ""))


CJK = r"一-鿿㐀-䶿"
SOFT_END = "，,；;：:、"  # 句中停顿，切开后可以去掉


def _pre_clean(text: str) -> str:
    """断句之前先去掉不念的东西，免得网址里的 ? 之类被当成句号。"""
    t = URL_RE.sub(" ", strip_tags(text or ""))
    prev = None
    while prev != t:  # 括号可能嵌套在颜文字里，多跑几遍；换成空格，夹在两句话中间时会变成停顿
        prev, t = t, BRACKET_RE.sub(" ", t)
    return re.sub(r"\.{3,}|。{2,}", "…", t)


def clean_for_tts(text: str) -> str:
    # 2026-11-10 实测 Breeze：破折号几乎不停（0.09 秒），省略号停 0.3–0.85 秒不稳定。
    # 所以省略号原样留给 Breeze（以前换成逗号，停顿就没了）；破折号换成 DASH_PAUSE，由说话服务保证停够。
    t = DASH_RE.sub("—", _pre_clean(text))  # 各种破折号先统一成一个 —
    t = DISALLOWED_RE.sub("", t)
    t = re.sub(r"~+", "，", t)  # ~ 只用来断句，不念
    t = re.sub(rf"(?<=[{CJK}])\s+(?=[{CJK}])", "，", t)  # 汉字之间的空格（多半是删掉颜文字留下的）当停顿
    t = re.sub(r"\s+", " ", t)
    p = re.escape(PUNCT.replace("—", ""))
    t = re.sub(rf"\s*([{p}])[{p}\s]*", r"\1", t)  # 连续标点只留第一个
    t = re.sub(rf"[{p}\s]*…+[{p}\s]*", "……", t)  # 省略号统一成“……”，前后的标点去掉
    t = re.sub(rf"[{p}\s…]*—[{p}\s…—]*", DASH_PAUSE, t)  # 破折号（连同旁边的标点）→ 固定停顿
    t = STUTTER_RE.sub(lambda m: m.group(1) + STUTTER_PAUSE, t)  # 结巴 → 短停顿
    return t.strip(PUNCT + " …\n\t")


def _split_long(s: str, max_len: int) -> list:
    if spoken_len(s) <= max_len:
        return [s]
    parts, cur = [], ""
    for piece in re.findall(rf"[^{SOFT_END}]+[{SOFT_END}]?", s):
        if cur and spoken_len(cur + piece) > max_len:
            parts.append(cur)
            cur = ""
        cur += piece
    if cur:
        parts.append(cur)
    out = []
    for p in parts:  # 一个逗号都没有的超长句，硬切
        while spoken_len(p) > max_len:
            out.append(p[:max_len])
            p = p[max_len:]
        if p:
            out.append(p)
    return [p.strip(SOFT_END + " ") for p in out if spoken_len(p) > 0]


def split_sentences(text: str, max_len: int = 40, merge_below: int = 8, merge_max: int = 30) -> list:
    """切成要合成的句子：先按 。？！~… 断句并清洗，太长的再按逗号切，太短的（哼、呜、诶？）并进下一句。"""
    sentences = []
    for seg in SENTENCE_RE.findall(_pre_clean(text)):
        s = clean_for_tts(seg)
        if spoken_len(s) == 0:
            continue
        end = seg.strip().rstrip("~…")[-1:]
        if end in "？?！!":
            s += "？" if end in "？?" else "！"  # 保留问号/感叹号，语气更对
        sentences.extend(_split_long(s, max_len))
    merged = []
    for s in sentences:
        if merged and spoken_len(merged[-1]) < merge_below and spoken_len(merged[-1] + s) <= merge_max:
            sep = "" if merged[-1][-1:] in PUNCT else "，"
            merged[-1] = merged[-1] + sep + s
        else:
            merged.append(s)
    return merged


def voice_chunks(text: str, max_len: int = 120) -> list:
    """一次回复要合成的几段：整段念成一条语音，超过 max_len 个字才在句子之间拆成几条（2026-11-10）。

    以前一句一条语音，大模型回两句就收到两条很短的语音；整段一起合成，语气也更连贯。
    """
    chunks = []
    for s in split_sentences(text, max_len=max_len):
        if chunks and spoken_len(chunks[-1] + s) <= max_len:
            sep = "" if chunks[-1][-1:] in PUNCT else "。"
            chunks[-1] = chunks[-1] + sep + s
        else:
            chunks.append(s)
    return chunks


def has_keyword(message: str, keywords) -> bool:
    return any(k and k in (message or "") for k in keywords)
