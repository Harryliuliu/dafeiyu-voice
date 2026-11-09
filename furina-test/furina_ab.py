"""芙宁娜参考音色：IndexTTS2 与 Breeze TTS 2 同台词对比（台词、情绪参数与 breeze-test/ab_test.py 相同）。

用法：
  mlx-indextts/.venv/bin/python -I furina_ab.py index
  breeze-test/.venv/bin/python  -I furina_ab.py breeze
  任一 venv                      -I furina_ab.py collect   # 把上次大肥鱼音色的版本复制进来作对照

两个模型都很吃内存，一次只跑一个；正式的说话服务要先停着。
"""

import hashlib
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

TOOLS = Path.home() / "dafeiyu-tools"
HERE = TOOLS / "furina-test"
OUT = Path.home() / "Documents" / "Claude Code" / "大肥鱼音色素材" / "输出_芙宁娜对比"
REF_AUDIO = HERE / "ref_furina.wav"
REF_TEXT = "根据故事走向，前往特定情景吗？听起来挺新颖。所以信封里写了什么？"
SEED = 42

# 与 tts_server.py 的 EMOTIONS 一致（用户试听定下的）
EMOTIONS = {
    "开心": [("happy", 0.8)],
    "傲娇": [("happy:0.45,surprised:0.25,angry:0.15,disgusted:0.15", 0.8)],
    "委屈": [("sad", 0.8)],
    "平静": [("calm", 0.6)],
    "害羞": [("happy:0.4,calm:0.4,afraid:0.2", 0.8), ("happy:0.5,calm:0.3,sad:0.2", 0.8)],
    "惊讶": [("surprised", 0.8)],
}

KEEP = "保持参考音频的音色，"
LINES = [
    ("1", "开心", "太好啦！你终于来群里陪我聊天了，我今天超级开心的！", "[笑] ",
     "开心雀跃的语气，声音明亮，语速稍快，带着笑意。"),
    ("2", "傲娇", "哼，谁、谁想你了啊？我只是刚好有空，才勉强回你一句而已。", "",
     "嘴硬心软的傲娇语气：表面不耐烦、有点别扭，其实心里很高兴，尾音略带撒娇。"),
    ("3", "委屈", "我明明等了你一整天，你却连一条消息都不回我……", "[叹气] ",
     "委屈的语气，声音低落、带一点哭腔，像忍着眼泪小声抱怨。"),
    ("4", "害羞", "你、你别一直盯着我看嘛，人家会不好意思的。", "",
     "害羞的语气，声音轻柔、有点迟疑和结巴，像是脸红了不好意思。"),
    ("5", "惊讶", "什么？你说这条鱼已经养了五年了？真的假的！", "",
     "非常惊讶的语气，语调明显上扬，带着难以置信的感觉。"),
    ("6", "平静", "今天天气还不错，晚上早点休息，明天见。", "",
     "平静温和的语气，语速平稳，放松自然。"),
]


def loudness():
    spec = importlib.util.spec_from_file_location("loudness", TOOLS / "loudness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save(path: Path, audio, sr, lv):
    import numpy as np
    import soundfile as sf
    audio = lv.level_loudness(np.asarray(audio, dtype=np.float32).reshape(-1), sr)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio *= 0.99 / peak
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return len(audio) / sr


def run_index():
    lv = loudness()
    os.chdir(TOOLS / "mlx-indextts")
    from mlx_indextts.generate_v2 import IndexTTSv2
    t0 = time.perf_counter()
    tts = IndexTTSv2(model_dir=str(TOOLS / "mlx-indextts" / "models" / "IndexTTS-2-MLX"),
                     memory_limit_gb=12.0)
    print(f"IndexTTS2 加载 {time.perf_counter() - t0:.1f} 秒", flush=True)
    for no, emo, text, _tag, _ins in LINES:
        variants = EMOTIONS[emo]
        e, alpha = variants[int(hashlib.md5(text.encode()).hexdigest(), 16) % len(variants)]
        t0 = time.perf_counter()
        audio = tts.generate(text=text, reference_audio=str(REF_AUDIO), emotion=e, emo_alpha=alpha,
                             temperature=1.0, seed=SEED, speed=1.0, diffusion_steps=15)
        secs = time.perf_counter() - t0
        dur = save(OUT / f"{no}_{emo}_IndexTTS2_芙宁娜.wav", audio, 22050, lv)
        print(f"{no} {emo} IndexTTS2：生成 {secs:.1f} 秒，音频 {dur:.1f} 秒", flush=True)


def run_breeze():
    import mlx.core as mx
    import numpy as np
    from mlx_audio.tts import load
    lv = loudness()
    mx.set_cache_limit(2 * 1024**3)
    t0 = time.perf_counter()
    model = load(str(TOOLS / "breeze-test" / "models" / "Breeze-TTS-2-mlx"))
    print(f"Breeze 加载 {time.perf_counter() - t0:.1f} 秒", flush=True)

    def synth(text, instruct):
        chunks = [np.array(r.audio, dtype=np.float32) for r in model.generate(
            text=text, instruct=instruct, ref_audio=str(REF_AUDIO), ref_text=REF_TEXT,
            cfg_scale=4.0, seed=SEED)]
        return np.concatenate(chunks), model.sample_rate

    for no, emo, text, tag, ins in LINES:
        t0 = time.perf_counter()
        audio, sr = synth(tag + text, KEEP + ins)
        secs = time.perf_counter() - t0
        dur = save(OUT / f"{no}_{emo}_Breeze_芙宁娜.wav", audio, sr, lv)
        print(f"{no} {emo} Breeze：生成 {secs:.1f} 秒，音频 {dur:.1f} 秒", flush=True)


# 害羞句的芙宁娜式语气候选（用户反馈：原指令太柔，不符合芙宁娜人设）
SHY_VARIANTS = [
    ("A_嘴硬", "害羞但嘴硬的语气，像大小姐被说中心事时强装镇定，声音清亮、稍微提高，语速略快，带点慌乱和结巴，不要压低声音。"),
    ("B_戏剧化", "戏剧化的害羞：一开始慌张结巴，后半句假装生气地抱怨，声音明亮有力，尾音带一点撒娇。"),
    ("C_慌乱", "被夸得手足无措的语气，又急又羞，声音偏高、带点颤，语速快，像在努力掩饰脸红。"),
    ("D_原指令加强", "害羞的语气，有点迟疑和结巴，像是脸红了不好意思，但声音保持明亮有精神，不要太轻柔。"),
]


def run_shy():
    import mlx.core as mx
    import numpy as np
    from mlx_audio.tts import load
    lv = loudness()
    mx.set_cache_limit(2 * 1024**3)
    model = load(str(TOOLS / "breeze-test" / "models" / "Breeze-TTS-2-mlx"))
    text = LINES[3][2]
    for name, ins in SHY_VARIANTS:
        chunks = [np.array(r.audio, dtype=np.float32) for r in model.generate(
            text=text, instruct=KEEP + ins, ref_audio=str(REF_AUDIO), ref_text=REF_TEXT,
            cfg_scale=4.0, seed=SEED)]
        dur = save(OUT / f"4_害羞_Breeze_芙宁娜_{name}.wav", np.concatenate(chunks), model.sample_rate, lv)
        print(f"害羞 {name}：音频 {dur:.1f} 秒", flush=True)


def collect():
    old = TOOLS / "breeze-test" / "对比"
    for no, emo, *_ in LINES:
        for model in ("IndexTTS2", "Breeze"):
            src = old / f"{no}_{emo}_{model}.wav"
            if src.exists():
                shutil.copy2(src, OUT / f"{no}_{emo}_{model}_大肥鱼.wav")
    shutil.copy2(REF_AUDIO, OUT / "0_芙宁娜参考原声.wav")
    print("已复制对照文件")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"index": run_index, "breeze": run_breeze, "shy": run_shy, "collect": collect}[sys.argv[1]]()
