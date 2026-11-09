"""芙宁娜 GPT-SoVITS V2ProPlus 微调模型试合成（同 furina_ab.py 的 6 句台词），和 Breeze 对比。

用法（必须在 GPT-SoVITS 目录里跑，GSV 的模型路径都是相对路径）：
  cd ~/dafeiyu-tools/GPT-SoVITS
  .venv/bin/python -I ../furina-test/gsv_ab.py same       # 6 句都用同一句参考（和 Breeze 现在用的一样）
  .venv/bin/python -I ../furina-test/gsv_ab.py emotion    # 每种情绪换一句同语气的参考（读 gsv_refs.json）

模型：ModelScope leletxh/Furina_GPT_SoVITS（GPT e15 + SoVITS e8，v2ProPlus），Mac 上用 CPU。
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

TOOLS = Path.home() / "dafeiyu-tools"
GSV = TOOLS / "GPT-SoVITS"
HERE = TOOLS / "furina-test"
MODEL = HERE / "model" / "v2proplus"
OUT = Path.home() / "Documents" / "Claude Code" / "大肥鱼音色素材" / "输出_芙宁娜对比" / "GSV对比"
REF_AUDIO = HERE / "ref_furina.wav"
REF_TEXT = "根据故事走向，前往特定情景吗？听起来挺新颖。所以信封里写了什么？"
SEED = 42

LINES = [
    ("1", "开心", "太好啦！你终于来群里陪我聊天了，我今天超级开心的！"),
    ("2", "傲娇", "哼，谁、谁想你了啊？我只是刚好有空，才勉强回你一句而已。"),
    ("3", "委屈", "我明明等了你一整天，你却连一条消息都不回我……"),
    ("4", "害羞", "你、你别一直盯着我看嘛，人家会不好意思的。"),
    ("5", "惊讶", "什么？你说这条鱼已经养了五年了？真的假的！"),
    ("6", "平静", "今天天气还不错，晚上早点休息，明天见。"),
]


def loudness():
    spec = importlib.util.spec_from_file_location("loudness", TOOLS / "loudness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(mode):
    sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS")]
    import numpy as np
    import soundfile as sf
    from TTS_infer_pack.TTS import TTS, TTS_Config

    lv = loudness()
    OUT.mkdir(parents=True, exist_ok=True)
    pre = "GPT_SoVITS/pretrained_models"
    t0 = time.perf_counter()
    tts = TTS(TTS_Config({"custom": {
        "device": "cpu", "is_half": False, "version": "v2ProPlus",
        "t2s_weights_path": str(MODEL / "Funina-e15.ckpt"),
        "vits_weights_path": str(MODEL / "Funina_e8_s464.pth"),
        "bert_base_path": f"{pre}/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": f"{pre}/chinese-hubert-base",
    }}))
    print(f"GSV 加载 {time.perf_counter() - t0:.1f} 秒", flush=True)

    refs = {}
    if mode == "emotion":
        refs = json.loads((HERE / "gsv_refs.json").read_text(encoding="utf-8"))

    for no, emo, text in LINES:
        ref_audio, ref_text = (refs[emo]["wav"], refs[emo]["text"]) if emo in refs else (str(REF_AUDIO), REF_TEXT)
        t = time.perf_counter()
        sr, audio = None, []
        for sr, chunk in tts.run({
            "text": text, "text_lang": "zh", "ref_audio_path": ref_audio,
            "prompt_text": ref_text, "prompt_lang": "zh",
            "top_k": 15, "top_p": 1, "temperature": 1, "text_split_method": "cut5",
            "batch_size": 1, "speed_factor": 1.0, "seed": SEED, "repetition_penalty": 1.35,
        }):
            audio.append(chunk)
        secs = time.perf_counter() - t
        a = np.concatenate(audio).astype(np.float32) / 32768.0
        a = lv.level_loudness(a, sr)
        peak = float(np.max(np.abs(a))) if a.size else 0.0
        if peak > 0.99:
            a *= 0.99 / peak
        name = "同一参考" if mode == "same" else "情绪参考"
        sf.write(str(OUT / f"{no}_{emo}_GSV_{name}.wav"), a, sr, subtype="PCM_16")
        print(f"{no} {emo}：生成 {secs:.1f} 秒，音频 {len(a) / sr:.1f} 秒（{sr} Hz）", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
