"""从芙宁娜中文语音包里挑句子，拼成 Seed-VC 唱歌参考（约 23 秒）。

两段都以社区 GPT-SoVITS 包里那句 7 秒参考（现在说话也在用，用户已认可）开头，再补句子：
  A 平稳：不带感叹、音高起伏小、音高接近她的常态
  B 明亮：音高偏高但不喊叫，看能不能让高音少一点电音
挑句子的硬条件：只用主目录（不含战斗/其它/带变量语音），不含（）心声台词，3–9 秒，不削波，
音色和那句 7 秒参考接近（CAM++ 说话人向量余弦相似度，Seed-VC 自己取音色也用它）。

用 seed-vc 的 venv 跑：
  ~/dafeiyu-tools/seed-vc/.venv/bin/python -I ~/dafeiyu-tools/furina-sing-test/pick_furina_ref.py
"""

import csv
import json
import sys
from pathlib import Path

TOOLS = Path.home() / "dafeiyu-tools"
SV_ROOT = TOOLS / "seed-vc"
sys.path.insert(0, str(SV_ROOT))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

DATA = TOOLS / "furina-test" / "dataset" / "芙宁娜"
SEED = TOOLS / "voices" / "芙宁娜_参考.wav"  # 社区包那句 7 秒
HERE = TOOLS / "furina-sing-test"
OUT = HERE / "参考"
SR = 44100
TARGET = 23.0
GAP = 0.15  # 句间留白

CAMPPLUS = next((SV_ROOT / "checkpoints").glob("models--funasr--campplus/snapshots/*/campplus_cn_common.bin"))
RMVPE_PT = next((SV_ROOT / "checkpoints").glob("models--lj1995--VoiceConversionWebUI/snapshots/*/rmvpe.pt"))


def load(path):
    return librosa.load(str(path), sr=SR, mono=True)[0].astype(np.float32)


def trim(a, top_db=40):
    """去掉头尾静音，各留 30 毫秒。"""
    _, (s, e) = librosa.effects.trim(a, top_db=top_db)
    pad = int(0.03 * SR)
    return a[max(0, s - pad): min(len(a), e + pad)]


def main():
    from modules.campplus.DTDNN import CAMPPlus
    from modules.rmvpe import RMVPE

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cam = CAMPPlus(feat_dim=80, embedding_size=192)
    cam.load_state_dict(torch.load(str(CAMPPLUS), map_location="cpu", weights_only=True))
    cam.eval().to(device)
    f0_model = RMVPE(str(RMVPE_PT), is_half=False, device=device)

    def analyse(a):
        with torch.no_grad():
            t = torch.from_numpy(a)[None]
            w16 = torchaudio.functional.resample(t, SR, 16000)
            feat = torchaudio.compliance.kaldi.fbank(w16, num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            emb = cam(feat.unsqueeze(0).to(device))[0].cpu().numpy()
            f0 = f0_model.infer_from_audio(w16[0].to(device), thred=0.03)
        voiced = f0[f0 > 1]
        semis = 12 * np.log2(voiced / 440.0) if voiced.size else np.array([0.0])
        return {
            "emb": emb / (np.linalg.norm(emb) + 1e-9),
            "median_hz": float(np.median(voiced)) if voiced.size else 0.0,
            "p90_hz": float(np.percentile(voiced, 90)) if voiced.size else 0.0,
            "pitch_std": float(np.std(semis)),
            "voiced": float(voiced.size / max(1, f0.size)),
        }

    seed = trim(load(SEED))
    seed_info = analyse(seed)
    seed_name = "vo_CLLQ001_3_furina_08.wav"  # 那句 7 秒在语音包里的原文件，别重复选

    rows = []
    files = sorted(DATA.glob("*.wav"))
    for i, wav in enumerate(files):
        lab = wav.with_suffix(".lab")
        text = lab.read_text(encoding="utf-8").strip() if lab.exists() else ""
        if not text or "（" in text or "(" in text or wav.name == seed_name:
            continue
        a = load(wav)
        if np.max(np.abs(a)) > 0.99:
            continue
        a = trim(a)
        dur = len(a) / SR
        if not 3.0 <= dur <= 9.0:
            continue
        info = analyse(a)
        if info["voiced"] < 0.35 or info["median_hz"] <= 0:
            continue
        info.update(file=wav.name, text=text, dur=dur, sim=float(info["emb"] @ seed_info["emb"]))
        rows.append(info)
        if i % 200 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    meds = np.array([r["median_hz"] for r in rows])
    typical = float(np.median(meds))
    print(f"候选 {len(rows)} 句，她说话的中位音高约 {typical:.0f} Hz；那句 7 秒 {seed_info['median_hz']:.0f} Hz", flush=True)

    calm_marks = "！!～~"
    for r in rows:
        dev = abs(12 * np.log2(r["median_hz"] / typical))
        r["score_A"] = (r["sim"] * 4 + r["voiced"] - 0.15 * r["pitch_std"] - 0.15 * dev
                        - (1.0 if any(c in r["text"] for c in calm_marks) else 0.0))
        up = 12 * np.log2(r["median_hz"] / typical)  # 比常态高几个半音
        r["score_B"] = (r["sim"] * 4 + r["voiced"] + 0.25 * min(up, 5) - 0.4 * max(0, up - 5)
                        - 0.15 * max(0, r["pitch_std"] - 4)
                        - (0.6 if r["text"].count("！") + r["text"].count("!") >= 2 else 0.0))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(HERE / "候选句子.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["文件", "文字", "秒", "音色相似", "中位Hz", "p90Hz", "音高起伏", "有声比例", "A分", "B分"])
        for r in sorted(rows, key=lambda r: -r["sim"]):
            w.writerow([r["file"], r["text"], f"{r['dur']:.1f}", f"{r['sim']:.3f}", f"{r['median_hz']:.0f}",
                        f"{r['p90_hz']:.0f}", f"{r['pitch_std']:.1f}", f"{r['voiced']:.2f}",
                        f"{r['score_A']:.2f}", f"{r['score_B']:.2f}"])

    def rms_norm(a, db=-20.0):
        fr = librosa.feature.rms(y=a)[0]
        loud = fr[fr > fr.max() * 0.1]
        cur = 20 * np.log10(np.sqrt(np.mean(loud ** 2)) + 1e-9)
        return a * (10 ** ((db - cur) / 20))

    report = {"typical_hz": round(typical), "seed": {"file": seed_name, "median_hz": round(seed_info["median_hz"])}}
    for tag, key, label in (("A", "score_A", "平稳"), ("B", "score_B", "明亮")):
        picked, total, used_prefix = [], len(seed) / SR, {seed_name.rsplit("_", 1)[0]}
        for r in sorted(rows, key=lambda r: -r[key]):
            prefix = r["file"].rsplit("_", 1)[0]  # 同一段剧情最多取一句，避免语气雷同
            if prefix in used_prefix or total + GAP + r["dur"] > TARGET + 0.5:
                continue
            picked.append(r)
            used_prefix.add(prefix)
            total += GAP + r["dur"]
            if total >= TARGET - 1.0:
                break
        gap = np.zeros(int(GAP * SR), dtype=np.float32)
        parts = [rms_norm(seed)]
        for r in picked:
            parts += [gap, rms_norm(trim(load(DATA / r["file"])))]
        audio = np.concatenate(parts)
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio *= 0.95 / peak
        name = f"芙宁娜_{tag}_{label}_{len(audio) / SR:.0f}秒"
        sf.write(str(OUT / f"{name}.wav"), audio, SR, subtype="PCM_16")
        whole = analyse(audio)
        lines = [f"{name}.wav：{len(audio) / SR:.1f} 秒，整段中位音高 {whole['median_hz']:.0f} Hz",
                 f"1. {seed_name}（社区 GSV 包参考，7 秒）：根据故事走向，前往特定情景吗？听起来挺新颖。所以信封里写了什么？"]
        for j, r in enumerate(picked, 2):
            lines.append(f"{j}. {r['file']}（{r['dur']:.1f} 秒，{r['median_hz']:.0f} Hz，相似 {r['sim']:.2f}）：{r['text']}")
        (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        report[tag] = {"wav": f"{name}.wav", "seconds": round(len(audio) / SR, 1),
                       "median_hz": round(whole["median_hz"]), "sentences": [r["file"] for r in picked]}
    (HERE / "参考" / "挑选结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
