"""在整首清唱人声里挑最适合当 Seed-VC 参考的一段：有声比例高、音量稳、不削波。"""
import sys
import numpy as np
import soundfile as sf

path, length = sys.argv[1], float(sys.argv[2])
a, sr = sf.read(path, dtype="float32")
mono = a.mean(axis=1) if a.ndim == 2 else a
hop = int(sr * 0.05)
n = len(mono) // hop
fr = mono[: n * hop].reshape(n, hop)
db = 10 * np.log10(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)
voiced = db > db.max() - 30
clip = np.max(np.abs(fr), axis=1) > 0.99
win = int(length / 0.05)
lo, hi = int(n * 0.08), int(n * 0.92) - win
best = None
for s in range(lo, hi, 4):  # 每 0.2 秒试一个起点
    v = voiced[s:s + win]
    if v.mean() < 0.5:
        continue
    spread = np.std(db[s:s + win][v])  # 有声部分的响度起伏，越小越稳
    score = v.mean() - 0.02 * spread - 5 * clip[s:s + win].mean()
    if best is None or score > best[0]:
        best = (score, s, v.mean(), spread)
print(f"{path.split('/')[-1]}: 全长 {len(mono)/sr:.0f}s, 声道 {a.ndim}, 全曲有声 {voiced.mean():.0%}, 削波帧 {clip.mean():.2%}")
_, s, vr, sp = best
print(f"  最佳 {length:.0f}s 起点 {s*0.05:.1f}s, 有声 {vr:.0%}, 响度起伏 {sp:.1f} dB")
print(f"START={s*0.05:.1f}")
