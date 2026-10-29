"""音量均衡：把一句话前后的响度拉平，再整体调到统一音量。

IndexTTS2 的输出本身很响（峰值顶满），而且常常越说越大声；直接发出去开头轻、结尾炸。
做法：按 50 ms 分帧算响度，只看有声帧，用约 1.5 秒的窗口求“慢变化”的响度曲线，
把它拉向目标值。窗口够长，所以字与字之间的轻重还在，只修整句的起伏。
"""

import numpy as np


def level_loudness(
    audio: np.ndarray,
    sr: int,
    target_db: float = -20.0,  # 有声部分的目标 RMS（dBFS）
    max_boost_db: float = 8.0,
    max_cut_db: float = 18.0,
    window_s: float = 1.5,
    peak_limit: float = 0.89,  # 约 -1 dBFS
) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    hop = int(sr * 0.05)
    n = len(a) // hop
    if n < 4:
        return a
    frames = a[: n * hop].reshape(n, hop)
    power = np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12
    db = 10 * np.log10(power)
    voiced = db > db.max() - 30

    win = np.hanning(max(3, int(window_s / 0.05) | 1))
    num = np.convolve(power * voiced, win, mode="same")
    den = np.convolve(voiced.astype(np.float64), win, mode="same")
    has = den > 1e-3
    loud = np.full(n, np.nan)
    loud[has] = 10 * np.log10(num[has] / den[has])
    idx = np.arange(n)
    loud = np.interp(idx, idx[has], loud[has])  # 静音处沿用旁边的响度，避免把底噪放大

    gain_db = np.clip(target_db - loud, -max_cut_db, max_boost_db)
    g_win = np.hanning(7)
    gain_db = np.convolve(np.pad(gain_db, 3, mode="edge"), g_win / g_win.sum(), mode="valid")

    centers = idx * hop + hop / 2
    gain = 10 ** (np.interp(np.arange(len(a)), centers, gain_db) / 20)
    out = a * gain.astype(np.float32)
    peak = float(np.max(np.abs(out)))
    if peak > peak_limit:
        out *= peak_limit / peak
    return out
