"""大肥鱼唱歌引擎：Seed-VC 歌声转换 + demucs 人声分离，模型只加载一次。

从 seed-vc/inference.py 的 load_models / main 拆出来：参考歌声的特征在启动时算好，
每首歌只做源人声这一半。Mac 补丁（见交接说明第 4 节）：
  - 音高数据转 float32 再上 GPU（inference.py 已打补丁，这里自己写的部分也照做）
  - BigVGAN 声码器放 CPU（load_models 里已打补丁）
  - demucs 用 CPU
必须用 seed-vc 的 venv 运行；导入前要设 PYTORCH_ENABLE_MPS_FALLBACK=1。
"""

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SV_ROOT = Path.home() / "dafeiyu-tools" / "seed-vc"
SR = 44100
HOP = 512


def find_tool(name: str) -> str:
    """AstrBot 拉起的进程 PATH 里没有 Homebrew，所以自己找。"""
    for d in ("/opt/homebrew/bin", "/usr/local/bin"):
        p = Path(d) / name
        if p.exists():
            return str(p)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"找不到 {name}，请先 brew install ffmpeg")
    return found


FFMPEG = find_tool("ffmpeg")
FFPROBE = find_tool("ffprobe")


def probe_duration(path: str) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def decode(path: str, start: float = 0.0, duration: float | None = None, channels: int = 2) -> np.ndarray:
    """用 ffmpeg 读任意格式（m4a/mp3/flac/wav），返回 (采样数, 声道) 的 float32，44.1 kHz。"""
    import soundfile as sf

    cmd = [FFMPEG, "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", str(channels), "-ar", str(SR), "-f", "wav", "-acodec", "pcm_f32le", "pipe:1"]
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio


class ChunkedVocoder:
    """BigVGAN 放在 Mac GPU 上分段跑。

    MPS 不支持输出长度超过 65536 采样的卷积（= 128 帧 mel），整段只能放 CPU，30 秒要一分半。
    这里每段只算 96 帧新内容，左右各带 12 帧上下文、多算 4 帧做交叉淡化，总长 124 帧，
    在限制以内，速度快十几倍。
    """

    HOP = 512

    def __init__(self, bigvgan_model, device, chunk=96, ctx=12, overlap=4):
        self.m = bigvgan_model.to(device)
        self.device, self.chunk, self.ctx, self.ov = device, chunk, ctx, overlap

    def __call__(self, mel):
        torch = __import__("torch")
        mel = mel.to(self.device)
        T, H = mel.size(2), self.HOP
        if T + 2 * self.ctx <= 120:
            return self.m(mel)
        ov = self.ov * H
        fade_in = torch.linspace(0, 1, ov, device=self.device)
        out = None
        for s in range(0, T, self.chunk):
            e = min(T, s + self.chunk + self.ov)  # 本段负责 [s, e)，末尾多出的 ov 帧和下一段重叠
            a, b = max(0, s - self.ctx), min(T, e + self.ctx)
            y = self.m(mel[:, :, a:b])[..., (s - a) * H:(e - a) * H]
            if out is None:
                out = y
            else:
                n = min(ov, y.size(-1), out.size(-1))
                out[..., -n:] = out[..., -n:] * (1 - fade_in[:n]) + y[..., :n] * fade_in[:n]
                out = torch.cat([out, y[..., n:]], dim=-1)
            if e >= T:
                break
        return out


class SingEngine:
    def __init__(self, ref_path: str, diffusion_steps: int = 40, cfg_rate: float = 0.7,
                 ref_seconds: float = 25.0, fp16: bool = False, gpu_vocoder: bool = True, log=print):
        self.log = log
        self.diffusion_steps = diffusion_steps
        self.cfg_rate = cfg_rate
        self.ref_seconds = ref_seconds
        self.fp16 = fp16
        os.chdir(SV_ROOT)  # inference.py 用相对路径找 checkpoints/ 和 configs/
        if str(SV_ROOT) not in sys.path:
            sys.path.insert(0, str(SV_ROOT))
        import inference as inf  # noqa: E402  导入时会把 HF_HUB_CACHE 设成 ./checkpoints/hf_cache

        self.torch = __import__("torch")
        self.inf = inf
        self.device = inf.device
        args = SimpleNamespace(f0_condition=True, checkpoint=None, config=None, fp16=False)
        (self.model, self.semantic_fn, self.f0_fn, self.vocoder_fn,
         self.campplus, self.to_mel, mel_args) = inf.load_models(args)
        assert mel_args["sampling_rate"] == SR
        if gpu_vocoder and self.device.type == "mps":
            # load_models 里打过补丁：vocoder_fn = lambda x: bigvgan_model(x.cpu()).to(device)，从闭包取出模型
            self.vocoder_fn = ChunkedVocoder(self.vocoder_fn.__closure__[0].cell_contents, self.device)
        self._prepare_reference(ref_path)
        # 2026-11-10 起分离改用 sep_worker.py（Roformer + 去和声），demucs 只在分离进程出问题时兜底，用到才加载
        self.demucs = None
        self.vocals_idx = None

    # ------------------------------------------------------------ 参考歌声 ----
    def _prepare_reference(self, ref_path: str):
        import librosa
        import torchaudio

        torch = self.torch
        with torch.no_grad():
            # Seed-VC 最多用 25 秒参考；参考越长，每轮能顺带转换的新内容越少（上下文共 30 秒）
            ref = librosa.load(ref_path, sr=SR)[0][: int(SR * min(25.0, self.ref_seconds))]
            ref_t = torch.tensor(ref).unsqueeze(0).float().to(self.device)
            ref_16k = torchaudio.functional.resample(ref_t, SR, 16000)
            self.S_ori = self.semantic_fn(ref_16k)
            self.mel2 = self.to_mel(ref_t.float())
            lengths = torch.LongTensor([self.mel2.size(2)]).to(self.device)
            feat2 = torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            self.style2 = self.campplus(feat2.unsqueeze(0))
            f0 = self.f0_fn(ref_16k[0], thred=0.03)
            self.ref_median_hz = float(np.median(f0[f0 > 1]))
            F0_ori = torch.from_numpy(f0).float().to(self.device)[None]
            self.prompt_condition = self.model.length_regulator(
                self.S_ori, ylens=lengths, n_quantizers=3, f0=F0_ori)[0]

    # ------------------------------------------------------------ 分离与音高 ----
    def separate(self, stereo: np.ndarray) -> np.ndarray:
        """stereo: (采样数, 2) 44.1 kHz → 人声单声道。CPU 上跑（MPS 不支持 demucs 的大卷积）。"""
        from demucs.apply import apply_model

        if self.demucs is None:
            from demucs.pretrained import get_model

            self.demucs = get_model("htdemucs").cpu().eval()
            self.vocals_idx = self.demucs.sources.index("vocals")
        torch = self.torch
        wav = torch.from_numpy(np.ascontiguousarray(stereo.T)).float()
        mono = wav.mean(0)
        mean, std = mono.mean(), mono.std() + 1e-8
        with torch.no_grad():
            out = apply_model(self.demucs, ((wav - mean) / std)[None], device="cpu",
                              split=True, overlap=0.25, progress=False)[0]
        vocals = out[self.vocals_idx] * std + mean
        return vocals.mean(0).numpy().astype(np.float32)

    def pitch_stats(self, vocals: np.ndarray) -> dict:
        """RMVPE 测音高：只统计有声帧，返回中位数和 90% 分位（Hz）。"""
        import torchaudio

        torch = self.torch
        with torch.no_grad():
            w16 = torchaudio.functional.resample(torch.from_numpy(vocals)[None].float(), SR, 16000)
            f0 = self.f0_fn(w16[0], thred=0.03)
        voiced = f0[f0 > 1]
        if voiced.size < 50:
            return {"median_hz": 0.0, "p90_hz": 0.0, "voiced_ratio": float(voiced.size / max(1, f0.size))}
        return {
            "median_hz": float(np.median(voiced)),
            "p90_hz": float(np.percentile(voiced, 90)),
            "voiced_ratio": float(voiced.size / f0.size),
        }

    # ------------------------------------------------------------ 转换 ----
    def _semantic_long(self, w16):
        """whisper 一次最多 30 秒；更长的按 inference.main 的办法分块，块间重叠 5 秒。"""
        torch = self.torch
        if w16.size(-1) <= 16000 * 30:
            return self.semantic_fn(w16)
        ov = 5
        parts, buffer, traversed = [], None, 0
        while traversed < w16.size(-1):
            if buffer is None:
                chunk = w16[:, traversed:traversed + 16000 * 30]
            else:
                chunk = torch.cat([buffer, w16[:, traversed:traversed + 16000 * (30 - ov)]], dim=-1)
            S = self.semantic_fn(chunk)
            parts.append(S if traversed == 0 else S[:, 50 * ov:])
            buffer = chunk[:, -16000 * ov:]
            traversed += 30 * 16000 if traversed == 0 else chunk.size(-1) - 16000 * ov
        return torch.cat(parts, dim=1)

    def convert(self, vocals: np.ndarray, semitone: int = 0) -> np.ndarray:
        """源人声（单声道 44.1 kHz）→ 大肥鱼音色，和 inference.main 的 f0 模式一致。"""
        import torchaudio

        torch, inf, device = self.torch, self.inf, self.device
        max_context_window = SR // HOP * 30
        overlap_frame_len = 16
        overlap_wave_len = overlap_frame_len * HOP

        with torch.no_grad():
            src = torch.from_numpy(vocals)[None].float().to(device)
            src_16k = torchaudio.functional.resample(src, SR, 16000)
            S_alt = self._semantic_long(src_16k)
            mel = self.to_mel(src.float())
            target_lengths = torch.LongTensor([mel.size(2)]).to(device)

            F0_alt = torch.from_numpy(self.f0_fn(src_16k[0], thred=0.03)).float().to(device)[None]
            shifted = torch.exp(torch.log(F0_alt + 1e-5))  # 和原版一样：无声帧变成 1e-5 而不是 0
            if semitone:
                shifted[F0_alt > 1] = inf.adjust_f0_semitones(shifted[F0_alt > 1], semitone)
            cond = self.model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted)[0]

            mel2 = self.mel2
            max_source_window = max_context_window - mel2.size(2)
            processed, chunks, previous = 0, [], None
            while processed < cond.size(1):
                chunk_cond = cond[:, processed:processed + max_source_window]
                is_last = processed + max_source_window >= cond.size(1)
                cat = torch.cat([self.prompt_condition, chunk_cond], dim=1)
                with torch.autocast(device_type=device.type, dtype=torch.float16 if self.fp16 else torch.float32):
                    vc = self.model.cfm.inference(
                        cat, torch.LongTensor([cat.size(1)]).to(device), mel2, self.style2, None,
                        self.diffusion_steps, inference_cfg_rate=self.cfg_rate)
                    vc = vc[:, :, mel2.size(-1):]
                wave = self.vocoder_fn(vc.float()).squeeze()[None]
                if processed == 0:
                    if is_last:
                        chunks.append(wave[0].cpu().numpy())
                        break
                    chunks.append(wave[0, :-overlap_wave_len].cpu().numpy())
                elif is_last:
                    chunks.append(inf.crossfade(previous.cpu().numpy(), wave[0].cpu().numpy(), overlap_wave_len))
                    break
                else:
                    chunks.append(inf.crossfade(previous.cpu().numpy(),
                                                wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len))
                previous = wave[0, -overlap_wave_len:]
                processed += vc.size(2) - overlap_frame_len
        return np.concatenate(chunks).astype(np.float32)


# ---------------------------------------------------------------- 后处理 ----
def finish(audio: np.ndarray, target_db: float = -18.0, fade_in: float = 0.15, fade_out: float = 1.5) -> np.ndarray:
    """统一响度（只看有声部分）+ 淡入淡出 + 防爆音。唱歌保留原本的强弱，不做逐段均衡。"""
    a = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    hop = SR // 20
    n = len(a) // hop
    if n > 4:
        power = np.mean(a[: n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1) + 1e-12
        db = 10 * np.log10(power)
        voiced = power[db > db.max() - 30]
        a *= 10 ** ((target_db - 10 * np.log10(voiced.mean())) / 20)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 0.89:
        a *= 0.89 / peak
    fi, fo = int(SR * fade_in), int(SR * fade_out)
    if len(a) > fi + fo:
        a[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)
        a[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32) ** 2
    return a


def loudest_window(mono: np.ndarray, length_s: float, skip_head: float = 0.15, skip_tail: float = 0.20) -> float:
    """在 [15%, 80%] 范围内找人声能量最大的一段，返回起点（秒）。"""
    total = len(mono) / SR
    if total <= length_s:
        return 0.0
    sec = np.array([np.mean(mono[i * SR:(i + 1) * SR] ** 2) for i in range(int(total))])
    win = int(length_s)
    lo, hi = int(total * skip_head), int(total * (1 - skip_tail)) - win
    if hi <= lo:
        lo, hi = 0, int(total) - win
    sums = np.convolve(sec, np.ones(win), mode="valid")
    best = lo + int(np.argmax(sums[lo:hi + 1]))
    return float(best)
