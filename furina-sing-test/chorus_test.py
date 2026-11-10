"""自动挑副歌试验：对歌曲库里每首原曲，整首分离人声（顺便存进唱歌服务的分离缓存），找歌词、挑副歌，
和 片段.txt 里用户手选的起点、以及原来“人声最响的一段”比较。

  PYTORCH_ENABLE_MPS_FALLBACK=1 ~/dafeiyu-tools/seed-vc/.venv/bin/python -I ~/dafeiyu-tools/furina-sing-test/chorus_test.py
"""

import json
import sys
from pathlib import Path

TOOLS = Path.home() / "dafeiyu-tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "seed-vc"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

import chorus  # noqa: E402
import sing_workers  # noqa: E402
from sing_engine import SR, decode, loudest_window  # noqa: E402

LIB = TOOLS / "歌曲库"
SEP_CACHE = LIB / "缓存" / "分离"
HERE = TOOLS / "furina-sing-test"


def mmss(t):
    return "—" if t is None else f"{int(t) // 60}:{int(t) % 60:02d}"


def main():
    from modules.rmvpe import RMVPE

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rmvpe_pt = next((TOOLS / "seed-vc" / "checkpoints").glob("models--lj1995--VoiceConversionWebUI/snapshots/*/rmvpe.pt"))
    f0_model = RMVPE(str(rmvpe_pt), is_half=False, device=device)

    def f0_track(mono):
        with torch.no_grad():
            w16 = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(mono)).float()[None], SR, 16000)
            return f0_model.infer_from_audio(w16[0].to(device), thred=0.03)

    manual = {}
    for line in (LIB / "片段.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            stem, _, t = line.strip().rpartition(" ")
            m, _, s = t.partition(":")
            manual[stem] = int(m) * 60 + float(s)

    sep = sing_workers.separator(print)
    tmp = HERE / "chorus_tmp"
    tmp.mkdir(exist_ok=True)
    report = {}
    for path in sorted((LIB / "原曲").iterdir()):
        if path.suffix.lower() not in {".m4a", ".mp3", ".flac", ".wav"}:
            continue
        stem = path.stem
        full = SEP_CACHE / f"{stem}.flac"
        whole = None
        if full.exists():
            whole = sf.read(str(full), dtype="float32", always_2d=True)[0]
            if whole.shape[1] != 2:
                whole = None
        if whole is None:
            src, dst = tmp / "mix.wav", tmp / "vocals.wav"
            sf.write(str(src), decode(str(path)), SR, subtype="FLOAT")
            sep.call({"op": "vocals", "input": str(src), "output": str(dst)}, timeout=900)
            whole = sf.read(str(dst), dtype="float32", always_2d=True)[0][:, :2]
            SEP_CACHE.mkdir(parents=True, exist_ok=True)
            sf.write(str(full), whole, SR, format="FLAC")
            print(f"整首分离：{stem}", flush=True)
        loud = loudest_window(whole.mean(axis=1), 30.0)
        got = chorus.pick_start(stem, whole, f0_track, LIB / "歌词", log=lambda m: print("  " + m, flush=True))
        auto = got[0] if got else None
        report[stem] = {"手选": manual.get(stem), "自动副歌": auto, "原来最响": loud, "说明": got[1] if got else ""}
        print(f"{stem}: 手选 {mmss(manual.get(stem))}｜自动副歌 {mmss(auto)}｜原来最响 {mmss(loud)}", flush=True)
        (HERE / "report_chorus.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sep.stop()


if __name__ == "__main__":
    main()
