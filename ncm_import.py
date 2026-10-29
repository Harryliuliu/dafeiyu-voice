"""把放进 歌曲库/待导入/ 的歌整理进 歌曲库/原曲/。

- .ncm（网易云会员下载的加密格式）：本地解密成 mp3/flac，歌手和歌名从文件自带的信息里读
- .mp3/.flac/.m4a/.wav 等：按文件名“歌手 - 歌名”（网易云客户端默认的命名）改成“歌手-歌名”
处理完的原文件挪到 待导入/已导入/，出错的挪到 待导入/导入失败/，不删除任何文件。
.ncm 格式参考 ncmdump（github.com/taurusxin/ncmdump）。
"""

import base64
import json
import re
import shutil
import struct
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

CORE_KEY = bytes.fromhex("687A4852416D736F356B496E62617857")
META_KEY = bytes.fromhex("2331346C6A6B5F215C5D2630553C2728")
AUDIO_EXT = {".mp3", ".flac", ".m4a", ".wav", ".aac", ".ogg"}


def _aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    dec = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
    out = dec.update(data) + dec.finalize()
    return out[:-out[-1]]  # 去掉 PKCS7 填充


def decrypt_ncm(path: Path):
    """返回 (音频字节, 元信息 dict, 扩展名)。"""
    with open(path, "rb") as f:
        if f.read(8) != b"CTENFDAM":
            raise ValueError("不是 .ncm 文件")
        f.seek(2, 1)
        n = struct.unpack("<I", f.read(4))[0]
        key = _aes_ecb_decrypt(CORE_KEY, bytes(b ^ 0x64 for b in f.read(n)))[17:]  # 去掉 "neteasecloudmusic"

        box = list(range(256))
        last, off = 0, 0
        for i in range(256):
            swap = box[i]
            c = (swap + last + key[off]) & 0xFF
            off = (off + 1) % len(key)
            box[i], box[c] = box[c], swap
            last = c

        meta = {}
        n = struct.unpack("<I", f.read(4))[0]
        if n:
            raw = bytes(b ^ 0x63 for b in f.read(n))[22:]  # 去掉 "163 key(Don't modify):"
            meta = json.loads(_aes_ecb_decrypt(META_KEY, base64.b64decode(raw))[6:])  # 去掉 "music:"

        f.seek(5, 1)  # crc32 + 1 字节版本
        frame_len, image_len = struct.unpack("<II", f.read(8))
        f.seek(max(frame_len, image_len), 1)  # 跳过封面
        data = np.frombuffer(f.read(), dtype=np.uint8)

    j = (np.arange(256) + 1) & 0xFF
    b = np.array(box, dtype=np.int64)
    stream = b[(b[j] + b[(b[j] + j) & 0xFF]) & 0xFF].astype(np.uint8)
    audio = (data ^ np.resize(stream, data.size)).tobytes()

    ext = (meta.get("format") or "").lower()
    if ext not in ("mp3", "flac"):
        ext = "flac" if audio[:4] == b"fLaC" else "mp3"
    return audio, meta, ext


def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", s).strip() or "未知"


def name_from_filename(stem: str):
    """网易云客户端默认命名“歌手1,歌手2 - 歌名”；去掉 (Live) 之外的多余空格。"""
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
    elif "-" in stem:
        artist, title = stem.split("-", 1)
    else:
        return "未知", stem.strip()
    return re.split(r"[,，、&/]", artist)[0].strip(), title.strip()


def import_pending(lib: Path, log=print) -> list:
    """处理 待导入/ 里的文件，返回新入库的“歌手-歌名”列表。"""
    inbox = lib / "待导入"
    inbox.mkdir(parents=True, exist_ok=True)
    dest_dir = lib / "原曲"
    dest_dir.mkdir(parents=True, exist_ok=True)
    added = []
    for f in sorted(inbox.iterdir()):
        if not f.is_file() or f.name.startswith(".") or f.suffix.lower() not in AUDIO_EXT | {".ncm"}:
            continue
        try:
            if f.suffix.lower() == ".ncm":
                audio, meta, ext = decrypt_ncm(f)
                artists = meta.get("artist") or []
                artist = artists[0][0] if artists and isinstance(artists[0], list) else name_from_filename(f.stem)[0]
                title = meta.get("musicName") or name_from_filename(f.stem)[1]
                dest = dest_dir / f"{safe_name(artist)}-{safe_name(title)}.{ext}"
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(audio)
                tmp.replace(dest)
            else:
                artist, title = name_from_filename(f.stem)
                dest = dest_dir / f"{safe_name(artist)}-{safe_name(title)}{f.suffix.lower()}"
                shutil.copy2(f, dest)
            done = inbox / "已导入"
            done.mkdir(exist_ok=True)
            shutil.move(str(f), str(done / f.name))
            added.append(dest.stem)
            log(f"导入：{f.name} → 原曲/{dest.name}")
        except Exception as e:  # noqa: BLE001
            bad = inbox / "导入失败"
            bad.mkdir(exist_ok=True)
            shutil.move(str(f), str(bad / f.name))
            log(f"导入失败：{f.name}（{type(e).__name__}: {e}）")
    return added


if __name__ == "__main__":
    import sys

    import_pending(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "dafeiyu-tools" / "歌曲库")
