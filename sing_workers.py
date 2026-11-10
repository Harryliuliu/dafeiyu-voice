"""唱歌服务调用的两个工作进程：人声分离（sep_worker.py）和芙宁娜 RVC（rvc_worker.py）。

它们各有自己的 venv（torch 版本不同，装不进 seed-vc 的环境），所以用子进程跑：
一行一个 JSON 请求/回应，走标准输入输出；模型在子进程里只加载一次。
唱歌服务退出时管道关掉，子进程读到结束就自己退出。子进程挂了，下次调用时自动重开一次。
"""

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

TOOLS = Path.home() / "dafeiyu-tools"
LOG_DIR = TOOLS / "logs"


class Worker:
    def __init__(self, name: str, python: Path, script: Path, cwd: Path, log=print):
        self.name, self.python, self.script, self.cwd = name, python, script, cwd
        self.log = log
        self.proc = None
        self.lines = None
        self.lock = threading.Lock()

    def available(self) -> bool:
        return self.python.exists() and self.script.exists()

    def _reader(self, proc, lines):
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)  # 子进程退出

    def _start(self, timeout: float):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        err = open(LOG_DIR / f"{self.name}_worker.log", "a", encoding="utf-8")
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1", PYTHONUNBUFFERED="1")
        t = time.time()
        self.proc = subprocess.Popen([str(self.python), "-I", str(self.script)], cwd=str(self.cwd), env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True,
                                     bufsize=1)
        err.close()
        self.lines = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc, self.lines), daemon=True).start()
        msg = self._read(timeout)
        if not msg.get("ready"):
            raise RuntimeError(f"{self.name} 进程没准备好：{msg}")
        self.log(f"{self.name} 进程就绪，用时 {time.time() - t:.0f} 秒")

    def _read(self, timeout: float) -> dict:
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError(f"{self.name} 进程 {timeout:.0f} 秒没回应")
            line = self.lines.get(timeout=left)
            if line is None:
                raise RuntimeError(f"{self.name} 进程退出了（看 logs/{self.name}_worker.log）")
            try:
                return json.loads(line)
            except ValueError:
                continue  # 不是回应的杂行，跳过

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def warm(self, timeout: float = 180):
        """提前把进程和模型拉起来（在后台线程里调用）。"""
        with self.lock:
            if not self.alive():
                self._start(timeout)

    def call(self, req: dict, timeout: float = 300) -> dict:
        with self.lock:
            for attempt in (1, 2):
                try:
                    if not self.alive():
                        self._start(180)
                    self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
                    self.proc.stdin.flush()
                    resp = self._read(timeout)
                    if not resp.get("ok"):
                        raise RuntimeError(resp.get("error") or resp)
                    return resp
                except (OSError, RuntimeError, TimeoutError, queue.Empty) as e:
                    if attempt == 2 or (isinstance(e, RuntimeError) and self.alive()):
                        raise  # 进程活着但报错：是这一首的问题，不重试
                    self.log(f"{self.name} 进程出问题（{e}），重开一次")
                    self.stop()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        self.proc = None


def separator(log=print) -> Worker:
    return Worker("sep", TOOLS / "sep-test" / ".venv" / "bin" / "python", TOOLS / "sep_worker.py", TOOLS, log)


def rvc(log=print) -> Worker:
    applio = TOOLS / "rvc-test" / "Applio"
    return Worker("rvc", applio / ".venv" / "bin" / "python", TOOLS / "rvc_worker.py", applio, log)
