#!/bin/bash
# 大肥鱼声音服务启停（macOS 自带 bash 3.2 可用）
#
# 用法：
#   bash ~/dafeiyu-tools/dafeiyu-voice.sh start   [tts|sing|all]   启动并等到能用（默认 all）
#   bash ~/dafeiyu-tools/dafeiyu-voice.sh stop    [tts|sing|all]   停止
#   bash ~/dafeiyu-tools/dafeiyu-voice.sh restart [tts|sing|all]
#   bash ~/dafeiyu-tools/dafeiyu-voice.sh status  [tts|sing|all]
#
# tts = 说话服务（127.0.0.1:9880，闲置 15 分钟自动退出）
# sing = 唱歌服务（127.0.0.1:9881，闲置 5 分钟自动退出）
# 想让服务一直开着：DAFEIYU_TTS_IDLE=0 / DAFEIYU_SING_IDLE=0 再 start。
# AstrBot 插件也是调用这个脚本来按需拉起服务。

TOOLS="$HOME/dafeiyu-tools"
LOG="$TOOLS/logs"
RUN="$TOOLS/run"
# AstrBot 从图形界面启动，PATH 里没有 Homebrew；服务要用 ffmpeg
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
mkdir -p "$LOG" "$RUN"

svc_py() {
  case "$1" in
    tts) echo "$TOOLS/mlx-indextts/.venv/bin/python" ;;
    sing) echo "$TOOLS/seed-vc/.venv/bin/python" ;;
  esac
}
svc_port() { case "$1" in tts) echo 9880 ;; sing) echo 9881 ;; esac; }
svc_name() { case "$1" in tts) echo "说话服务" ;; sing) echo "唱歌服务" ;; esac; }

health() { curl -s -m 2 "http://127.0.0.1:$(svc_port "$1")/health"; }

rotate_log() {  # 日志超过 5 MB 时留一份旧的
  local f="$LOG/$1_server.log"
  if [ -f "$f" ] && [ "$(stat -f%z "$f")" -gt 5000000 ]; then mv -f "$f" "$f.1"; fi
}

LOCKS=""
trap 'for d in $LOCKS; do rmdir "$d" 2>/dev/null; done' EXIT

take_lock() {  # 同一个服务同一时间只让一个人去启动；mkdir 是原子的，bash 3.2 也能用
  local d="$RUN/$1.starting" i
  for i in $(seq 1 300); do
    if mkdir "$d" 2>/dev/null; then LOCKS="$LOCKS $d"; return 0; fi
    # 上一个启动者被强行打断时会留下锁，超过 5 分钟就当作失效
    if [ -n "$(find "$d" -maxdepth 0 -mmin +5 2>/dev/null)" ]; then rmdir "$d" 2>/dev/null; continue; fi
    sleep 1
  done
  return 1
}

start_one() {
  local s="$1" name pid i out
  name="$(svc_name "$s")"
  take_lock "$s" || { echo "${name}：等别人启动等太久了"; return 1; }
  case "$(health "$s")" in
    *'"status":"ready"'*) echo "${name}：已在运行"; return 0 ;;
    *'"status":"loading"'*) echo "${name}：正在启动（另一个请求拉起的），等它就绪" ;;
    *)
      rotate_log "$s"
      nohup "$(svc_py "$s")" "$TOOLS/${s}_server.py" >> "$LOG/${s}_server.log" 2>&1 < /dev/null &
      pid=$!
      echo "${name}：启动中（进程 ${pid}）"
      ;;
  esac
  for i in $(seq 1 240); do  # 最多等 4 分钟
    out="$(health "$s")"
    case "$out" in
      *'"status":"ready"'*) echo "${name}：就绪"; return 0 ;;
      *'"status":"error"'*) echo "${name}：模型加载失败 $out"; return 1 ;;
    esac
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "${name}：进程退出了，日志最后几行（$LOG/${s}_server.log）："
      tail -5 "$LOG/${s}_server.log"
      return 1
    fi
    sleep 1
  done
  echo "${name}：等了 4 分钟还没就绪，看日志 $LOG/${s}_server.log"
  return 1
}

stop_one() {
  local s="$1" name pidf pid i
  name="$(svc_name "$s")"
  pidf="$RUN/${s}_server.pid"
  if [ -z "$(health "$s")" ] && [ ! -f "$pidf" ]; then echo "${name}：没在运行"; return 0; fi
  curl -s -m 3 -X POST "http://127.0.0.1:$(svc_port "$s")/shutdown" > /dev/null
  for i in $(seq 1 20); do
    [ -z "$(health "$s")" ] && break
    sleep 0.5
  done
  if [ -f "$pidf" ]; then
    pid="$(cat "$pidf")"
    # 确认这个进程号确实是我们的服务再动手，避免进程号被别的程序复用
    if ps -p "$pid" -o command= 2>/dev/null | grep -q "${s}_server.py"; then
      kill "$pid" 2>/dev/null
      sleep 2
      ps -p "$pid" > /dev/null 2>&1 && kill -9 "$pid" 2>/dev/null
    fi
    rm -f "$pidf"
  fi
  echo "${name}：已停止"
}

status_one() {
  local out
  out="$(health "$1")"
  if [ -z "$out" ]; then
    echo "$(svc_name "$1")：未运行"
  else
    echo "$(svc_name "$1")：$(echo "$out" | grep -o '"status":"[a-z]*"' | head -1 | cut -d'"' -f4)  $out"
  fi
}

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }

cmd="${1:-status}"
target="${2:-all}"
case "$target" in
  tts | sing) list="$target" ;;
  all) list="tts sing" ;;
  *) usage; exit 2 ;;
esac

rc=0
case "$cmd" in
  start) for s in $list; do start_one "$s" || rc=1; done ;;
  stop) for s in $list; do stop_one "$s"; done ;;
  restart) for s in $list; do stop_one "$s"; start_one "$s" || rc=1; done ;;
  status) for s in $list; do status_one "$s"; done ;;
  *) usage; rc=2 ;;
esac
exit $rc
