#!/bin/bash
# 大肥鱼声音服务启停（macOS 自带 bash 3.2 可用）
#
# 用法：
#   bash ~/dafeiyu-tools/dafeiyu-voice.sh start   [tts|sing|all]   启动并等到能用（默认 all）
#   bash ~/dafeiyu-tools/dafeiyu-voice.sh stop    [tts|sing|all]   停止
#   bash ~/dafeiyu-tools/dafeiyu-voice.sh restart [tts|sing|all]
#   bash ~/dafeiyu-tools/dafeiyu-voice.sh status  [tts|sing|musicdl|all]
#   bash ~/dafeiyu-tools/dafeiyu-voice.sh engine  [index|breeze]       看 / 切换说话用的模型
#   bash ~/dafeiyu-tools/dafeiyu-voice.sh voice   [old|new]            看 / 切换唱歌用的音色
#
# tts = 说话服务（127.0.0.1:9880，闲置 15 分钟自动退出）。两个模型二选一，占同一个端口：
#       index  = IndexTTS2（默认，情绪更强烈）
#       breeze = Breeze TTS 2（声音更好听；仅限非商业用途）
#       切换时说话服务正开着就自动重启成新模型；唱歌不受影响。
# sing = 唱歌服务（127.0.0.1:9881，闲置 5 分钟自动退出）。音色二选一：
#       new = 新音色（voices/歌声参考_雨爱_23秒.wav，默认）
#       old = 原音色（voices/歌声参考_23秒.wav）
#       切换时唱歌服务正开着就自动重启；两种音色的缓存分开存，来回切不用重唱。
# musicdl = 学歌用的 go-music-dl（127.0.0.1:37777，开机自动常驻；start/stop 也可单独管它，
#           stop 后要手动 start 才会回来；all 不含它，status all 会显示它）
# 想让服务一直开着：DAFEIYU_TTS_IDLE=0 / DAFEIYU_SING_IDLE=0 再 start。
# AstrBot 插件也是调用这个脚本来按需拉起服务。

TOOLS="$HOME/dafeiyu-tools"
LOG="$TOOLS/logs"
RUN="$TOOLS/run"
# AstrBot 从图形界面启动，PATH 里没有 Homebrew；服务要用 ffmpeg
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
mkdir -p "$LOG" "$RUN"

# 说话用哪个模型，记在这个文件里（index / breeze），没有就是 index
ENGINE_FILE="$TOOLS/tts_engine.conf"
engine() {
  local e
  [ -f "$ENGINE_FILE" ] && e="$(tr -d ' \t\r\n' < "$ENGINE_FILE")"
  case "$e" in breeze) echo breeze ;; *) echo index ;; esac
}
engine_label() { case "$1" in breeze) echo "Breeze TTS 2" ;; *) echo "IndexTTS2" ;; esac; }
running_engine() {  # 正在跑的说话服务是哪个模型；没在跑就输出空
  local h
  h="$(health tts)"
  [ -z "$h" ] && return
  case "$h" in *'"engine":"breeze"'*) echo breeze ;; *) echo index ;; esac
}

# 唱歌用哪个音色，记在这个文件里（new / old），没有就是 new
VOICE_FILE="$TOOLS/sing_voice.conf"
voice() {
  local v
  [ -f "$VOICE_FILE" ] && v="$(tr -d ' \t\r\n' < "$VOICE_FILE")"
  case "$v" in old) echo old ;; *) echo new ;; esac
}
voice_label() { case "$1" in old) echo "原音色" ;; new) echo "新音色·雨爱" ;; *) echo "其他参考" ;; esac; }
voice_ref() { case "$1" in old) echo "$TOOLS/voices/歌声参考_23秒.wav" ;; *) echo "$TOOLS/voices/歌声参考_雨爱_23秒.wav" ;; esac; }
running_voice() {  # 正在跑的唱歌服务用的哪个音色；没在跑就输出空
  local h
  h="$(health sing)"
  [ -z "$h" ] && return
  case "$h" in
    *'"ref":"歌声参考_23秒.wav"'*) echo old ;;
    *'"ref":"歌声参考_雨爱_23秒.wav"'*) echo new ;;
    *) echo other ;;
  esac
}

svc_py() {
  case "$1" in
    tts) if [ "$(engine)" = breeze ]; then echo "$TOOLS/breeze-test/.venv/bin/python"
         else echo "$TOOLS/mlx-indextts/.venv/bin/python"; fi ;;
    sing) echo "$TOOLS/seed-vc/.venv/bin/python" ;;
  esac
}
svc_script() {
  case "$1" in
    tts) if [ "$(engine)" = breeze ]; then echo "$TOOLS/breeze_tts_server.py"; else echo "$TOOLS/tts_server.py"; fi ;;
    *) echo "$TOOLS/${1}_server.py" ;;
  esac
}
svc_port() { case "$1" in tts) echo 9880 ;; sing) echo 9881 ;; musicdl) echo 37777 ;; esac; }
svc_name() { case "$1" in tts) echo "说话服务" ;; sing) echo "唱歌服务" ;; musicdl) echo "音乐下载（go-music-dl）" ;; esac; }

health() {
  if [ "$1" = musicdl ]; then curl -s -m 2 "http://127.0.0.1:37777/music/healthz"; return; fi
  curl -s -m 2 "http://127.0.0.1:$(svc_port "$1")/health"
}

# go-music-dl 由 launchd 管（开机自启、崩溃自动重启），这里只是装载 / 卸载它
MDL_LABEL="com.dafeiyu.musicdl"
MDL_PLIST="$HOME/Library/LaunchAgents/$MDL_LABEL.plist"
start_musicdl() {
  local name i
  name="$(svc_name musicdl)"
  case "$(health musicdl)" in *'"app":"go-music-dl"'*) echo "${name}：已在运行"; return 0 ;; esac
  if launchctl print "gui/$(id -u)/$MDL_LABEL" > /dev/null 2>&1; then
    launchctl kickstart -k "gui/$(id -u)/$MDL_LABEL"
  else
    launchctl bootstrap "gui/$(id -u)" "$MDL_PLIST" || { echo "${name}：装载失败（$MDL_PLIST）"; return 1; }
  fi
  for i in $(seq 1 20); do
    case "$(health musicdl)" in *'"app":"go-music-dl"'*) echo "${name}：就绪"; return 0 ;; esac
    sleep 0.5
  done
  echo "${name}：没起来。桌面版 music-dl-desktop-go 开着的话会占用 37777 端口，先退出它；日志 $LOG/musicdl.log"
  tail -5 "$LOG/musicdl.log" 2>/dev/null
  return 1
}
stop_musicdl() {
  launchctl bootout "gui/$(id -u)/$MDL_LABEL" > /dev/null 2>&1
  echo "$(svc_name musicdl)：已停止（下次开机会自动启动；现在要用就 start musicdl）"
}

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
  if [ "$s" = musicdl ]; then start_musicdl; return; fi
  name="$(svc_name "$s")"
  [ "$s" = tts ] && name="${name}（$(engine_label "$(engine)")）"
  [ "$s" = sing ] && name="${name}（$(voice_label "$(voice)")）"
  take_lock "$s" || { echo "${name}：等别人启动等太久了"; return 1; }
  case "$(health "$s")" in
    *'"status":"ready"'*)
      if [ "$s" = tts ] && [ "$(running_engine)" != "$(engine)" ]; then
        echo "说话服务：已在运行，但用的是 $(engine_label "$(running_engine)")（要换就 restart tts）"
      elif [ "$s" = sing ] && [ "$(running_voice)" != "$(voice)" ]; then
        echo "唱歌服务：已在运行，但用的是 $(voice_label "$(running_voice)")（要换就 restart sing）"
      else
        echo "${name}：已在运行"
      fi
      return 0 ;;
    *'"status":"loading"'*) echo "${name}：正在启动（另一个请求拉起的），等它就绪" ;;
    *)
      rotate_log "$s"
      if [ "$s" = sing ]; then  # 手动设了 DAFEIYU_SING_REF 就听它的，否则按 sing_voice.conf
        DAFEIYU_SING_REF="${DAFEIYU_SING_REF:-$(voice_ref "$(voice)")}" \
          nohup "$(svc_py "$s")" "$(svc_script "$s")" >> "$LOG/${s}_server.log" 2>&1 < /dev/null &
      else
        nohup "$(svc_py "$s")" "$(svc_script "$s")" >> "$LOG/${s}_server.log" 2>&1 < /dev/null &
      fi
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
  if [ "$s" = musicdl ]; then stop_musicdl; return; fi
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
    # （tts_server.py 也能匹配 breeze_tts_server.py，两个说话模型都认）
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
  local out name
  out="$(health "$1")"
  name="$(svc_name "$1")"
  if [ -z "$out" ]; then
    [ "$1" = tts ] && name="${name}（选的是 $(engine_label "$(engine)")）"
    [ "$1" = sing ] && name="${name}（选的是 $(voice_label "$(voice)")）"
    echo "${name}：未运行"
  else
    [ "$1" = tts ] && name="${name}（$(engine_label "$(running_engine)")）"
    [ "$1" = sing ] && name="${name}（$(voice_label "$(running_voice)")）"
    echo "${name}：$(echo "$out" | grep -o '"status":"[a-z]*"' | head -1 | cut -d'"' -f4)  $out"
  fi
}

set_engine() {
  local new="$1" old running
  old="$(engine)"
  case "$new" in
    "") echo "说话模型：$(engine_label "$old")（${old}）。切换：engine index 或 engine breeze"; return 0 ;;
    index | breeze) ;;
    *) echo "不认识的模型：${new}（只能是 index 或 breeze）"; return 2 ;;
  esac
  echo "$new" > "$ENGINE_FILE"
  echo "说话模型：$(engine_label "$old") → $(engine_label "$new")"
  running="$(running_engine)"
  if [ -n "$running" ] && [ "$running" != "$new" ]; then
    echo "说话服务正开着，重启成新模型："
    stop_one tts
    start_one tts
  else
    echo "下次说话服务启动时生效（插件需要时会自动拉起）"
  fi
}

set_voice() {
  local new="$1" old running
  old="$(voice)"
  case "$new" in
    "") echo "唱歌音色：$(voice_label "$old")（${old}）。切换：voice old 或 voice new"; return 0 ;;
    old | new) ;;
    *) echo "不认识的音色：${new}（只能是 old 或 new）"; return 2 ;;
  esac
  echo "$new" > "$VOICE_FILE"
  echo "唱歌音色：$(voice_label "$old") → $(voice_label "$new")"
  running="$(running_voice)"
  if [ -n "$running" ] && [ "$running" != "$new" ]; then
    echo "唱歌服务正开着，重启换成$(voice_label "$new")："
    stop_one sing
    start_one sing
  else
    echo "下次唱歌服务启动时生效（插件需要时会自动拉起）"
  fi
}

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; }

cmd="${1:-status}"
if [ "$cmd" = engine ]; then set_engine "$2"; exit $?; fi
if [ "$cmd" = voice ]; then set_voice "$2"; exit $?; fi
target="${2:-all}"
case "$target" in
  tts | sing | musicdl) list="$target" ;;
  all) list="tts sing"; [ "$cmd" = status ] && list="tts sing musicdl" ;;
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
