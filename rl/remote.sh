#!/usr/bin/env bash
# 远端 DQN 训练助手脚本。
#
# 用法：
#   ./remote.sh start              # 同步代码 + 启动训练（脱离 SSH 会话）
#   ./remote.sh logs               # 实时跟踪日志（Ctrl-C 离开不影响训练）
#   ./remote.sh status             # 显示 PID、最新一行日志、GPU 占用
#   ./remote.sh stop               # 终止远端训练
#   ./remote.sh pull               # 把 ckpt/ 下载到本地
#   ./remote.sh play [--no-render] # 在远端跑 play.py（可选不开窗口）
#   ./remote.sh shell              # 直接 ssh 进远端 rl 目录
#
# 训练参数也可以覆盖：
#   EPISODES=10000 BALLS=4 ./remote.sh start

set -euo pipefail

# ---- 配置（用环境变量覆盖即可）-----------------------------------
HOST="${HOST:-xlisp@192.168.1.104}"
REMOTE_DIR="${REMOTE_DIR:-/home/xlisp/PyPro/billiards-skills}"
PY="${PY:-/home/xlisp/miniconda3/envs/codegpt/bin/python}"
PIP="${PIP:-/home/xlisp/miniconda3/envs/codegpt/bin/pip}"

EPISODES="${EPISODES:-5000}"
BALLS="${BALLS:-3}"
DEVICE="${DEVICE:-cuda}"
LOG_EVERY="${LOG_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-500}"

LOG_FILE="${REMOTE_DIR}/rl/logs/train.log"
CKPT_DIR="${REMOTE_DIR}/rl/ckpt"
LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-./ckpt_remote}"

cmd="${1:-help}"

# ---- 工具函数 -----------------------------------------------------
remote() { ssh -o ConnectTimeout=8 "$HOST" "$@"; }

ensure_deps() {
    echo "[deps] checking remote python deps..."
    remote "$PY -c 'import torch, gymnasium, numpy, pygame' 2>/dev/null" \
        || { echo "[deps] installing missing packages..."; \
             remote "$PIP install -q gymnasium pygame numpy"; }
}

sync_code() {
    # 优先用 git；本地有未提交改动则用 rsync 兜底。
    if git -C "$(dirname "$0")/.." diff --quiet HEAD 2>/dev/null \
       && git -C "$(dirname "$0")/.." diff --cached --quiet 2>/dev/null; then
        echo "[sync] git pull on remote"
        remote "cd $REMOTE_DIR && git pull --ff-only"
    else
        echo "[sync] working tree dirty → rsync rl/"
        rsync -az --delete \
            --exclude '__pycache__' --exclude 'logs' --exclude 'ckpt' \
            "$(dirname "$0")/" "$HOST:$REMOTE_DIR/rl/"
    fi
}

is_running() {
    remote "pgrep -f '[r]l/train.py' >/dev/null"
}

# ---- 命令分发 -----------------------------------------------------
case "$cmd" in
start)
    if is_running; then
        echo "[start] training already running on $HOST. use './remote.sh stop' first or check status."
        exit 1
    fi
    sync_code
    ensure_deps
    echo "[start] launching: episodes=$EPISODES balls=$BALLS device=$DEVICE"
    remote "cd $REMOTE_DIR/rl && mkdir -p logs ckpt && \
        setsid $PY -u train.py \
            --episodes $EPISODES --balls $BALLS --device $DEVICE \
            --log-every $LOG_EVERY --save-every $SAVE_EVERY \
            < /dev/null > logs/train.log 2>&1 & disown"
    sleep 2
    remote "pgrep -af '[r]l/train.py' | head -1"
    echo "[start] OK. follow logs:  ./remote.sh logs"
    ;;

logs)
    remote "tail -F $LOG_FILE"
    ;;

status)
    echo "=== process ==="
    remote "pgrep -af '[r]l/train.py' || echo '(not running)'"
    echo "=== last 5 log lines ==="
    remote "tail -5 $LOG_FILE 2>/dev/null || echo '(no log yet)'"
    echo "=== gpu ==="
    remote "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || echo '(no gpu)'"
    echo "=== checkpoints ==="
    remote "ls -lt $CKPT_DIR 2>/dev/null | head -6 || echo '(no ckpt)'"
    ;;

stop)
    echo "[stop] killing training on $HOST..."
    remote "pkill -f '[r]l/train.py' && echo killed || echo 'nothing to kill'"
    ;;

pull)
    mkdir -p "$LOCAL_CKPT_DIR"
    echo "[pull] $HOST:$CKPT_DIR/  →  $LOCAL_CKPT_DIR/"
    rsync -avz "$HOST:$CKPT_DIR/" "$LOCAL_CKPT_DIR/"
    ;;

play)
    shift || true
    extra="${*:-}"
    echo "[play] running play.py on remote ($extra)"
    remote "cd $REMOTE_DIR/rl && $PY play.py --ckpt ckpt/dqn_latest.pt --balls $BALLS --device $DEVICE $extra"
    ;;

shell)
    ssh -t "$HOST" "cd $REMOTE_DIR/rl && exec \$SHELL -l"
    ;;

help|*)
    awk '/^set -/{exit} {print}' "$0"
    ;;
esac
