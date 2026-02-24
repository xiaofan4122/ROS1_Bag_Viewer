#!/usr/bin/env bash
# 这个脚本尽量使用兼容写法，bash / zsh 都可以执行：
#   bash run_viewer.sh
#   zsh  run_viewer.sh

# 如果是 zsh，开启更接近 sh 的兼容行为（可选，但更稳）
if [ -n "${ZSH_VERSION:-}" ]; then
  emulate -L sh
fi

# 加载用户 shell 配置（可选）
# 注意：bash 用 ~/.bashrc，zsh 用 ~/.zshrc，这里按当前 shell 类型加载
if [ -n "${ZSH_VERSION:-}" ] && [ -f "$HOME/.zshrc" ]; then
  . "$HOME/.zshrc"
elif [ -n "${BASH_VERSION:-}" ] && [ -f "$HOME/.bashrc" ]; then
  . "$HOME/.bashrc"
fi

# 如果 conda 命令存在，则尝试退出当前环境（失败也不终止）
if command -v conda >/dev/null 2>&1; then
  conda deactivate >/dev/null 2>&1 || true
fi

# 1. Source ROS 工作空间（Noetic）
if [ -f /opt/ros/noetic/setup.bash ]; then
  . /opt/ros/noetic/setup.bash
elif [ -f /opt/ros/noetic/setup.zsh ]; then
  . /opt/ros/noetic/setup.zsh
else
  echo "[run_viewer] 错误：未找到 /opt/ros/noetic/setup.bash 或 setup.zsh"
  exit 1
fi

# 1.5 确保 roscore 运行
if ! pgrep -f roscore >/dev/null 2>&1; then
  echo "[run_viewer] roscore 未运行，正在启动..."
  roscore >/tmp/roscore.log 2>&1 &

  # 等待 roscore 就绪（最多 10 秒，每 0.5 秒检测一次）
  count=0
  while [ "$count" -lt 20 ]; do
    if rostopic list >/dev/null 2>&1; then
      echo "[run_viewer] roscore 已就绪。"
      break
    fi
    sleep 0.5
    count=$((count + 1))
  done
fi

# 2. 执行 Python 脚本，并将所有参数传递过去
python3 ./rosbag_viewer.py "$@"