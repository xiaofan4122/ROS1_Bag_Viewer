#!/bin/zsh

source ~/.zshrc
conda deactivate

# 1. Source ROS工作空间
source /opt/ros/noetic/setup.zsh

# 1.5 确保 roscore 运行
if ! pgrep -f roscore >/dev/null 2>&1; then
  echo "[run_viewer] roscore 未运行，正在启动..."
  roscore >/tmp/roscore.log 2>&1 &
  # 等待 roscore 就绪（最多 10 秒）
  for i in {1..20}; do
    if rostopic list >/dev/null 2>&1; then
      echo "[run_viewer] roscore 已就绪。"
      break
    fi
    sleep 0.5
  done
fi

# 2. 执行Python脚本, 并将所有参数传递过去
python3 ./rosbag_viewer.py "$@"
