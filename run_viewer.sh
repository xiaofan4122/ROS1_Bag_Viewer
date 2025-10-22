#!/bin/zsh

source ~/.zshrc
conda deactivate

# 1. Source ROS工作空间
source /media/ss/Fan/ws_clion/ws_lio_submit/devel/setup.sh
#source /media/ss/Fan/ws_clion/ws_fast_LIVO/ws_livox_ros_driver/devel/setup.sh # livox_ros_driver_1
source /home/ss/ws_ROS1_noetic/ws_livox/ws_livox/devel/setup.sh # livox_ros_driver_2

# 2. 执行Python脚本, 并将所有参数传递过去
python3 /media/ss/Fan/pycharm_project/ws_yifan_sync/rosbag_viewer.py "$@"
