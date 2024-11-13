#!/bin/bash

launch_file=`sed -n 's/^LAUNCH_FILE *= *"\(.*\)"/\1/p' topic_names.py`
launch_pkg=`sed -n 's/^LAUNCH_PKG *= *"\(.*\)"/\1/p' topic_names.py`
img_topic=`sed -n 's/^IMAGE_TOPIC *= *"\(.*\)"/\1/p' topic_names.py`
ctrl_pkg=`sed -n 's/^TELEOP_PKG *= *"\(.*\)"/\1/p' topic_names.py`
ctrl_launch=`sed -n 's/^TELEOP_LAUNCH *= *"\(.*\)"/\1/p' topic_names.py`

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "roslaunch ${launch_pkg} ${launch_file}" Enter

# Run the teleop.py script in the second pane
tmux select-pane -t 1
tmux send-keys "sleep 1" Enter
tmux send-keys "rosrun ${ctrl_pkg} ${ctrl_launch} speed:.5 turn:.5" Enter # "roslaunch ${ctrl_pkg} ${ctrl_launch}" Enter

# Change the directory to ../topomaps/bags and run the rosbag record command in the third pane
tmux select-pane -t 2
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "rosbag record ${img_topic} -o $1" # change topic if necessary

# Attach to the tmux session
tmux -2 attach-session -t $session_name