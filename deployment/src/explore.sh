#!/bin/bash

launch_file=`sed -n 's/^LAUNCH_FILE *= *"\(.*\)"/\1/p' topic_names.py`
launch_pkg=`sed -n 's/^LAUNCH_PKG *= *"\(.*\)"/\1/p' topic_names.py`

# Failsafe to make sure that pip install -e has been executed
# This is necessary to ensure that the package needed are present
#eval "$(conda shell.bash hook)"

#conda activate vint_deployment
source /opt/ros/noetic/setup.bash
# Navigate to the directory containing the package
cd /workspace/src/visualnav-transformer
# Install the package in editable mode
pip install -e train/

# Change back the directory to the working dir with the navigate.py script
cd /workspace/src/visualnav-transformer/deployment/src


# Create a new tmux session
session_name="vint_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane

# Run the roslaunch command in the first pane
tmux select-pane -t 0
#tmux send-keys "roslaunch ${launch_pkg} ${launch_file}" Enter

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 1
#tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 explore.py $@" Enter

# Run the teleop.py script in the third pane
tmux select-pane -t 2
#tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 joy_teleop.py" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 3
#tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 pd_controller.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
