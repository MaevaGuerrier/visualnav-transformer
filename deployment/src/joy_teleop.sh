#!/bin/bash

# Create a new tmux session
session_name="teleop_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into two panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "roslaunch gnm_locobot.launch" Enter

# Run the teleop.py script in the second pane
tmux select-pane -t 1
tmux send-keys "conda init" Enter
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "pip install -e ../../train/" Enter
tmux send-keys "python joy_teleop.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name