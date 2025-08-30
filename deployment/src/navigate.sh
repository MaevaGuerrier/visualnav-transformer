#!/bin/bash

# Create a new tmux session
session_name="bunker_nomad_$(date +%s)"
tmux new-session -d -s $session_name

tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 0
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 navigate.py $@" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 1
tmux send-keys "python3 navigate_pd_controller.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
