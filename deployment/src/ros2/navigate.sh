#!/bin/bash


# Change back the directory to the working dir with the navigate.py script
cd /workspace/src/visualnav-transformer/deployment/src/ros2


SESSION=navigate_bunker

# Start tmux session detached
tmux new-session -d -s $SESSION

# --- Create top 3 panes ---
# Pane 0 exists by default
tmux split-window -h -t $SESSION:0.0          # Creates Pane 1 (0 left, 1 right)
tmux split-window -h -t $SESSION:0.1          # Creates Pane 2 (0 left, 1 middle, 2 right)

# --- Create big bottom pane ---
tmux split-window -v -t $SESSION:0.0          # Splits pane 0 vertically → Pane 3 at bottom
tmux join-pane -h -t $SESSION:0.3             # Merge all bottom splits into pane 3 (if needed)



tmux select-pane -t $SESSION:0.0
tmux send-keys "python3 navigate_${1}.py ${@:2}" Enter


tmux select-pane -t $SESSION:0.1
tmux send-keys "python3 pd_controller.py" Enter 


tmux select-pane -t $SESSION:0.2
tmux send-keys "python3 topic_hz_monitor.py" Enter 

tmux select-pane -t $SESSION:0.3
tmux send-keys "python3 monitor.py" Enter

# Attach to the session
tmux attach -t $SESSION
