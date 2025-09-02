#!/bin/bash
set -e

# --- Source ROS1 ---
source /opt/ros/noetic/setup.bash

# --- Source ROS2 (change humble->foxy if needed) ---
source /opt/ros/humble/setup.bash

# --- Start ros1_bridge in background ---
ros2 run ros1_bridge dynamic_bridge > /tmp/bridge.log 2>&1 &
BRIDGE_PID=$!

# --- Run Python pipeline (ROS2) ---
python3 depth_to_costmap.py

# --- Kill bridge on exit ---
kill $BRIDGE_PID
