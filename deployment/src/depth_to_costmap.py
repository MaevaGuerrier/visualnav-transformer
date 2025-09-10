#!/usr/bin/env python3
# ros node for depth to costmap conversion


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import numpy as np
import subprocess
import os

# UTILS 

from topic_names import (DEPTH_IMAGE_TOPIC,
                        COSTMAP_TOPIC)


class DepthToCostmap(Node):
    def __init__(self):
        super().__init__('depth_to_costmap')

        # Params
        self.declare_parameter('depth_topic', DEPTH_IMAGE_TOPIC)
        self.declare_parameter('scan_topic', '/topoplan/scan')
        self.declare_parameter('fov', 60.0)  # degrees

        self.depth_topic = self.get_parameter('depth_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.fov = np.deg2rad(self.get_parameter('fov').value)

        self.bridge = CvBridge()

        # Sub depth
        self.sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)

        # Pub scan
        self.pub_scan = self.create_publisher(LaserScan, self.scan_topic, 10)

        # Also subscribe to costmap for custom processing
        from nav_msgs.msg import OccupancyGrid
        self.sub_costmap = self.create_subscription(
            OccupancyGrid, COSTMAP_TOPIC, self.costmap_callback, 10
        )

        # Launch nav2_costmap_2d node in background
        self.launch_costmap_node()

    def depth_callback(self, msg: Image):
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = depth_img.shape

        center_row = depth_img[h // 2, :]

        scan = LaserScan()
        scan.header = msg.header
        scan.angle_min = -self.fov/2
        scan.angle_max = self.fov/2
        scan.angle_increment = self.fov / w
        scan.range_min = 0.1
        scan.range_max = 5.0
        scan.ranges = np.where(np.isfinite(center_row), center_row, 0.0).tolist()

        self.pub_scan.publish(scan)

    def costmap_callback(self, msg):
        # At this point, you can add **your processing** here
        self.get_logger().info(f"Received costmap {msg.info.width}x{msg.info.height}")

    def launch_costmap_node(self):


        # open yaml file
        config_file = "../config/nav2_costmap_params.yaml"

        # Launch costmap_2d_node in background
        subprocess.Popen([
            "ros2", "run", "nav2_costmap_2d", "costmap_2d_node",
            "--ros-args", "--params-file", config_file
        ])

def main(args=None):
    rclpy.init(args=args)
    node = DepthToCostmap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
