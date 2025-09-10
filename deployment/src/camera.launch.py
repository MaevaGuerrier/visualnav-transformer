from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # USB Camera node
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            output='screen',
            parameters=[{
                'video_device': '/dev/video0',
                'image_width': 640,
                'image_height': 480,
                'framerate': 10.0,
                'camera_frame_id': 'camera_frame',
                'io_method': 'mmap'
            }]
        ),

        # Relay node to republish with different QoS
        # Node(
        #     package='topic_tools',
        #     executable='relay',
        #     name='image_relay',
        #     arguments=[
        #         '/image_raw',              # input from usb_cam
        #         '/image_raw_best_effort'   # output topic
        #     ],
        #     parameters=[{
        #         'qos.reliability': 'best_effort',
        #         'qos.history': 'keep_last',
        #         'qos.depth': 5
        #     }]
        # )
    ])
