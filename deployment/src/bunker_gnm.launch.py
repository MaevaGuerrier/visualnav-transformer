from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch.actions import IncludeLaunchDescription


def generate_launch_description():
    return LaunchDescription([
        # Example: include first launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('bunker_base'),
                    'launch',
                    'bunker_base.launch.py'
                ])
            ])
        ),

        # Node(
        #     package='usb_cam',
        #     executable='usb_cam_node_exe',
        #     name='usb_cam',
        #     output='screen',
        # ),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('bunker_robot_server'),
                    'launch',
                    'bunker_robot_server.launch.py'
                ])
            ])
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('usb_cam'),
                    'launch',
                    'camera.launch.py'
                ])
            ])
        ),


    ])
