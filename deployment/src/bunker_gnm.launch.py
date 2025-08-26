from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
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
                    FindPackageShare('depthai_ros_driver'),
                    'launch',
                    'camera.launch.py'
                ])
            ])
        ),


    ])
