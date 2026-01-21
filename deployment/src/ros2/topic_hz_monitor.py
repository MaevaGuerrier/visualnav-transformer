#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from collections import deque
import time
import yaml
from rosidl_runtime_py.utilities import get_message
import importlib

class TopicRateMonitor(Node):
    def __init__(self, topics=None, expected_rates=None, rate_tolerance=0.3, window_size=10):
        super().__init__('topic_rate_monitor')
        
        assert topics is not None, "You must provide a list of topics to monitor."
        assert expected_rates is not None, "You must provide a list of expected rates for the topics."

        # Parameters
        self.declare_parameter('topics', topics)
        self.declare_parameter('expected_rates', expected_rates)
        self.declare_parameter('rate_tolerance', rate_tolerance)
        self.declare_parameter('window_size', window_size)
        
        self.topics_to_monitor = self.get_parameter('topics').value
        self.expected_rates = self.get_parameter('expected_rates').value
        self.rate_tolerance = self.get_parameter('rate_tolerance').value
        self.window_size = self.get_parameter('window_size').value
        
        # Storage for timing data
        self.topic_times = {}
        self.subscribers = {}
        
        # Diagnostics publisher
        self.diag_pub = self.create_publisher(DiagnosticArray, '/topics_rate', 10)
        
        # Subscribe to topics
        for i, topic in enumerate(self.topics_to_monitor):
            self.topic_times[topic] = {
                'times': deque(maxlen=self.window_size),
                'expected_rate': self.expected_rates[i] if i < len(self.expected_rates) else 10.0,
                'last_msg_time': None
            }
            
            # Get topic type
            try:
                topic_type = self.get_topic_type(topic)
                if topic_type:
                    msg_class = self.get_message_class(topic_type)
                    if msg_class:
                        self.subscribers[topic] = self.create_subscription(
                            msg_class,
                            topic,
                            lambda msg, topic_name=topic: self.topic_callback(msg, topic_name),
                            10
                        )
                        self.get_logger().info(f"Monitoring topic: {topic} (expected rate: {self.topic_times[topic]['expected_rate']} Hz)")
                    else:
                        self.get_logger().warn(f"Could not load message class for topic: {topic}")
                else:
                    self.get_logger().warn(f"Could not determine type for topic: {topic}")
            except Exception as e:
                self.get_logger().error(f"Failed to subscribe to {topic}: {e}")
        
        # Timer for publishing diagnostics (1 Hz)
        self.timer = self.create_timer(1.0, self.publish_diagnostics)
        
    def get_topic_type(self, topic_name):
        """Get the message type for a topic"""
        topic_names_and_types = self.get_topic_names_and_types()
        
        for name, types in topic_names_and_types:
            if name == topic_name:
                if types:
                    return types[0]  # Return first type
        return None
    
    def get_message_class(self, topic_type):
        """Convert topic type string to message class"""
        try:
            # topic_type format: 'package/msg/MessageType'
            parts = topic_type.split('/')
            if len(parts) == 3:
                package_name = parts[0]
                msg_name = parts[2]
                
                # Import the message module
                module = importlib.import_module(f'{package_name}.msg')
                msg_class = getattr(module, msg_name)
                return msg_class
            else:
                self.get_logger().error(f"Invalid topic type format: {topic_type}")
                return None
        except Exception as e:
            self.get_logger().error(f"Failed to load message class for {topic_type}: {e}")
            return None
    
    def topic_callback(self, msg, topic_name):
        """Callback for each monitored topic"""
        current_time = time.time()
        
        if self.topic_times[topic_name]['last_msg_time'] is not None:
            dt = current_time - self.topic_times[topic_name]['last_msg_time']
            self.topic_times[topic_name]['times'].append(dt)
        
        self.topic_times[topic_name]['last_msg_time'] = current_time
    
    def calculate_rate(self, topic_name):
        """Calculate current rate for a topic"""
        times = self.topic_times[topic_name]['times']
        
        if len(times) < 2:
            return None
        
        # Average time between messages
        avg_dt = sum(times) / len(times)
        
        if avg_dt > 0:
            return 1.0 / avg_dt
        return None
    
    def publish_diagnostics(self):
        """Publish diagnostic messages"""
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for topic_name, data in self.topic_times.items():
            status = DiagnosticStatus()
            status.name = f"Topic Rate: {topic_name}"
            status.hardware_id = topic_name
            
            current_rate = self.calculate_rate(topic_name)
            expected_rate = data['expected_rate']
            
            if current_rate is None:
                status.level = DiagnosticStatus.STALE
                status.message = "Waiting for messages..."
                status.values.append(KeyValue(key="Expected Rate (Hz)", value=str(expected_rate)))
                status.values.append(KeyValue(key="Current Rate (Hz)", value="N/A"))
                self.get_logger().warn(f"No messages received yet on topic: {topic_name}", throttle_duration_sec=5.0)
            else:
                # Check if rate is within tolerance
                min_rate = expected_rate * (1 - self.rate_tolerance)
                max_rate = expected_rate * (1 + self.rate_tolerance)
                
                rate_error = abs(current_rate - expected_rate) / expected_rate
                
                if current_rate < min_rate:
                    status.level = DiagnosticStatus.ERROR
                    status.message = f"{topic_name} Rate too low! {current_rate:.2f} Hz (expected {expected_rate:.2f} Hz)"
                    self.get_logger().warn(status.message, throttle_duration_sec=5.0)
                # elif current_rate > max_rate:
                #     status.level = DiagnosticStatus.WARN
                #     status.message = f"{topic_name} Rate too high! {current_rate:.2f} Hz (expected {expected_rate:.2f} Hz)"
                #     self.get_logger().warn(status.message, throttle_duration_sec=5.0)
                # else:
                #     status.level = DiagnosticStatus.OK
                #     status.message = f"Rate OK: {current_rate:.2f} Hz"
                
                status.values.append(KeyValue(key="Expected Rate (Hz)", value=str(expected_rate)))
                status.values.append(KeyValue(key="Current Rate (Hz)", value=f"{current_rate:.2f}"))
                status.values.append(KeyValue(key="Rate Error (%)", value=f"{rate_error*100:.1f}"))
                status.values.append(KeyValue(key="Min Acceptable (Hz)", value=f"{min_rate:.2f}"))
                status.values.append(KeyValue(key="Max Acceptable (Hz)", value=f"{max_rate:.2f}"))
                status.values.append(KeyValue(key="Messages in Window", value=str(len(data['times']))))
            
            msg.status.append(status)
        
        self.diag_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Load configuration files
        with open('/workspace/src/visualnav-transformer/deployment/config/monitor.yaml', 'r') as f:
            monitor_cfg = yaml.safe_load(f)

        with open('/workspace/src/visualnav-transformer/deployment/config/robot.yaml', 'r') as f:
            robot_cfg = yaml.safe_load(f)

        topics = list(monitor_cfg['topics'].keys())
        expected_rates = list(monitor_cfg['topics'].values())

        cmd_vel_rate = robot_cfg["frame_rate"]
        cmd_vel_topic = robot_cfg["vel_navi_topic"]

        topics.append(cmd_vel_topic)
        expected_rates.append(cmd_vel_rate)

        # Assign /waypoint rate with cmd_vel rate
        if '/waypoint' in topics:
            expected_rates[topics.index('/waypoint')] = cmd_vel_rate

        print(f"Monitoring topics: {topics} with expected rates: {expected_rates}")

        monitor = TopicRateMonitor(topics=topics, expected_rates=expected_rates)
        rclpy.spin(monitor)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()