#!/usr/bin/env python3
# Only work on NVIDIA Jetson platforms with tegrastats available
# Make sure you specified -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro when launching the docker container

import rclpy
from rclpy.node import Node
import subprocess
import re
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

def parse_tegrastats(line):
    """Parse tegrastats output and return averages"""
    data = {}
    
    try:
        # GPU usage
        gpu_match = re.search(r'GR3D_FREQ\s+(\d+)%', line)
        if gpu_match:
            data['gpu'] = float(gpu_match.group(1))
        
        # RAM usage
        ram_match = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
        if ram_match:
            used = int(ram_match.group(1))
            total = int(ram_match.group(2))
            data['memory'] = round((used / total) * 100, 1)
        
        # CPU usage - get all CPU percentages and average them
        cpu_matches = re.findall(r'(\d+)%@\d+', line)
        if cpu_matches:
            cpu_values = [float(x) for x in cpu_matches]
            data['cpu'] = round(sum(cpu_values) / len(cpu_values), 1)
        
    except Exception as e:
        # Note: logger not available in this function
        print(f"Parse error: {e}")
    
    return data

class SimpleJetsonMonitor(Node):
    def __init__(self):
        super().__init__('simple_jetson_monitor')
        
        self.publisher_ = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        
        self.get_logger().info("Simple Jetson Monitor started")
        
        # Start tegrastats
        try:
            self.process = subprocess.Popen(
                ['tegrastats', '--interval', '1000'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
        except FileNotFoundError:
            self.get_logger().error("tegrastats not found!")
            raise
        
        # Create a timer to read tegrastats output
        self.timer = self.create_timer(0.1, self.timer_callback)
        
    def timer_callback(self):
        """Read and publish tegrastats data"""
        try:
            line = self.process.stdout.readline()
            
            if not line:
                return
            
            data = parse_tegrastats(line.strip())
            
            if not data:
                return
            
            # Create diagnostic message
            msg = DiagnosticArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            
            status = DiagnosticStatus()
            status.name = "Jetson Usage"
            status.hardware_id = "Jetson"
            status.level = DiagnosticStatus.OK
            status.message = "System OK"
            
            # Add simple values
            if 'gpu' in data:
                status.values.append(KeyValue(key="GPU (%)", value=str(data['gpu'])))
            
            if 'cpu' in data:
                status.values.append(KeyValue(key="CPU (%)", value=str(data['cpu'])))
            
            if 'memory' in data:
                status.values.append(KeyValue(key="Memory (%)", value=str(data['memory'])))
            
            msg.status.append(status)
            self.publisher_.publish(msg)
            
        except Exception as e:
            self.get_logger().warn(f"Error in timer callback: {e}")
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        self.get_logger().info("Shutting down...")
        if hasattr(self, 'process'):
            self.process.terminate()
            self.process.wait()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SimpleJetsonMonitor()
    try:
        
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()