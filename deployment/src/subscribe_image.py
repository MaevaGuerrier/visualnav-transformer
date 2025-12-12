
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
# from cv_bridge import CvBridge

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        
        # Create QoS profile matching the publisher
        # qos_profile = QoSProfile(
        #     reliability=QoSReliabilityPolicy.RELIABLE,
        #     history=QoSHistoryPolicy.KEEP_LAST,
        #     depth=10,  # Choose appropriate depth
        #     durability=QoSDurabilityPolicy.VOLATILE
        # )
        
        # Create subscription
        self.subscription = self.create_subscription(
            Image,
            '/usb_cam/image_raw',  # Replace with your actual topic name
            self.image_callback,
            10
        )
        
        # CvBridge to convert ROS Image to OpenCV format (optional)
        # self.bridge = CvBridge()
        
        self.get_logger().info('Image subscriber initialized')
    
    def image_callback(self, msg):
        self.get_logger().info(f'Received image: {msg.width}x{msg.height}')
        
        # Optional: Convert to OpenCV format and display
        # try:
        #     cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #     cv2.imshow('Camera Feed', cv_image)
        #     cv2.waitKey(1)
        # except Exception as e:
        #     self.get_logger().error(f'Error converting image: {e}')

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    
    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        image_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()