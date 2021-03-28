"""サーボモータを操作"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
# from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Int8

import Adafruit_PCA9685







class Servo(Node):
    def __init__(self, servo_name, servo_id):
        super().__init__('servo_node_' + servo_name)
        self.init_pca9685()
        self.servo_name = servo_name
        self.servo_id = servo_id
        self.sub_topic_name = '/output/servo/' + servo_name
        self.sub_servo = self.create_subscription(Int8, self.sub_topic_name, self.servo_callback, 10)
    
    
    
