"""opencvでカメラから位置情報取得
後回し
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

import pigpio
import time

import numpy as np
import math
import cv2