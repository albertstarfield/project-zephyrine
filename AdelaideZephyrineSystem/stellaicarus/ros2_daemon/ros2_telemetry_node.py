#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import json
import time

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:
    # Fail gracefully if ROS2 is not installed or available
    print("{\"error\": \"ROS2 (rclpy) not found. ROS2 Telemetry node disabled.\", \"elp_level\": 0}")
    sys.exit(0)

class AdelaideRos2TelemetryNode(Node):
    def __init__(self):
        super().__init__('adelaide_telemetry_node')
        
        # Example subscription - change this to actual hardware topics like sensor_msgs/msg/JointState
        self.subscription = self.create_subscription(
            String,
            '/robot_telemetry',
            self.listener_callback,
            10
        )
        
        # We output to stdout as JSON so the StellaIcarus daemon manager can parse it
        # and send it to the Ada backend with ELP2/3 priority tagging.
        self.get_logger().info('Adelaide ROS2 Telemetry Node started.')

    def listener_callback(self, msg):
        # We tag this as ELP2 to ensure deterministic low-latency handling in the Ada server
        payload = {
            "source": "ros2_telemetry",
            "elp_level": 2,
            "data": msg.data,
            "timestamp": time.time()
        }
        
        # The communication protocol requires a single line valid JSON string to stdout
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

def main(args=None):
    rclpy.init(args=args)
    node = AdelaideRos2TelemetryNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy.shutdown() throws error if already shutdown, but we should be clean
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    # Disable buffering to ensure immediate transmission to the daemon manager
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()
