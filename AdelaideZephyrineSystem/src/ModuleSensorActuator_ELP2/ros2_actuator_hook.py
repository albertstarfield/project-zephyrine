import json
import re
import time

try:
    import rclpy
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# We use a global node so we don't initialize/shutdown rclpy per hook execution
# In a real environment, the StellaIcarus daemon would maintain this node.
_ROS2_NODE = None

def get_ros2_node():
    global _ROS2_NODE
    if not ROS2_AVAILABLE:
        return None

    if _ROS2_NODE is None:
        if not rclpy.ok():
            rclpy.init(args=None)
        _ROS2_NODE = rclpy.create_node('adelaide_actuator_hook')
    return _ROS2_NODE

# 1. Provide a compiled regex PATTERN that must match the full user input
PATTERN = re.compile(r"^actuate\s+(?P<servo_id>\w+)\s+(?P<angle>-?\d+\.?\d*)$", re.IGNORECASE)

# 2. Provide the handler function
def handler(match, user_input, session_id):
    if not ROS2_AVAILABLE:
        return "ROS2 Actuator Hook: ERROR - rclpy not available. ROS2 environment is not configured."

    servo_id = match.group("servo_id")
    try:
        angle = float(match.group("angle"))
    except ValueError:
        return f"ROS2 Actuator Hook: ERROR - Invalid angle parameter '{match.group('angle')}'"

    node = get_ros2_node()
    if node is None:
        return "ROS2 Actuator Hook: ERROR - Failed to initialize ROS2 node."

    try:
        # Publish to a standard ROS2 topic for actuators (e.g. /cmd_actuator)
        publisher = node.create_publisher(String, '/cmd_actuator', 10)

        # In a real system, you'd use a specific message type like sensor_msgs/JointState
        msg = String()

        # Serialize the command
        payload = {
            "servo_id": servo_id,
            "angle": angle,
            "timestamp": time.time(),
            "priority": "ELP3" # Indicate high priority/low latency
        }
        msg.data = json.dumps(payload)

        publisher.publish(msg)

        # Give ROS2 DDS a tiny moment to send the message before returning
        time.sleep(0.001)

        # Cleanup publisher to avoid memory leak if called frequently
        node.destroy_publisher(publisher)

        return f"[StellaIcarus-ELP3] Published actuation command to {servo_id} for angle {angle}° via ROS2."
    except Exception as e:
        return f"ROS2 Actuator Hook: FATAL EXCEPTION - {e!s}"
