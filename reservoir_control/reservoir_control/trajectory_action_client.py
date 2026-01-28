import matplotlib.pyplot as plt
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node

from reservoir_msgs.action import Trajectory


class TrajectoryActionClient(Node):
    def __init__(self):
        super().__init__("trajectory_action_client")
        self._action_client = ActionClient(self, Trajectory, "trajectory")
        self.setpoints_x = []
        self.setpoints_y = []

    def send_goal(self, waypoints):
        """
        Send a trajectory goal with the given waypoints.

        Args:
            waypoints: List of tuples [(x1, y1), (x2, y2), ...]
        """
        self.get_logger().info("Waiting for action server...")
        self._action_client.wait_for_server()

        # Clear previous setpoints
        self.setpoints_x = []
        self.setpoints_y = []

        # Create path message
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        for x, y in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            path.poses.append(pose)

        goal_msg = Trajectory.Goal()
        goal_msg.path = path

        self.get_logger().info(f"Sending goal with {len(waypoints)} waypoints")
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Progress: {feedback.progress:.2f}%")

        # Accumulate setpoints
        self.setpoints_x.append(feedback.setpoint.x)
        self.setpoints_y.append(feedback.setpoint.y)

    def get_result_callback(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info("Trajectory completed successfully!")
            self.plot_trajectory()
        else:
            self.get_logger().info("Trajectory failed")

    def plot_trajectory(self):
        """Plot the accumulated setpoints."""
        if len(self.setpoints_x) == 0:
            self.get_logger().warn("No setpoints to plot")
            return

        plt.figure(figsize=(8, 8))
        plt.plot(
            self.setpoints_x, self.setpoints_y, "b-", linewidth=2, label="Trajectory"
        )
        plt.plot(
            self.setpoints_x[0], self.setpoints_y[0], "go", markersize=10, label="Start"
        )
        plt.plot(
            self.setpoints_x[-1], self.setpoints_y[-1], "ro", markersize=10, label="End"
        )
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Trajectory Setpoints")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()

        self.get_logger().info(f"Plotted {len(self.setpoints_x)} setpoints")


def main(args=None):
    rclpy.init(args=args)

    action_client = TrajectoryActionClient()

    # Example waypoints - modify as needed
    waypoints = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]

    action_client.send_goal(waypoints)

    rclpy.spin(action_client)

    action_client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
