import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import ChangeState
from nav_msgs.msg import Path
from rcl_interfaces.msg import Parameter, ParameterType
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from reservoir_msgs.action import Trajectory


class PIDTunerCLI(Node):
    def __init__(self):
        super().__init__("pid_tuner_cli")

        # Action Client for trajectories
        self._action_client = ActionClient(self, Trajectory, "trajectory")

        # Service Clients for expert trainer parameters
        self.get_params_client = self.create_client(
            GetParameters, "/expert_trainer_node/get_parameters"
        )
        self.set_params_client = self.create_client(
            SetParameters, "/expert_trainer_node/set_parameters"
        )

        # Service Clients for expert trainer actions
        self.srv_train = self.create_client(Trigger, "/train_reservoir_ridge")
        self.srv_enable_train = self.create_client(Trigger, "/enable_training")
        self.srv_clear_train = self.create_client(Trigger, "/clear_training_data")
        self.srv_use_res = self.create_client(Trigger, "/use_reservoir_control")

        # Service Client for AMCL lifecycle management
        self.amcl_state_client = self.create_client(
            ChangeState, "/amcl/change_state"
        )

        # Publishers for resetting localization
        self.set_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/set_pose", 10
        )
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )

        # Publisher for the resultant image
        self.image_pub = self.create_publisher(Image, "/trajectory_plot", 10)
        self.bridge = CvBridge()

        # Data storage for plotting
        self.setpoints_x = []
        self.setpoints_y = []
        self.current_x = []
        self.current_y = []

        # Target node parameters to manage
        self.managed_params = [
            "linear_kp",
            "linear_ki",
            "linear_kd",
            "angular_kp",
            "angular_ki",
            "angular_kd",
            "average_speed",
            "v_max",
            "w_max",
        ]
        self.current_param_values = {param: 0.0 for param in self.managed_params}
        
        # Predefined Trajectories
        self.trajectories = {
            "CW Square": [(0.5, 0.0), (0.5, -0.5), (0.0, -0.5), (0.0, 0.0)],
            "CCW Square": [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (0.0, 0.0)],
            "CW Diagonal": [(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5), (0.0, 0.0)],
            "CCW Diagonal": [(0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.0, 0.0)],
            "Original CW": [(0.5, 0.0), (0.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.0, 0.0)]
        }

    def wait_for_services(self, timeout_sec=3.0):
        if not self.get_params_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().warn(
                "Could not connect to /expert_trainer_node/get_parameters"
            )
            return False
        return True

    def fetch_current_parameters(self):
        if not self.get_params_client.wait_for_service(timeout_sec=1.0):
            return

        req = GetParameters.Request()
        req.names = self.managed_params
        future = self.get_params_client.call_async(req)
        
        # Process this by waiting for the background spin thread to complete the future
        if self.wait_for_future(future, timeout=2.0):
            for i, val in enumerate(future.result().values):
                self.current_param_values[self.managed_params[i]] = val.double_value
        else:
            self.get_logger().error("Failed to fetch parameters.")

    def wait_for_future(self, future, timeout=5.0):
        """Helper to wait for a future while the background thread is spinning."""
        start_time = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.05)
        return future.done()

    def set_parameter(self, param_name, value):
        if not self.set_params_client.wait_for_service(timeout_sec=1.0):
            print("Set parameter service not available.")
            return

        req = SetParameters.Request()
        param = Parameter()
        param.name = param_name
        param.value.type = ParameterType.PARAMETER_DOUBLE
        param.value.double_value = float(value)
        req.parameters.append(param)

        future = self.set_params_client.call_async(req)
        if self.wait_for_future(future, timeout=2.0):
            res = future.result()
            if res and res.results[0].successful:
                print(f"Successfully updated {param_name} to {value}")
                self.current_param_values[param_name] = float(value)
            else:
                print(f"Failed to update {param_name}: {res.results[0].reason if res else 'Unknown error'}")
        else:
            print(f"Failed to set parameter {param_name} due to timeout.")

    def call_trigger_service(self, client):
        if not client.wait_for_service(timeout_sec=1.0):
            print("Service not available.")
            return
        
        req = Trigger.Request()
        future = client.call_async(req)
        if self.wait_for_future(future, timeout=2.0):
            print(f"Service Response: {future.result().message}")
        else:
            print("Service call timed out.")

    def reset_localization(self):
        print("\nResetting localization...")
        # 1. Reset EKFs via /set_pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        # Identity pose (0,0,0) with high covariance to let AMCL pull it in
        pose_msg.pose.pose.position.x = 0.0
        pose_msg.pose.pose.position.y = 0.0
        pose_msg.pose.pose.position.z = 0.0
        pose_msg.pose.pose.orientation.w = 1.0
        # High covariance
        pose_msg.pose.covariance[0] = 0.5
        pose_msg.pose.covariance[7] = 0.5
        pose_msg.pose.covariance[35] = 0.2
        
        self.set_pose_pub.publish(pose_msg)
        print("- Published zero pose to EKF (/set_pose)")

        # 2. Restart AMCL via Lifecycle Manager
        if self.amcl_state_client.wait_for_service(timeout_sec=1.0):
            req = ChangeState.Request()
            # Deactivate (3)
            req.transition.id = 3 
            future = self.amcl_state_client.call_async(req)
            self.wait_for_future(future)
            # Cleanup (2)
            req.transition.id = 2
            future = self.amcl_state_client.call_async(req)
            self.wait_for_future(future)
            # Configure (1)
            req.transition.id = 1
            future = self.amcl_state_client.call_async(req)
            self.wait_for_future(future)
            # Activate (3)
            req.transition.id = 3
            future = self.amcl_state_client.call_async(req)
            self.wait_for_future(future)
            print("- Cycled AMCL through Complete Lifecycle Restart")
            
            # 3. Publish initial pose to AMCL
            self.initial_pose_pub.publish(pose_msg)
            print("- Published zero pose to AMCL (/initialpose)")
        else:
            print("- WARN: AMCL lifecycle service not found. Make sure nav2_lifecycle_manager is running.")
            
        print("Waiting 2 seconds for filters to settle...")
        time.sleep(2.0)

    def execute_trajectory(self, trajectory_name):
        waypoints = self.trajectories[trajectory_name]
        
        input(f"\n[!!] Ready to execute {trajectory_name}.\n[!!] Please physically place the robot at the origin (0,0).\n[!!] PRESS ENTER to reset localization and start...")

        self.reset_localization()

        if not self._action_client.wait_for_server(timeout_sec=3.0):
            print("Trajectory action server not available.")
            return

        # Clear previous setpoints
        self.setpoints_x = []
        self.setpoints_y = []
        self.current_x = []
        self.current_y = []

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

        print(f"Sending goal with {len(waypoints)} waypoints...")
        future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        
        if not self.wait_for_future(future):
            print("Action goal request timed out.")
            return

        goal_handle = future.result()
        
        if not goal_handle.accepted:
            print("Goal rejected by server.")
            return

        print("Goal accepted! Executing...")
        result_future = goal_handle.get_result_async()
        if self.wait_for_future(result_future, timeout=300.0): # Long timeout for execution
            result = result_future.result().result
            if result.success:
                print("Trajectory completed successfully!")
                self.generate_and_publish_plot()
            else:
                print("Trajectory failed.")
        else:
            print("Trajectory execution timed out.")

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # Accumulate setpoints
        self.setpoints_x.append(feedback.setpoint.x)
        self.setpoints_y.append(feedback.setpoint.y)

        self.current_x.append(feedback.current.x)
        self.current_y.append(feedback.current.y)

    def generate_and_publish_plot(self):
        if len(self.setpoints_x) == 0:
            print("No setpoints to plot.")
            return

        print("Generating and publishing plot...")
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        fig = Figure(figsize=(8, 8))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        ax.plot(self.setpoints_x, self.setpoints_y, "b-", linewidth=2, label="Trajectory")
        ax.plot(self.current_x, self.current_y, "g-", label="Actual")
        ax.plot(self.setpoints_x[0], self.setpoints_y[0], "go", markersize=10, label="Start")
        ax.plot(self.setpoints_x[-1], self.setpoints_y[-1], "ro", markersize=10, label="End")
        
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Trajectory Setpoints vs Actual")
        ax.legend()
        ax.grid(True)
        # Handle axis equal constraint safely in a non-GUI figure
        ax.set_aspect('equal', 'box')

        # Draw the canvas, cache the renderer
        canvas.draw()
        
        # Get the RGBA buffer from the figure
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = image_flat.reshape(canvas.get_width_height()[::-1] + (3,))

        # Publish the image
        img_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(img_msg)
        print("Plot successfully published to /trajectory_plot")


def cli_loop(node):
    while rclpy.ok():
        node.fetch_current_parameters()
        
        print("\n\n" + "="*40)
        print("      EXPERT TRAINER PID TUNER GUI      ")
        print("="*40)
        
        print("\n--- Current Parameters ---")
        for i, param in enumerate(node.managed_params):
            print(f"[{i}] {param}: {node.current_param_values[param]}")
            
        print("\n--- Services ---")
        print("[t] Enable/Disable Training")
        print("[c] Clear Training Data")
        print("[r] Train Ridge Regression")
        print("[u] Toggle Use Reservoir Control")
        
        print("\n--- Exec Trajectory ---")
        keys = list(node.trajectories.keys())
        for i, traj_name in enumerate(keys):
            print(f"[{i + 10}] Execute {traj_name}")
            
        print("\n[q] Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'q':
            print("Exiting PID Tuner CLI...")
            break
            
        # Services
        if choice == 't':
            node.call_trigger_service(node.srv_enable_train)
        elif choice == 'c':
            node.call_trigger_service(node.srv_clear_train)
        elif choice == 'r':
            node.call_trigger_service(node.srv_train)
        elif choice == 'u':
            node.call_trigger_service(node.srv_use_res)
            
        # Parameter editing
        elif choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(node.managed_params):
                param_name = node.managed_params[idx]
                new_val = input(f"Enter new value for {param_name} (Current: {node.current_param_values[param_name]}): ")
                try:
                    node.set_parameter(param_name, float(new_val))
                except ValueError:
                    print("Invalid input! Must be a number.")
            elif 10 <= idx < 10 + len(keys):
                traj_idx = idx - 10
                node.execute_trajectory(keys[traj_idx])
            else:
                print("Invalid choice, out of range.")
        else:
            print("Invalid input.")


def main(args=None):
    rclpy.init(args=args)
    
    node = PIDTunerCLI()
    
    # We will run the ROS spinning logic dynamically in a background thread 
    # to avoid blocking on `input()` in the main thread.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    print("Initializing GUI and waiting for services...")
    node.wait_for_services()
    
    # Run the CLI Loop
    try:
        cli_loop(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        print(f"\n[FATAL ERROR IN CLI LOOP] {e}\n")
        traceback.print_exc()
    finally:
        print("Shutting down node...")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        sys.exit(0)


if __name__ == "__main__":
    main()
