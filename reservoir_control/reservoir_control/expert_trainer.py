import pickle
import threading

import joblib
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from simple_pid import PID
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from std_srvs.srv import Trigger
from tf_transformations import euler_from_quaternion

from reservoir_control.simulated_reservoirs import SimulatedReservoir
from reservoir_control.trajectory_generator import TrajectoryGenerator
from reservoir_control.utils import (
    RobotState2D,
    get_local_error,
    initialize_reservoir,
    pi_clip,
    scale_input,
)
from reservoir_msgs.action import Trajectory


class ExpertTrainerNode(Node):
    """
    This class will take the place of an expert policy and run the reservoir alongside to help with feature collection.

    1. Support Trajectory action and use expert PID controller to follow the trajectory.
    2. During the execution of the trajectory, collect reservoir states and expert control outputs. Data can be
    collected across multiple trajectories. Training data consists of reservoir features and expert control outputs.
    3. Invoke ridge regression fitting via Trigger service to train and save linear mapping for use in SimulatedReservoirNode.

    4. Possibly allow evaluation of learned policy by switching from expert PID control to reservoir control after training.

    Subscribes to:
        1. Odometry
    Publishes to:
        1. Velocity command(s)
        2. Errors
    Actions:
        1. Trajectory action server to receive trajectories to follow.
    Services:
        1. Trigger service to perform ridge regression training and save output parameters.
    """

    def __init__(self):
        super().__init__("expert_trainer_node")

        self._declare_parameters()
        self._get_parameters()

        self.reservoir = initialize_reservoir(
            self.reservoir_type, self.config_file, self.get_name(), self.get_logger()
        )

        self.training_ridge = Ridge(alpha=self.alpha)
        self.scaler = StandardScaler()

        self.get_logger().info(
            f"ExpertTrainerNode initialized with reservoir type: {self.reservoir_type}"
        )

        self._initialize_trajectory_generator_and_pid()

        self.add_post_set_parameters_callback(self._post_param_set_callback)

        # Trajectory action server
        self.trajectory_handle = None
        self.trajectory_lock = threading.Lock()
        self.trajectory_action_server = ActionServer(
            self,
            Trajectory,
            "trajectory",
            execute_callback=self._execute_trajectory_callback,
            goal_callback=self._goal_callback,
            handle_accepted_callback=self._handle_accepted_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info("Trajectory action server initialized.")

        # Robot pose and command
        self.state_lock = threading.Lock()
        self.state: RobotState2D = None
        self.state_subscription = self.create_subscription(
            Odometry, "/odom", self._odom_callback, 10
        )
        self.cmd_publisher = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.get_logger().info(
            "Odometry subscription and cmd_vel publisher initialized."
        )

        # Service to trigger ridge regression training and saving
        self.train_service = self.create_service(
            Trigger, "train_reservoir_ridge", self._train_and_save_ridge
        )
        self.enable_training_service = self.create_service(
            Trigger, "enable_training", self._enable_training_callback
        )
        self.clear_training_data_service = self.create_service(
            Trigger, "clear_training_data", self._clear_training_data_callback
        )
        # Training data storage
        self.is_training = False
        self.X_train_data = []  # Reservoir features
        self.Y_train_data = []  # Expert control outputs
        self.get_logger().info(
            "Training services initialized and training data storage set up."
        )

        # Reservoir control. Reservoir control and reservoir training are mutually exclusive.
        self.use_reservoir_control = False
        self.use_reservoir_control_service = self.create_service(
            Trigger, "use_reservoir_control", self._use_reservoir_control_callback
        )

        self.get_logger().info("ExpertTrainerNode setup complete.")

    def _declare_parameters(self):
        self.declare_parameter("reservoir_type", "double_pendulum")
        self.declare_parameter("ridge_alpha", 1.0)
        self.declare_parameter("output_scaler_file", "scaler.pkl")
        self.declare_parameter("output_ridge_file", "ridge.pkl")
        self.declare_parameter("average_speed", 1.0)
        self.declare_parameter("linear_kp", 1.0)
        self.declare_parameter("linear_ki", 0.0)
        self.declare_parameter("linear_kd", 0.1)
        self.declare_parameter("angular_kp", 1.0)
        self.declare_parameter("angular_ki", 0.0)
        self.declare_parameter("angular_kd", 0.1)
        self.declare_parameter("v_max", 2.0)
        self.declare_parameter("w_max", 4.0)
        self.declare_parameter("test_size", 0.2)
        self.declare_parameter("seed", 42)

    def _get_parameters(self):
        self.reservoir_type = (
            self.get_parameter("reservoir_type").get_parameter_value().string_value
        )
        self.alpha = (
            self.get_parameter("ridge_alpha").get_parameter_value().double_value
        )
        output_scaler_filename = (
            self.get_parameter("output_scaler_file").get_parameter_value().string_value
        )
        output_ridge_filename = (
            self.get_parameter("output_ridge_file").get_parameter_value().string_value
        )
        self.average_speed = (
            self.get_parameter("average_speed").get_parameter_value().double_value
        )
        self.linear_kp = (
            self.get_parameter("linear_kp").get_parameter_value().double_value
        )
        self.linear_ki = (
            self.get_parameter("linear_ki").get_parameter_value().double_value
        )
        self.linear_kd = (
            self.get_parameter("linear_kd").get_parameter_value().double_value
        )
        self.angular_kp = (
            self.get_parameter("angular_kp").get_parameter_value().double_value
        )
        self.angular_ki = (
            self.get_parameter("angular_ki").get_parameter_value().double_value
        )
        self.angular_kd = (
            self.get_parameter("angular_kd").get_parameter_value().double_value
        )
        self.v_max = self.get_parameter("v_max").get_parameter_value().double_value
        self.w_max = self.get_parameter("w_max").get_parameter_value().double_value
        self.test_size = (
            self.get_parameter("test_size").get_parameter_value().double_value
        )
        self.seed = self.get_parameter("seed").get_parameter_value().integer_value

        self.config_file = (
            get_package_share_directory("reservoir_control")
            + "/config/simulated_"
            + self.reservoir_type
            + ".yaml"
        )
        self.output_scaler_file = (
            get_package_share_directory("reservoir_control")
            + "/config/"
            + output_scaler_filename
        )
        self.output_ridge_file = (
            get_package_share_directory("reservoir_control")
            + "/config/"
            + output_ridge_filename
        )

    def _post_param_set_callback(self, params):
        self._get_parameters()
        self._initialize_trajectory_generator_and_pid()

    def _initialize_trajectory_generator_and_pid(self):
        self.trajectory_generator = TrajectoryGenerator(self.average_speed)
        self.pid_linear = PID(
            self.linear_kp,
            self.linear_ki,
            self.linear_kd,
            setpoint=0,  # Target distance is zero
            output_limits=(0, self.v_max),  # Only forward speed, max at v_max
        )
        self.pid_angular = PID(
            self.angular_kp,
            self.angular_ki,
            self.angular_kd,
            setpoint=0,  # Target angle error is zero
            error_map=pi_clip,
            output_limits=(-self.w_max, self.w_max),  # Angular velocity limits
        )

        self.get_logger().info("Trajectory generator and PID controllers initialized.")

    def _train_and_save_ridge(self, request, response):
        """
        Reservoir features and control outputs collected during trajectory following are used to fit a ridge regression model.
        The learned parameters are saved to a file for later use in the SimulatedReservoirNode.
        """
        if len(self.X_train_data) == 0 or len(self.Y_train_data) == 0:
            response.success = False
            response.message = (
                "No training data available. Cannot train ridge regression."
            )
            return response

        X_all, Y_all = np.array(self.X_train_data), np.array(self.Y_train_data)
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_all, Y_all, test_size=self.test_size, random_state=self.seed
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        self.training_ridge.fit(X_train_scaled, Y_train)
        self.get_logger().info(
            f"Ridge regression training complete. Training score: {self.training_ridge.score(X_train_scaled, Y_train):.4f}, Validation score: {self.training_ridge.score(X_val_scaled, Y_val):.4f}",
        )

        # Save learned parameters and scaler to file
        joblib.dump(self.training_ridge, self.output_ridge_file)
        joblib.dump(self.scaler, self.output_scaler_file)

        self.get_logger().info(
            f"Learned ridge regression parameters and scaler saved to {self.output_ridge_file} and {self.output_scaler_file}."
        )

        response.success = True
        response.message = "Ridge regression training and saving complete."
        return response

    def _goal_callback(self, goal_request):
        self.get_logger().info("Received trajectory goal request.")
        return rclpy.action.GoalResponse.ACCEPT

    def _handle_accepted_callback(self, goal_handle):
        with self.trajectory_lock:
            # Ensure only one trajectory is executed at a time
            if self.trajectory_handle is not None and self.trajectory_handle.is_active:
                self.get_logger().info(
                    "Aborting current active trajectory to accept new goal."
                )
                self.trajectory_handle.abort()
            self.trajectory_handle = goal_handle

        goal_handle.execute()

    def _cancel_callback(self, goal):
        self.get_logger().info("Received request to cancel trajectory goal.")
        return rclpy.action.CancelResponse.ACCEPT

    def _execute_trajectory_callback(self, goal_handle):
        self.get_logger().info("Executing trajectory goal.")
        path = goal_handle.request.path
        waypoints = [self.state.as_tuple()] + [
            (pose.pose.position.x, pose.pose.position.y) for pose in path.poses
        ]
        total_time = self.trajectory_generator.generate_trajectory(waypoints)
        res_features = None
        linear_vel, angular_vel = 0.0, 0.0

        # Run warmup sequence if needed before starting trajectory
        if self.is_training or self.use_reservoir_control:
            self.get_logger().info(
                "Running reservoir warmup sequence before trajectory execution."
            )
            local_error = get_local_error(self.state.as_tuple(), waypoints[0])
            self.reservoir.warm_up(scale_input(local_error))
            self.get_logger().info("Reservoir warmup complete.")

        start_time = self.get_clock().now()
        end_time = start_time + rclpy.duration.Duration(seconds=total_time)

        while self.get_clock().now() < end_time:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Trajectory goal canceled.")
                goal_handle.canceled()
                return Trajectory.Result(success=False)

            if not goal_handle.is_active:
                self.get_logger().info("Trajectory goal no longer active.")
                return Trajectory.Result(success=False)

            # Trajectory querying
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            x, y, progress = self.trajectory_generator.query_trajectory(elapsed)

            # Error computation
            local_error = get_local_error(
                self.state.as_tuple(), (x, y)
            )  # [dist, heading_error]

            # Reservoir feature generation
            if self.is_training or self.use_reservoir_control:
                res_input = scale_input(local_error)
                res_features = self.reservoir.step(res_input)

            # Control
            if self.use_reservoir_control:
                action = self.training_ridge.predict(
                    self.scaler.transform(res_features.reshape(1, -1))
                )[0]
                linear_vel = np.clip(action[0], 0.0, self.v_max)
                angular_vel = np.clip(action[1], -self.w_max, self.w_max)
            else:
                linear_vel = self.pid_linear(local_error[0])
                angular_vel = self.pid_angular(local_error[1])
                # Reservoir feature collection
                if self.is_training:
                    self.X_train_data.append(res_features)
                    self.Y_train_data.append(np.array([linear_vel, angular_vel]))

            cmd_msg = TwistStamped()
            cmd_msg.header.stamp = self.get_clock().now().to_msg()
            cmd_msg.twist.linear.x = linear_vel
            cmd_msg.twist.angular.z = angular_vel
            self.cmd_publisher.publish(cmd_msg)

            # Feedback
            feedback = Trajectory.Feedback()
            feedback.progress = progress
            feedback.setpoint.x = x
            feedback.setpoint.y = y
            goal_handle.publish_feedback(feedback)

        self.get_logger().info("Trajectory goal completed successfully.")
        goal_handle.succeed()
        return Trajectory.Result(success=True)

    def _odom_callback(self, msg):
        with self.state_lock:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            orientation_q = msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion(
                [
                    orientation_q.x,
                    orientation_q.y,
                    orientation_q.z,
                    orientation_q.w,
                ]
            )
            self.state = RobotState2D(x, y, yaw)

    def _enable_training_callback(self, request, response):
        self.is_training = not self.is_training
        response.success = True
        response.message = (
            f"Training mode {'enabled' if self.is_training else 'disabled'}."
        )
        return response

    def _clear_training_data_callback(self, request, response):
        self.X_train_data.clear()
        self.Y_train_data.clear()
        response.success = True
        response.message = "Training data cleared."
        return response

    def _use_reservoir_control_callback(self, request, response):
        self.use_reservoir_control = not self.use_reservoir_control
        response.success = True
        response.message = f"Reservoir control {'enabled' if self.use_reservoir_control else 'disabled'}."
        return response


def main(args=None):
    rclpy.init(args=args)

    expert_trainer_node = ExpertTrainerNode()

    executor = MultiThreadedExecutor()
    rclpy.spin(expert_trainer_node, executor=executor)

    expert_trainer_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
