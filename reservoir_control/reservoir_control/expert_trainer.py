import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from simple_pid import PID
from sklearn.linear_model import Ridge

from reservoir_control.simulated_reservoirs import SimulatedReservoir
from reservoir_control.trajectory_generator import TrajectoryGenerator
from reservoir_control.utils import initialize_reservoir, pi_clip
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

        reservoir = initialize_reservoir(
            self.reservoir_type, self.config_file, self.get_name(), self.get_logger()
        )

        self.training_ridge = Ridge(alpha=self.alpha)

        self.get_logger().info(
            f"ExpertTrainerNode initialized with reservoir type: {self.reservoir_type}"
        )

        self._initialize_trajectory_generator_and_pid()

        self.add_post_set_parameters_callback(self._post_param_set_callback)

        self.get_logger().info("ExpertTrainerNode setup complete.")

    def _declare_parameters(self):
        self.declare_parameter("reservoir_type", "double_pendulum")
        self.declare_parameter("ridge_alpha", 1.0)
        self.declare_parameter("output_params_file", "output_params.npy")
        self.declare_parameter("average_speed", 1.0)
        self.declare_parameter("linear_kp", 1.0)
        self.declare_parameter("linear_ki", 0.0)
        self.declare_parameter("linear_kd", 0.1)
        self.declare_parameter("angular_kp", 1.0)
        self.declare_parameter("angular_ki", 0.0)
        self.declare_parameter("angular_kd", 0.1)

    def _get_parameters(self):
        self.reservoir_type = (
            self.get_parameter("reservoir_type").get_parameter_value().string_value
        )
        self.alpha = (
            self.get_parameter("ridge_alpha").get_parameter_value().double_value
        )
        output_params_filename = (
            self.get_parameter("output_params_file").get_parameter_value().string_value
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

        self.config_file = (
            get_package_share_directory("reservoir_control")
            + "/config/simulated_"
            + self.reservoir_type
            + ".yaml"
        )
        self.output_params_file = (
            get_package_share_directory("reservoir_control")
            + "/config/"
            + output_params_filename
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
            setpoint=0,
        )
        self.pid_angular = PID(
            self.angular_kp,
            self.angular_ki,
            self.angular_kd,
            setpoint=0,
            error_map=pi_clip,
        )

        self.get_logger().info("Trajectory generator and PID controllers initialized.")

    def _train_and_save_ridge(self):
        """
        Reservoir features and control outputs collected during trajectory following are used to fit a ridge regression model.
        The learned parameters are saved to a file for later use in the SimulatedReservoirNode.
        """
        pass


def main(args=None):
    rclpy.init(args=args)

    expert_trainer_node = ExpertTrainerNode()

    executor = MultiThreadedExecutor()
    rclpy.spin(expert_trainer_node, executor=executor)

    expert_trainer_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
