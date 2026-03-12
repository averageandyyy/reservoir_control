import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node


class SimulatedReservoirNode(Node):
    """
    Generic ROS2 node for simulated reservoir computing systems.

    Wraps any SimulatedReservoir instance and exposes standard interfaces.
    Reservoir type and configuration loaded from parameters.
    """

    def __init__(self):
        super().__init__(
            "simulated_reservoir_node",
        )

        # Declare and get parameters
        self.reservoir_type = (
            self.declare_parameter("reservoir_type", "double_pendulum")
            .get_parameter_value()
            .string_value
        )

        self.config_file = (
            get_package_share_directory("reservoir_control")
            + "/config/simulated_"
            + self.reservoir_type
            + ".yaml"
        )

        self.initialize_reservoir()
        self.reservoir.reset()

        self.get_logger().info(
            f"SimulatedReservoirNode initialized with reservoir type: {self.reservoir_type}"
        )

    def initialize_reservoir(self):
        with open(self.config_file, "r") as f:
            full_config = yaml.safe_load(f)

        self.config = full_config["/" + self.get_name()]["ros__parameters"]["reservoir_parameters"]

        for param_name, param_value in self.config.items():
            if param_name.endswith("file"):
                self.config[param_name] = (
                    get_package_share_directory("reservoir_control")
                    + "/config/"
                    + param_value
                )
            self.get_logger().info(
                f"Setting reservoir parameter: {param_name} = {self.config[param_name]}"
            )

        self.create_reservoir()

    def create_reservoir(self):
        if self.reservoir_type == "double_pendulum":
            from reservoir_control.simulated_reservoirs import DoublePendulumReservoir

            self.reservoir = DoublePendulumReservoir(**self.config)
        else:
            self.get_logger().error(f"Unknown reservoir type: {self.reservoir_type}")
            raise ValueError(f"Unknown reservoir type: {self.reservoir_type}")


def main(args=None):
    rclpy.init(args=args)
    node = SimulatedReservoirNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
