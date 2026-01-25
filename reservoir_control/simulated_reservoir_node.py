import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.parameter import parameter_dict_from_yaml_file, parameter_value_to_python


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
        self.config = {}
        prefix = "reservoir_parameters."
        params = parameter_dict_from_yaml_file(
            self.config_file, target_nodes=["/" + self.get_name()]
        )

        for key in params:
            if key.startswith(prefix):
                param_name = key[len(prefix) :]
                param_value = parameter_value_to_python(params[key]._value)
                if param_name.endswith("file"):
                    param_value = (
                        get_package_share_directory("reservoir_control")
                        + "/config/"
                        + param_value
                    )
                self.get_logger().info(
                    f"Setting reservoir parameter: {param_name} = {param_value}"
                )
                self.config[param_name] = param_value

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
