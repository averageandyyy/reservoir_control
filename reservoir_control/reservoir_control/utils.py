import math

import numpy as np
from ament_index_python.packages import get_package_share_directory
from rclpy.logging import RcutilsLogger
from rclpy.parameter import parameter_dict_from_yaml_file, parameter_value_to_python


def create_reservoir(reservoir_type: str, config: dict, logger: RcutilsLogger):
    if reservoir_type == "double_pendulum":
        from reservoir_control.simulated_reservoirs import DoublePendulumReservoir

        return DoublePendulumReservoir(**config)
    else:
        logger.error(f"Unknown reservoir type: {reservoir_type}")
        raise ValueError(f"Unknown reservoir type: {reservoir_type}")


def initialize_reservoir(
    reservoir_type: str, config_file: str, node_name: str, logger: RcutilsLogger
):
    config = {}
    prefix = "reservoir_parameters."
    params = parameter_dict_from_yaml_file(config_file, target_nodes=["/" + node_name])

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
            logger.info(f"Setting reservoir parameter: {param_name} = {param_value}")
            config[param_name] = param_value

    return create_reservoir(reservoir_type, config, logger)


def pi_clip(angle):
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return angle


def body_error(robot_state, target):
    """Compute body-frame error (ex, ey)."""
    rx, ry, rtheta = robot_state
    tx, ty = target
    dx, dy = tx - rx, ty - ry

    c, s = np.cos(rtheta), np.sin(rtheta)
    ex = c * dx + s * dy
    ey = -s * dx + c * dy
    return np.array([ex, ey], dtype=float)


def get_local_error(robot_state, target):
    """Compute [dist, heading_error] - raw input for reservoir (like working version)."""
    rx, ry, rtheta = robot_state
    tx, ty = target
    dx, dy = tx - rx, ty - ry
    dist = np.sqrt(dx**2 + dy**2)
    global_angle = np.arctan2(dy, dx)
    heading_error = np.arctan2(
        np.sin(global_angle - rtheta), np.cos(global_angle - rtheta)
    )
    return np.array([dist, heading_error])
