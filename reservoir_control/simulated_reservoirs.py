from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import mujoco
import numpy as np
from sklearn.kernel_approximation import RBFSampler


class SimulatedReservoir(ABC):
    """Generic interface for reservoir computing systems."""

    @abstractmethod
    def step(self, **kwargs) -> np.ndarray:
        """
        Execute one step of the reservoir dynamics.

        Args:
            **kwargs: Flexible input parameters specific to implementation

        Returns:
            np.ndarray: Reservoir features/state
        """
        raise NotImplementedError("Step method must be implemented by subclass.")

    @abstractmethod
    def reset(self, **kwargs) -> None:
        """
        Reset the reservoir to initial state.

        Args:
            **kwargs: Optional reset parameters
        """
        raise NotImplementedError("Reset method must be implemented by subclass.")

    @abstractmethod
    def warm_up(self, **kwargs) -> None:
        """
        Pre-settle the reservoir with initial conditions.

        Args:
            **kwargs: Warm-up configuration (e.g., input signal, steps)
        """
        raise NotImplementedError("Warm-up method must be implemented by subclass.")

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current internal state of reservoir.

        Returns:
            Dict containing relevant state information
        """
        raise NotImplementedError("Get state method must be implemented by subclass.")

    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Get dimensionality of reservoir features.

        Returns:
            int: Feature dimension
        """
        raise NotImplementedError(
            "Get feature dimension method must be implemented by subclass."
        )


class DoublePendulumReservoir(SimulatedReservoir):
    """
    MuJoCo-based double pendulum reservoir implementation.

    Uses a physically simulated double pendulum as the reservoir dynamics
    with RBF feature extraction and leaky integration.
    """

    def __init__(
        self,
        xml_file: str,
        dt_ctrl: float,
        mujoco_substeps: int,
        rbf_gamma: float,
        rbf_components: int,
        ctrl_scale: float,
        leak_rate: float,
        seed: int = 42,
        warmup_steps: int = 30,
    ):
        """
        Initialize double pendulum reservoir.

        Args:
            xml_file: MuJoCo XML model file path
            dt_ctrl: Control timestep
            mujoco_substeps: Number of MuJoCo steps per control step
            rbf_gamma: RBF kernel width parameter
            rbf_components: Number of RBF features
            ctrl_scale: Control input scaling factor
            leak_rate: Leaky integration rate (0-1)
            seed: Random seed for reproducibility
        """
        with open(xml_file, "r") as f:
            xml_string = f.read()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self.mujoco_substeps = mujoco_substeps
        self.ctrl_scale = ctrl_scale
        self.leak_rate = leak_rate
        self.dt_ctrl = dt_ctrl
        self.warmup_steps = warmup_steps

        # RBF feature map with leaky integration
        self.rbf = RBFSampler(
            gamma=rbf_gamma, n_components=rbf_components, random_state=seed
        )
        # Initialize with pendulum state dimension: qpos(2) + qvel(2)
        self.rbf.fit(np.zeros((1, 4)))

        # Internal leaky-integrated state
        self.h = np.zeros(rbf_components)
        self.rbf_components = rbf_components

    def step(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Drive pendulum and return leaky-integrated RBF features.

        Args:
            input_signal (np.ndarray): Control input (2D torque commands)

        Returns:
            np.ndarray: Leaky-integrated RBF features
        """
        # Apply scaled control to pendulum
        self.data.ctrl[0] = input_signal[0] * self.ctrl_scale
        self.data.ctrl[1] = input_signal[1] * self.ctrl_scale

        # Simulate physics
        for _ in range(self.mujoco_substeps):
            mujoco.mj_step(self.model, self.data)

        # Extract state: [qpos, qvel * 0.1]
        raw_state = np.concatenate([self.data.qpos, self.data.qvel * 0.1])
        features = self.rbf.transform(raw_state.reshape(1, -1))[0]

        # Apply leaky integration for temporal memory
        self.h = (1.0 - self.leak_rate) * self.h + self.leak_rate * features

        return self.h

    def reset(self) -> None:
        """
        Reset pendulum and internal state.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.h = np.zeros_like(self.h)

    def warm_up(self, initial_input: np.ndarray) -> None:
        """
        Settle reservoir with repeated input.
        Args:
            initial_input (np.ndarray): Input signal to apply during warm-up
            steps (int): Number of warm-up steps
        """
        for _ in range(self.warmup_steps):
            self.step(input_signal=initial_input)

    def get_state(self) -> Dict[str, Any]:
        """
        Get current reservoir state.

        Returns:
            Dict containing:
                - qpos: Joint positions
                - qvel: Joint velocities
                - features: Current RBF features
                - integrated_state: Leaky-integrated state
        """
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "features": self.rbf.transform(
                np.concatenate([self.data.qpos, self.data.qvel * 0.1]).reshape(1, -1)
            )[0],
            "integrated_state": self.h.copy(),
            "ctrl": self.data.ctrl.copy(),
        }

    def get_feature_dim(self) -> int:
        """
        Get RBF feature dimension.

        Returns:
            int: Number of RBF components
        """
        return self.rbf_components
