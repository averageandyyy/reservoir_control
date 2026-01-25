"""
Minimal Reservoir Computing Controller
=======================================
Goal: Establish if a MuJoCo double-pendulum reservoir can learn trajectory tracking
      without any skip connections or complex features.

Architecture (NO SKIP CONNECTION):
"""

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# Configuration (minimal set)
# ==========================================
CONFIG = {
    "dt_ctrl": 0.05,
    "mujoco_timesteps": 5,
    # Reservoir
    "rbf_components": 1000,
    "rbf_gamma": 0.05,  # Broader features for generalization
    "ctrl_scale": 12.0,
    "leak_rate": 0.3,  # Faster responsiveness
    # Input scaling
    "error_scale": 1.0,
    "feedback_gain": 0.0,  # NO FEEDBACK (like working version)
    "training_noise": 0.05,
    "training_action_noise": 0.15,  # Inject noise to create off-track data
    # Expert policy
    "v_max": 2.0,
    "w_max": 4.0,
    "k_v": 2.0,
    "k_w": 8.0,
    # Training
    "train_episodes": 100,
    "steps_per_episode": 400,
    "warmup_steps": 30,  # Settle reservoir before episode starts
    "washout": 5,  # Reduced from 50 (warm-up handles settling)
    "init_pos_noise": 0.5,  # Robust recovery training
    "init_theta_noise": 0.5,
    # Ridge regularization
    "alpha": 0.01,  # Higher-fidelity fit
    # Evaluation
    "eval_starts": 5,
    "eval_steps": 503,
    "seed": 42,
    # Recording (set to None to disable, or e.g. "reservoir_tracking.mp4" or "reservoir_tracking.gif")
    "save_animation": "reservoir_tracking.gif",
}


# ==========================================
# MuJoCo XML
# ==========================================
def make_xml(timestep):
    return f"""
<mujoco model="double_pendulum">
  <option timestep="{timestep}" integrator="RK4" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 2">
      <joint name="pin1" type="hinge" axis="0 1 0" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05" rgba="0.8 0.2 0.2 1" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="pin2" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05" rgba="0.2 0.8 0.2 1" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="pin1" name="torque1" gear="1"/>
    <motor joint="pin2" name="torque2" gear="1"/>
  </actuator>
</mujoco>
"""


# ==========================================
# Reservoir
# ==========================================
class Reservoir:
    def __init__(
        self,
        xml_string,
        dt_ctrl,
        mujoco_substeps,
        rbf_gamma,
        rbf_components,
        ctrl_scale,
        leak_rate,
        seed,
    ):
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self.mujoco_substeps = mujoco_substeps
        self.ctrl_scale = ctrl_scale
        self.leak_rate = leak_rate

        # RBF feature map with leaky integration
        self.rbf = RBFSampler(
            gamma=rbf_gamma, n_components=rbf_components, random_state=seed
        )
        self.rbf.fit(np.zeros((1, 4)))  # 4 = qpos(2) + qvel(2)
        self.h = np.zeros(rbf_components)  # Internal state for leaky integration

    def step(self, input_signal):
        """Drive pendulum with scaled input, return leaky-integrated RBF features."""
        # Note: input_signal may be 3D or 5D depending on feedback
        self.data.ctrl[0] = input_signal[0] * self.ctrl_scale
        self.data.ctrl[1] = input_signal[1] * self.ctrl_scale

        for _ in range(self.mujoco_substeps):
            mujoco.mj_step(self.model, self.data)

        # Raw state: [qpos, qvel * 0.1]
        raw = np.concatenate([self.data.qpos, self.data.qvel * 0.1])
        features = self.rbf.transform(raw.reshape(1, -1))[0]

        # Apply leaky integration for temporal memory
        self.h = (1.0 - self.leak_rate) * self.h + self.leak_rate * features
        return self.h

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.h = np.zeros_like(self.h)  # Reset internal state

    def warm_up(self, initial_input, steps):
        """Pump reservoir with static input to settle from zero state."""
        for _ in range(steps):
            self.step(initial_input)


# ==========================================
# Robot
# ==========================================
class DiffDriveRobot:
    def __init__(self, dt):
        self.dt = dt
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def step(self, v, w):
        self.x += v * np.cos(self.theta) * self.dt
        self.y += v * np.sin(self.theta) * self.dt
        self.theta += w * self.dt

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0


# ==========================================
# Utilities
# ==========================================
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


def expert_policy(robot_state, target, cfg):
    """Simple P-control expert (like working version) - NO stop-and-turn, NO look-ahead."""
    local_error = get_local_error(robot_state, target)
    dist, heading_error = local_error[0], local_error[1]

    # Pure proportional control
    v_expert = np.clip(cfg["k_v"] * dist, 0, cfg["v_max"])
    w_expert = np.clip(cfg["k_w"] * heading_error, -cfg["w_max"], cfg["w_max"])

    return np.array([v_expert, w_expert])


def scale_input(local_error, cfg):
    """Raw [dist, heading_error] clipped to [-1, 1] for reservoir drive (like working version)."""
    # Just clip to [-1, 1] range - let the reservoir handle the nonlinearity
    return np.clip(local_error, -1.0, 1.0)


def reference_heading(ref_x, ref_y, idx):
    """Compute periodic tangent heading."""
    n = len(ref_x)
    i = int(idx) % n
    i_prev = (i - 1) % n
    i_next = (i + 1) % n
    dx = ref_x[i_next] - ref_x[i_prev]
    dy = ref_y[i_next] - ref_y[i_prev]
    return np.arctan2(dy, dx)


# ==========================================
# Chirp Signal
# ==========================================
def chirp_signal(t, amplitude=0.4, frequencies=[0.1, 0.3, 0.7, 1.5], seed=42):
    np.random.seed(seed)
    phases = np.random.uniform(0, 2 * np.pi, len(frequencies))
    signal = sum(np.sin(2 * np.pi * f * t + p) for f, p in zip(frequencies, phases))
    return amplitude * signal / len(frequencies)


# ==========================================
# Visualization
# ==========================================
def visualize_rollout(
    reservoir,
    robot,
    scaler,
    readout,
    ref_x,
    ref_y,
    start_idx,
    steps,
    cfg,
    save_path=None,
):
    """Real-time animation (Learned vs Expert).

    Args:
        save_path: If provided, saves animation to file (supports .mp4 or .gif)
    """
    # 1. Expert Rollout (no reservoir warm-up needed, just robot)
    reservoir.reset()
    robot.reset()
    robot.x, robot.y = ref_x[start_idx], ref_y[start_idx]
    robot.theta = reference_heading(ref_x, ref_y, start_idx)

    expert_traj = {"x": [], "y": []}
    for t_step in range(steps):
        t_idx = (start_idx + t_step) % len(ref_x)
        target = (ref_x[t_idx], ref_y[t_idx])
        act = expert_policy((robot.x, robot.y, robot.theta), target, cfg)
        expert_traj["x"].append(robot.x)
        expert_traj["y"].append(robot.y)
        robot.step(act[0], act[1])

    # 2. Learned Rollout
    reservoir.reset()
    robot.reset()
    robot.x, robot.y = ref_x[start_idx], ref_y[start_idx]
    robot.theta = reference_heading(ref_x, ref_y, start_idx)

    # Warm-up: settle reservoir with initial error
    local_error_0 = get_local_error(
        (robot.x, robot.y, robot.theta), (ref_x[start_idx], ref_y[start_idx])
    )
    res_input_0 = scale_input(local_error_0, cfg)
    reservoir.warm_up(res_input_0, cfg["warmup_steps"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_xlim(-0.6, 0.6)
    ax1.set_ylim(0.9, 2.6)
    ax1.set_aspect("equal")
    ax1.set_title("Double Pendulum Reservoir")
    ax1.grid(True, alpha=0.3)
    (line1,) = ax1.plot([], [], "o-", lw=4, color="red")
    (line2,) = ax1.plot([], [], "o-", lw=4, color="green")

    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_aspect("equal")
    ax2.set_title("Robot Tracking: Learned (Blue) vs Expert (Red)")
    ax2.plot(ref_x, ref_y, "k--", lw=2, alpha=0.2, label="Reference")
    ax2.plot(
        expert_traj["x"], expert_traj["y"], "r:", lw=2, alpha=0.4, label="Expert Path"
    )
    (robot_trail,) = ax2.plot([], [], "b-", lw=2, alpha=0.6, label="Learned")
    robot_marker = Circle((robot.x, robot.y), 0.1, color="blue", zorder=5)
    ax2.add_patch(robot_marker)
    # Reference target marker (the "rabbit" the robot is chasing)
    ref_marker = Circle(
        (ref_x[start_idx], ref_y[start_idx]),
        0.15,
        color="green",
        alpha=0.7,
        zorder=4,
        label="Target",
    )
    ax2.add_patch(ref_marker)
    arrow = ax2.arrow(
        0, 0, 0, 0, head_width=0.15, head_length=0.15, fc="blue", ec="blue", zorder=6
    )
    error_text = ax2.text(
        0.02,
        0.98,
        "",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    trajectory = {"x": [], "y": []}
    arrow_artist = {"obj": arrow}

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        robot_trail.set_data([], [])
        return line1, line2, robot_trail, robot_marker, ref_marker, arrow

    def update(frame):
        if frame >= steps:
            return line1, line2, robot_trail, robot_marker, ref_marker, arrow

        t_idx = (start_idx + frame) % len(ref_x)
        target = (ref_x[t_idx], ref_y[t_idx])
        state = (robot.x, robot.y, robot.theta)
        local_error = get_local_error(state, target)
        dist = local_error[0]

        res_input = scale_input(local_error, cfg)
        res_features = reservoir.step(res_input)

        # NO feedback - just reservoir features
        action = readout.predict(scaler.transform(res_features.reshape(1, -1)))[0]
        action[0] = np.clip(action[0], 0.0, cfg["v_max"])
        action[1] = np.clip(action[1], -cfg["w_max"], cfg["w_max"])

        robot.step(action[0], action[1])

        q1, q2 = reservoir.data.qpos[0], reservoir.data.qpos[1]
        x1, y1 = 0.5 * np.sin(q1), 2.0 - 0.5 * np.cos(q1)
        x2, y2 = x1 + 0.5 * np.sin(q1 + q2), y1 - 0.5 * np.cos(q1 + q2)
        line1.set_data([0, x1], [2.0, y1])
        line2.set_data([x1, x2], [y1, y2])

        trajectory["x"].append(robot.x)
        trajectory["y"].append(robot.y)
        robot_trail.set_data(trajectory["x"], trajectory["y"])
        robot_marker.center = (robot.x, robot.y)
        # Update reference target marker (the "rabbit")
        ref_marker.center = (ref_x[t_idx], ref_y[t_idx])
        arrow_artist["obj"].remove()
        arrow_artist["obj"] = ax2.arrow(
            robot.x,
            robot.y,
            0.3 * np.cos(robot.theta),
            0.3 * np.sin(robot.theta),
            head_width=0.15,
            head_length=0.15,
            fc="blue",
            ec="blue",
            zorder=6,
        )
        error_text.set_text(
            f"Step: {frame}\nError: {dist:.3f}\nv: {action[0]:.2f}\nω: {action[1]:.2f}"
        )
        return line1, line2, robot_trail, robot_marker, ref_marker, arrow_artist["obj"]

    anim = FuncAnimation(
        fig, update, init_func=init, frames=steps, interval=50, blit=False, repeat=False
    )
    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=20)
        else:
            anim.save(save_path, writer="ffmpeg", fps=20)
        print(f"Animation saved!")

    plt.show()
    return trajectory


# ==========================================
# Main
# ==========================================
def main():
    cfg = CONFIG
    np.random.seed(cfg["seed"])

    timestep = cfg["dt_ctrl"] / cfg["mujoco_timesteps"]
    reservoir = Reservoir(
        make_xml(timestep),
        cfg["dt_ctrl"],
        cfg["mujoco_timesteps"],
        cfg["rbf_gamma"],
        cfg["rbf_components"],
        cfg["ctrl_scale"],
        cfg["leak_rate"],
        cfg["seed"],
    )
    robot = DiffDriveRobot(cfg["dt_ctrl"])

    n_ref = 800
    t_ref = np.linspace(0, 2 * np.pi, n_ref)
    A = 4.0
    denom = 1 + np.sin(t_ref) ** 2
    ref_x, ref_y = A * np.cos(t_ref) / denom, A * np.sin(t_ref) * np.cos(t_ref) / denom

    print("Collecting training data...")
    X_train_data, Y_train_data = [], []
    for ep in range(cfg["train_episodes"]):
        reservoir.reset()
        robot.reset()
        start_idx = np.random.randint(0, n_ref)
        robot.x = ref_x[start_idx] + np.random.normal(0, cfg["init_pos_noise"])
        robot.y = ref_y[start_idx] + np.random.normal(0, cfg["init_pos_noise"])
        robot.theta = reference_heading(ref_x, ref_y, start_idx) + np.random.normal(
            0, cfg["init_theta_noise"]
        )

        # Warm-up: settle reservoir with initial error before episode starts
        local_error_0 = get_local_error(
            (robot.x, robot.y, robot.theta), (ref_x[start_idx], ref_y[start_idx])
        )
        res_input_0 = scale_input(local_error_0, cfg)
        reservoir.warm_up(res_input_0, cfg["warmup_steps"])

        for i in range(cfg["steps_per_episode"]):
            t_idx = (start_idx + i) % n_ref
            state = (robot.x, robot.y, robot.theta)
            target = (ref_x[t_idx], ref_y[t_idx])
            local_error = get_local_error(state, target)
            res_input = scale_input(local_error, cfg) + np.random.normal(
                0, cfg["training_noise"], size=2
            )
            res_features = reservoir.step(res_input)
            action = expert_policy(state, target, cfg)
            if i >= cfg["washout"]:
                # NO feedback - just reservoir features
                X_train_data.append(res_features)
                Y_train_data.append(action)
            noisy_act = action + np.random.normal(
                0, cfg["training_action_noise"], size=2
            )
            robot.step(noisy_act[0], noisy_act[1])

    X_all, Y_all = np.array(X_train_data), np.array(Y_train_data)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=cfg["seed"]
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    readout = Ridge(alpha=cfg["alpha"])
    readout.fit(X_train_scaled, Y_train)
    print(
        f"Training R^2: {readout.score(X_train_scaled, Y_train):.4f}, Validation R^2: {readout.score(X_val_scaled, Y_val):.4f}"
    )

    print("\nEvaluating learned controller...")
    eval_indices = np.linspace(0, n_ref, cfg["eval_starts"], endpoint=False, dtype=int)
    all_errors, all_diverge = [], []
    for start_idx in eval_indices:
        reservoir.reset()
        robot.reset()
        robot.x, robot.y = ref_x[start_idx], ref_y[start_idx]
        robot.theta = reference_heading(ref_x, ref_y, start_idx)

        # Warm-up: settle reservoir with initial error
        local_error_0 = get_local_error(
            (robot.x, robot.y, robot.theta), (ref_x[start_idx], ref_y[start_idx])
        )
        res_input_0 = scale_input(local_error_0, cfg)
        reservoir.warm_up(res_input_0, cfg["warmup_steps"])

        errors = []
        diverged_at = None
        for i in range(cfg["eval_steps"]):
            t_idx = (start_idx + i) % n_ref
            state = (robot.x, robot.y, robot.theta)
            target = (ref_x[t_idx], ref_y[t_idx])
            local_error = get_local_error(state, target)
            dist = local_error[0]
            errors.append(dist)
            if diverged_at is None and dist > 3.0:
                diverged_at = i
            res_input = scale_input(local_error, cfg)
            res_features = reservoir.step(res_input)
            # NO feedback - just reservoir features
            action = readout.predict(scaler.transform(res_features.reshape(1, -1)))[0]
            action[0] = np.clip(action[0], 0.0, cfg["v_max"])
            action[1] = np.clip(action[1], -cfg["w_max"], cfg["w_max"])
            robot.step(action[0], action[1])
        all_errors.append(np.mean(errors))
        all_diverge.append(
            diverged_at if diverged_at is not None else cfg["eval_steps"]
        )

    print(
        f"\nResults:\n  Mean error: {np.mean(all_errors):.3f}\n  Mean diverge: {np.mean(all_diverge):.1f}"
    )

    # Static plot of best rollout
    best_idx = np.argmin(all_errors)
    start_idx = eval_indices[best_idx]
    reservoir.reset()
    robot.reset()
    robot.x, robot.y = ref_x[start_idx], ref_y[start_idx]
    robot.theta = reference_heading(ref_x, ref_y, start_idx)

    # Warm-up: settle reservoir with initial error
    local_error_0 = get_local_error(
        (robot.x, robot.y, robot.theta), (ref_x[start_idx], ref_y[start_idx])
    )
    res_input_0 = scale_input(local_error_0, cfg)
    reservoir.warm_up(res_input_0, cfg["warmup_steps"])

    path_x, path_y = [], []
    for i in range(cfg["eval_steps"]):
        t_idx = (start_idx + i) % n_ref
        state = (robot.x, robot.y, robot.theta)
        target = (ref_x[t_idx], ref_y[t_idx])
        local_error = get_local_error(state, target)
        res_features = reservoir.step(scale_input(local_error, cfg))
        # NO feedback - just reservoir features
        action = readout.predict(scaler.transform(res_features.reshape(1, -1)))[0]
        robot.step(
            np.clip(action[0], 0, cfg["v_max"]),
            np.clip(action[1], -cfg["w_max"], cfg["w_max"]),
        )
        path_x.append(robot.x)
        path_y.append(robot.y)

    plt.figure(figsize=(10, 8))
    plt.plot(ref_x, ref_y, "k--", alpha=0.5, label="Reference")
    plt.plot(path_x, path_y, "b-", label="Learned")
    plt.scatter([ref_x[start_idx]], [ref_y[start_idx]], c="green", s=100, label="Start")
    plt.axis("equal")
    plt.legend()
    plt.show()

    # Synchronize animation with full evaluation horizon
    print("\nStarting real-time animation...")
    visualize_rollout(
        reservoir,
        robot,
        scaler,
        readout,
        ref_x,
        ref_y,
        start_idx,
        cfg["eval_steps"],
        cfg,
        save_path=cfg["save_animation"],
    )


if __name__ == "__main__":
    main()
