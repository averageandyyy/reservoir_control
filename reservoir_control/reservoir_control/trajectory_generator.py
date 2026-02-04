import time

import matplotlib.pyplot as plt
import numpy as np


class TrajectoryGenerator:
    """
    2D Minimum Acceleration Trajectory Generator

    Given a set of waypoints, this class generates a queryable minimum acceleration trajectory. Given N points, it generates N-1 segments of 3rd order polynomials (cubic splines) that minimize acceleration.
    """

    def __init__(self, average_speed=1.0):
        self.average_speed = average_speed
        # Coefficients for each segment
        self.coefficients_x = []
        self.coefficients_y = []
        # Duration of each segment
        self.segment_times = []
        # Start time of each segment
        self.accumulated_times = []

    def _calculate_time_segments(self, waypoints: list[tuple[float, float]]) -> None:
        """
        Calculate time durations for each segment based on average speed and distance.
        """
        self.segment_times = []
        self.accumulated_times = [0.0]

        for i in range(len(waypoints) - 1):
            dist = np.linalg.norm(np.array(waypoints[i + 1]) - np.array(waypoints[i]))

            # Avoid division by zero
            dt = dist / self.average_speed if dist > 1e-6 else 0.1

            self.segment_times.append(dt)
            self.accumulated_times.append(self.accumulated_times[-1] + dt)

    def generate_trajectory(self, waypoints: list[tuple[float, float]]) -> float:
        """
        Generate the trajectory given a list of waypoints.
        Each segment is represented by cubic polynomials for x and y coordinates.
        """
        self._calculate_time_segments(waypoints)
        x_points = [p[0] for p in waypoints]
        y_points = [p[1] for p in waypoints]

        self.coefficients_x = self._solve_cubic_spline(x_points)
        self.coefficients_y = self._solve_cubic_spline(y_points)

        return self.accumulated_times[-1]

    def _solve_cubic_spline(self, points: list[float]) -> np.ndarray:
        """
        With N waypoints, there are N-1 segments. Each segment i has coeffcients a_i, b_i, c_i, d_i such that pos(t) = a_i + b_i*t + c_i*t^2 + d_i*t^3 for t in [0, dt_i].

        There will be a total of 4*(N-1) coefficients to solve for.
        """
        num_segments = len(points) - 1
        num_unknowns = 4 * num_segments

        # Ax = b
        A = np.zeros((num_unknowns, num_unknowns))
        b = np.zeros(num_unknowns)

        row = 0

        for i in range(num_segments):
            dt = self.segment_times[i]

            # Position constraint at start of segment, a_i = points[i]
            A[row, 4 * i] = 1.0
            b[row] = points[i]
            row += 1

            # Position constraint at end of segment, a_i + b_i*dt + c_i*dt^2 + d_i*dt^3 = points[i+1]
            A[row, 4 * i] = 1.0
            A[row, 4 * i + 1] = dt
            A[row, 4 * i + 2] = dt**2
            A[row, 4 * i + 3] = dt**3
            b[row] = points[i + 1]
            row += 1

            # Continuity constraints
            if i < num_segments - 1:
                # Velocity continuity: b_i + 2*c_i*dt + 3*d_i*dt^2 - b_{i+1} = 0
                A[row, 4 * i + 1] = 1.0
                A[row, 4 * i + 2] = 2 * dt
                A[row, 4 * i + 3] = 3 * dt**2
                A[row, 4 * (i + 1) + 1] = -1.0
                row += 1

                # Acceleration continuity: 2*c_i + 6*d_i*dt - 2*c_{i+1} = 0
                A[row, 4 * i + 2] = 2.0
                A[row, 4 * i + 3] = 6 * dt
                A[row, 4 * (i + 1) + 2] = -2.0
                row += 1

        # Boundary constraints: zero initial and final velocity and acceleration
        # Initial velocity: b_0 = 0
        A[row, 1] = 1.0
        row += 1

        # End velocity
        last_idx = num_segments - 1
        last_dt = self.segment_times[last_idx]
        A[row, 4 * last_idx + 1] = 1.0
        A[row, 4 * last_idx + 2] = 2 * last_dt
        A[row, 4 * last_idx + 3] = 3 * last_dt**2

        coefficients = np.linalg.solve(A, b)
        return coefficients.reshape((num_segments, 4))

    def query_trajectory(self, t: float) -> tuple[float, float, float]:
        """
        Query the trajectory at time t, returning (x, y, progress).
        Returns the (x, y) position at time t along the trajectory.
        progress: fraction of total trajectory completed (0 to 1).
        """
        if t <= 0:
            return (self.coefficients_x[0][0], self.coefficients_y[0][0], 0.0)

        # We want time local to a particular segment
        if t >= self.accumulated_times[-1]:
            seg_idx = len(self.segment_times) - 1
            local_t = self.segment_times[seg_idx]
        else:
            seg_idx = np.searchsorted(self.accumulated_times, t) - 1
            local_t = t - self.accumulated_times[seg_idx]

        coeffs_x = self.coefficients_x[seg_idx]
        coeffs_y = self.coefficients_y[seg_idx]

        x = (
            coeffs_x[0]
            + coeffs_x[1] * local_t
            + coeffs_x[2] * local_t**2
            + coeffs_x[3] * local_t**3
        )
        y = (
            coeffs_y[0]
            + coeffs_y[1] * local_t
            + coeffs_y[2] * local_t**2
            + coeffs_y[3] * local_t**3
        )

        return (
            x,
            y,
            t / (self.accumulated_times[-1]),
        )


def main():
    waypoints = [(0, 0), (1, 2), (4, 3), (7, 0)]
    traj_gen = TrajectoryGenerator(average_speed=1.0)
    traj_gen.generate_trajectory(waypoints)

    total_time = traj_gen.accumulated_times[-1]
    query_times = np.linspace(0, total_time, num=100)

    trajectory_points = [traj_gen.query_trajectory(t) for t in query_times]

    trajectory_points = np.array(trajectory_points)
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], label="Trajectory")
    waypoints_np = np.array(waypoints)
    plt.plot(waypoints_np[:, 0], waypoints_np[:, 1], "ro", label="Waypoints")
    plt.legend()
    plt.title("2D Minimum Acceleration Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
