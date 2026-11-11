import numpy as np
import pandas as pd
import dill
import os
import sys

# go to parent directory
project_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules import *


# Get the directory of the current file, then go up one level to 'IMU_only'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pkl_path = os.path.join(project_root, "out", "filter_funcs_IMU_only.pkl")

with open(pkl_path, "rb") as f:
    esekf_funcs = dill.load(f)


class Filter:
    def __init__(self):

        self.start_time = None
        self.previous_time = None
        self.current_time = None
        self.elapsed_time = None

        self.x = None
        self.P = None
        self.Q = None
        self.J = None
        self.W = None

        self.p_slice = get_def("IMU position")["slice"]
        self.v_slice = get_def("IMU velocity")["slice"]
        self.q_slice = get_def("IMU orientation")["slice"]
        self.a_b_slice = get_def("IMU accel bias")["slice"]
        self.omega_b_slice = get_def("IMU gyro bias")["slice"]

        # Since I will not have it always be the case that there is zero speakers and the only measurement is gravity,
        # I need to construct the matrices individually but also ideally automatically
        self._initialize_error_state_covariance_matrix_P()
        self._initialize_process_noise_covariance_matrix_Q()
        self._initialize_process_noise_jacobian_matrix_J()
        self._initialize_measurement_noise_covariance_matrix_W()

        self.calibration_complete = False

        self.accel_calibration_buffer = []
        self.gyro_calibration_buffer = []

        self.state_history = []
        self.input_and_state_history = []

    def process_IMU_sample(self, data_row):
        # prepare all inputs
        timestamp = data_row["timestamp [ns]"] / 1e9

        # fmt: off
        accel_in_g = np.array([data_row[f"acceleration {i} [g]"] for i in ["x", "y", "z"]])
        accel_in_m_ss = accel_in_g * gravity_constant

        gyro_in_deg_s = np.array([data_row[f"gyro {i} [deg/s]"] for i in ["x", "y", "z"]])
        gyro_in_rad_s = gyro_in_deg_s * (np.pi / 180.0)
        # fmt: on

        if not self.start_time:
            """only for very first sample"""
            self.start_time = timestamp

        self.current_time = timestamp
        self.elapsed_time = self.current_time - self.start_time

        if not self.calibration_complete:
            self._calibration_step(accel_in_g, gyro_in_rad_s)
        else:
            dt = self.current_time - self.previous_time
            self._prediction_step(accel_in_m_ss, gyro_in_rad_s, dt)
            # prediction step attempts
            # static_accel_threshold = 0.1
            # static_gyro_threshold = 0.1
            # accel_seems_static = (
            #     np.absolute(
            #         np.linalg.norm(
            #             accel_in_m_ss - esekf.x[get_def("IMU accel bias")["slice"]]
            #         )
            #         - gravity_constant
            #     )
            #     < static_accel_threshold
            # )
            # gyro_seems_static = (
            #     np.linalg.norm(gyro_in_rad_s - esekf.x[get_def("IMU accel bias")["slice"]])
            #     < static_gyro_threshold
            # )
            # if accel_seems_static and gyro_seems_static:
            #     # print(
            #     #     f"========================\ncorrection step now! (index {index}):\naccel: {accel_in_g}\ngyro: {gyro_in_deg_s}\n========================\n"
            #     # )
            # self._correction_step(accel_in_m_ss, gravity_constant)
            # else:
            #     corr_step_triggered += 1
            #     # print(
            #     #     f"measurements don't seem static (index: {index}):\naccel {accel_in_g}\ngyro {gyro_in_deg_s}\n"
            #     # )

        # append to history for plotting
        self.input_and_state_history.append(
            np.hstack(
                [
                    self.x.copy() if self.x is not None else None,
                    accel_in_m_ss,
                    gyro_in_rad_s,
                    self.elapsed_time,
                ]
            )
        )

        self.previous_time = self.current_time

    def _calibration_step(self, accel_in_g, gyro_in_rad_s):
        self.accel_calibration_buffer.append(accel_in_g)
        self.gyro_calibration_buffer.append(gyro_in_rad_s)

        if self.elapsed_time < calibration_dt:
            # keep calibarting if we are not over the calibration time window
            return

        self._initialize_state_x()

        self.calibration_complete = True
        """
        print(
            f"\nCalibration ended after {self.elapsed_time:.6f} seconds | {len(self.accel_calibration_buffer)} samples used | Initial state after calibration:\n"
        )
        for i in self.x:
            print(i)
        """
        print(f"determinant of covariance P at the start: {np.linalg.det(self.P)}")

        self.accel_calibration_buffer = []
        self.gyro_calibration_buffer = []

    def _initialize_state_x(self):
        accel_mean = np.mean(np.array(self.accel_calibration_buffer), axis=0)
        accel_mean_normalized = normalize_vector(accel_mean)
        gyro_mean = np.mean(np.array(self.gyro_calibration_buffer), axis=0)

        # orientation
        gravity_in_g = np.array([0.0, 0.0, 1.0])
        qw_init = 1.0 + np.dot(accel_mean_normalized, gravity_in_g)
        qx_init, qy_init, qz_init = np.cross(accel_mean_normalized, gravity_in_g)

        q_init = normalize_vector([qw_init, qx_init, qy_init, qz_init])

        # accel bias
        # subtract what pure gravity should look like (1g) to be left with bias
        accel_mean_without_gravity = accel_mean - accel_mean_normalized
        # g to m/s^2 conversion
        a_b_init = accel_mean_without_gravity * gravity_constant

        # gyro bias
        omega_b_init = gyro_mean

        self.x = np.zeros(get_def("IMU nominal state")["length"])
        # position and velocity start out at zero
        self.x[self.q_slice] = q_init
        self.x[self.a_b_slice] = a_b_init
        self.x[self.omega_b_slice] = omega_b_init

    def _initialize_error_state_covariance_matrix_P(self):
        # for now, these values are trial and error
        # (low number -> low variance -> high certainty)
        initial_certainties = {
            "position": 1e-5,
            "velocity": 1e-5,
            "orientation": 1e-5,
            "accel_bias": 1e-5,
            "gyro_bias": 1e-5,
        }

        state_order = ["position", "velocity", "orientation", "accel_bias", "gyro_bias"]
        self.P = np.diag(
            np.repeat([initial_certainties[key] for key in state_order], 3)
        )

    def _initialize_process_noise_covariance_matrix_Q(self):
        process_noise = {
            "velocity": 1e-5,
            "orientation": 1e-5,
            "accel_bias": 1e-5,
            "gyro_bias": 1e-5,
        }

        process_noise_order = ["velocity", "orientation", "accel_bias", "gyro_bias"]
        self.Q = np.diag(
            np.repeat([process_noise[key] for key in process_noise_order], 3)
        )

    def _initialize_measurement_noise_covariance_matrix_W(self):
        measurement_noise = {"accel_gravity": 1e-5}

        measurement_noise_order = ["accel_gravity"]

        self.W = np.diag(
            np.repeat([measurement_noise[key] for key in measurement_noise_order], 3)
        )

    def _initialize_process_noise_jacobian_matrix_J(self):
        self.J = np.vstack((np.zeros((3, 12)), np.eye(12)))

    def _prediction_step(self, a_tilde, omega_tilde, dt):
        self.x = esekf_funcs["f_x_IMU"](
            *self.x, *a_tilde, *omega_tilde, gravity_constant, dt
        ).flatten()
        self.x[self.q_slice] = normalize_vector(self.x[self.q_slice])

        # hier ist die Frage ob ich das vorher mache oder nachher? Also
        F_IMU = esekf_funcs["F_delta_x"](
            *self.x[self.q_slice],
            *self.x[self.a_b_slice],
            *self.x[self.omega_b_slice],
            *a_tilde,
            *omega_tilde,
            dt,
        )

        self.P = F_IMU @ self.P @ F_IMU.T + self.J @ self.Q @ self.J.T

    def _correction_step(self, a_tilde):
        H = esekf_funcs["H_delta_x"](*self.x[6:10], gravity_constant)
        h = esekf_funcs["h"](
            *self.x[self.q_slice], *self.x[self.a_b_slice], gravity_constant
        ).flatten()

        try:
            K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.W)
        except np.linalg.LinAlgError:
            print(f"x.shape: {self.x.shape}")
            print(f"P.shape: {self.P.shape}")
            print(f"H.shape: {H.shape}")
            print(f"W.shape: {self.W.shape}")
            S = H @ self.P @ H.T + self.W
            print("Determinant of P:", np.linalg.det(self.P))
            print("Determinant of S:", np.linalg.det(S))
            print(f"P matrix: {self.P}")
        delta_x = K @ (a_tilde - h)
        self.x = esekf_funcs["x_IMU_true"](*self.x, *delta_x).flatten()
        self.x[self.q_slice] = normalize_vector(self.x[self.q_slice])
        self.P = (np.eye(15) - K @ H) @ np.linalg.inv(self.P)
