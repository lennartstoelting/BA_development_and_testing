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


with open(os.path.join(project_root, "out", "filter_funcs_IMU_only.pkl"), "rb") as f:
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
        accel_in_m_ss = g2ms2(accel_in_g)

        gyro_in_deg_s = np.array([data_row[f"gyro {i} [deg/s]"] for i in ["x", "y", "z"]])
        gyro_in_rad_s = np.deg2rad(gyro_in_deg_s)
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

            if self._IMU_seems_static(accel_in_m_ss, gyro_in_rad_s):

                print("\n=== IMU seems static, triggering correction step")
                print(f"accel (m/s^2): {accel_in_m_ss}; gyro (rad/s): {gyro_in_rad_s}")
                print(f"det of P before correction step {np.linalg.det(self.P)}\n")
                with np.printoptions(precision=3, suppress=True):
                    print("P before:\n", np.array2string(self.P, precision=3))
                self._correction_step(accel_in_m_ss)
                print(f"\ndet of P after correction step {np.linalg.det(self.P)}")
                with np.printoptions(precision=3, suppress=True):
                    print("P after:\n", np.array2string(self.P, precision=3))

        accel_norm = np.absolute(
            np.linalg.norm(accel_in_m_ss - self.x[get_def("IMU accel bias")["slice"]])
            - gravity_constant
        )
        gyro_norm = np.linalg.norm(
            gyro_in_rad_s - self.x[get_def("IMU accel bias")["slice"]]
        )

        # very high values don't interest us, only the lower values are useful to determine when to cut off
        accel_norm = accel_norm if accel_norm < 1.5 else 1.5
        accel_norm = accel_norm if accel_norm < 1 else 1

        accel_threshold = 0.2
        gyro_threshold = 0.1

        # append to history for plotting
        self.input_and_state_history.append(
            np.hstack(
                [
                    self.x.copy(),
                    accel_in_m_ss,
                    gyro_in_rad_s,
                    accel_norm,
                    accel_threshold,
                    gyro_norm,
                    gyro_threshold,
                    np.linalg.slogdet(self.P)[1],
                    self.elapsed_time,
                ]
            )
        )

        self.previous_time = self.current_time

    def _calibration_step(self, accel_in_g, gyro_in_rad_s):
        self.accel_calibration_buffer.append(accel_in_g)
        self.gyro_calibration_buffer.append(gyro_in_rad_s)

        # placeholder array full of zeros except for where the quaternion need a one in the scalar
        self.x = np.eye(get_def("IMU nominal state")["len"])[6]

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

        self.x = np.zeros(get_def("IMU nominal state")["len"])
        # position and velocity start out at zero
        self.x[self.q_slice] = q_init
        self.x[self.a_b_slice] = a_b_init
        self.x[self.omega_b_slice] = omega_b_init

    def _initialize_error_state_covariance_matrix_P(self):
        # for now, these values are trial and error
        # (low number -> low variance -> high certainty)
        initial_certainties = {
            "position": 1e-3,  # guess
            "velocity": 1e-3,  # guess
            "orientation": 5e-3,  # guess
            "accel_bias": 0.06,  # guess
            "gyro_bias": 2e-3,  # guess
        }

        self.P = np.diag(
            np.repeat([sigma_sqrd for _, sigma_sqrd in initial_certainties.items()], 3)
        )

    def _initialize_process_noise_covariance_matrix_Q(self):
        accelNSD_mug = 230  # mug/√Hz
        gyroNSD_deg = 0.015  # d/s/√Hz

        # unit conversion
        accelNSD = g2ms2(accelNSD_mug * 1e-6)  # m/s2/√Hz
        gyroNSD = np.deg2rad(gyroNSD_deg)  # rad/s/√Hz

        # multiplied by bandwidth to get standard deviation
        sigma_accel = accelNSD * np.sqrt(IMU_sample_freq)
        sigma_gyro = gyroNSD * np.sqrt(IMU_sample_freq)

        # square sigma for variance and multiply by dt^2
        variance_accel = (sigma_accel**2) * ((1 / IMU_sample_freq) ** 2)
        variance_gyro = (sigma_gyro**2) * ((1 / IMU_sample_freq) ** 2)

        # vorher hatte ich 1e-5 überall, halt komplett ins blaue geschätzt
        process_noise = {
            "velocity": variance_accel,
            "orientation": variance_gyro,
            "accel_bias": 9e-9,  # guess
            "gyro_bias": 4e-12,  # guess
        }

        self.Q = np.diag(
            np.repeat([sigma_sqrd for _, sigma_sqrd in process_noise.items()], 3)
        )

    def _initialize_measurement_noise_covariance_matrix_W(self):
        measurement_noise = {
            "accel_gravity": 2.5e-3,  # guess
        }

        self.W = np.diag(
            np.repeat([sigma_sqrd for _, sigma_sqrd in measurement_noise.items()], 3)
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

    def _IMU_seems_static(self, accel_in_m_ss, gyro_in_rad_s) -> bool:
        """using the data from accelerometer and gyro, are the values in range of a certain threshold
        which lets us assume the IMU measures no additional acceleration or angle velocity
        """
        static_accel_threshold = 0.2
        accel_absolute = np.absolute(
            np.linalg.norm(accel_in_m_ss - self.x[get_def("IMU accel bias")["slice"]])
            - gravity_constant
        )
        accel_seems_static = accel_absolute < static_accel_threshold

        static_gyro_threshold = 0.1
        gyro_absolute = np.linalg.norm(
            gyro_in_rad_s - self.x[get_def("IMU accel bias")["slice"]]
        )
        gyro_seems_static = gyro_absolute < static_gyro_threshold

        return accel_seems_static and gyro_seems_static

    def _correction_step(self, a_tilde):
        H = esekf_funcs["H_delta_x"](*self.x[6:10], gravity_constant)
        h = esekf_funcs["h"](
            *self.x[self.q_slice], *self.x[self.a_b_slice], gravity_constant
        ).flatten()

        # fmt: off
        print(f"\n{np.array2string(h, precision=3)} <- prediction of accelerometer measurement")
        print(f"{np.array2string(a_tilde, precision = 3)} <- accelerometer measurement\n")
        # fmt: on

        # Eigentlich könnte ich das was h macht ja hier händisch rechnen

        h_test = (
            R(self.x[self.q_slice]).T @ [0, 0, gravity_constant]
            + self.x[self.a_b_slice]
        )
        print(np.array2string(h_test, precision=3))

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
        # lesbarer print von delta_x
        [
            print(
                f"{comp['name']}:\n{np.array2string(delta_x[comp['slice']], precision=3)}"
            )
            for comp in get_def("IMU error state")["components"]
        ]
        self.x = esekf_funcs["x_IMU_true"](*self.x, *delta_x).flatten()
        self.x[self.q_slice] = normalize_vector(self.x[self.q_slice])
        self.P = (np.eye(15) - K @ H) @ np.linalg.inv(self.P)
