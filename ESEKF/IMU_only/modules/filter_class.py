import dill
import numpy as np

from config import FILTER_FUNCS_FILE
from modules import *

with open(FILTER_FUNCS_FILE, "rb") as f:
    esekf_funcs = dill.load(f)

ACCEL_STATIC_THRESHOLD = 0.3
GYRO_STATIC_THRESHOLD = 0.1


class Filter:
    # ---------------------------------------------------------------------------
    # instance initialization
    # ---------------------------------------------------------------------------

    def __init__(self):

        self.x = None

        # self.p_slice = get_def("imu position")["slice"]  # unused
        # self.v_slice = get_def("imu velocity")["slice"]  # unused
        self.q_slice = get_def("imu orientation")["slice"]
        self.a_b_slice = get_def("imu accel bias")["slice"]
        self.omega_b_slice = get_def("imu gyro bias")["slice"]

        self.P = None
        self.Q = None
        self.J = None
        self.W = None
        self._initialize_matrices()  # temporary

        self.start_time = None
        self.previous_time = None
        self.current_time = None
        self.elapsed_time = None

        self.calibration_complete = False
        self.accel_calibration_buffer = []
        self.gyro_calibration_buffer = []

        self.filter_history = []

    # ---------------------------------------------------------------------------
    # matrices initialization (implementation temporary)
    # ---------------------------------------------------------------------------

    def _initialize_matrices(self):
        self._initialize_error_state_covariance_matrix_P()
        self._initialize_process_noise_covariance_matrix_Q()
        self._initialize_process_noise_jacobian_matrix_J()
        self._initialize_measurement_noise_covariance_matrix_W()

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
        sigma_accel = accelNSD * np.sqrt(imu_SAMPLE_FREQ)
        sigma_gyro = gyroNSD * np.sqrt(imu_SAMPLE_FREQ)

        # square sigma for variance and multiply by dt^2
        variance_accel = (sigma_accel**2) * ((1 / imu_SAMPLE_FREQ) ** 2)
        variance_gyro = (sigma_gyro**2) * ((1 / imu_SAMPLE_FREQ) ** 2)

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

    # ---------------------------------------------------------------------------
    # processing one sample at a time
    # ---------------------------------------------------------------------------

    def process_imu_sample(self, data_row, sample_index):
        # prepare all inputs
        timestamp = data_row["timestamp [ns]"] / 1e9

        neon_orientation_estimate = [
            data_row["roll [deg]"],
            data_row["pitch [deg]"],
            data_row["yaw [deg]"],
        ]

        # fmt: off
        accel_in_g = np.array([data_row[f"acceleration {i} [g]"] for i in ["x", "y", "z"]])
        accel_in_m_ss = g2ms2(accel_in_g)

        gyro_in_deg_s = np.array([data_row[f"gyro {i} [deg/s]"] for i in ["x", "y", "z"]])
        gyro_in_rad_s = np.deg2rad(gyro_in_deg_s)
        # fmt: on

        imu_static, accel_norm, gyro_norm = self._imu_seems_static(
            accel_in_m_ss, gyro_in_rad_s
        )

        if not self.start_time:
            self.start_time = timestamp  # only for very first sample

        self.current_time = timestamp
        self.elapsed_time = self.current_time - self.start_time

        if not self.calibration_complete:
            self._calibration_step(accel_in_g, gyro_in_rad_s)
        else:
            dt = self.current_time - self.previous_time

            self._prediction_step(accel_in_m_ss, gyro_in_rad_s, dt)

            if imu_static:
                # print(f"\ntriggering correction step at sample {sample_index}")
                self._correction_step(accel_in_m_ss)

        self.filter_history.append(
            np.hstack(
                [
                    # fmt: off
                    self.x.copy(),  # state
                    *[accel_in_m_ss, gyro_in_rad_s],  # raw imu sensor measurements (units converted)
                    *[accel_norm, ACCEL_STATIC_THRESHOLD],  # accel static check
                    *[gyro_norm, GYRO_STATIC_THRESHOLD],  # gyro static check
                    np.linalg.slogdet(self.P)[1],  # det P
                    neon_orientation_estimate,  # neon glasses orientation estimate
                    self.elapsed_time,
                    # fmt: on
                ]
            )
        )

        self.previous_time = self.current_time

    def _imu_seems_static(self, accel_in_m_ss, gyro_in_rad_s):
        """using the data from accelerometer and gyro, are the values in range of a certain threshold
        which lets us assume the imu measures no additional acceleration or angle velocity
        """
        accel_absolute = np.absolute(np.linalg.norm(accel_in_m_ss) - GRAVITY_CONSTANT)
        accel_seems_static = accel_absolute < ACCEL_STATIC_THRESHOLD

        gyro_absolute = np.linalg.norm(gyro_in_rad_s)
        gyro_seems_static = gyro_absolute < GYRO_STATIC_THRESHOLD

        imu_seems_static = accel_seems_static and gyro_seems_static
        # if imu_seems_static:
        #     print(f"imu seems static with these values:")
        #     print(f"accel (m/s^2): {accel_in_m_ss}; gyro (rad/s): {gyro_in_rad_s}")

        return (
            imu_seems_static,
            accel_absolute,
            gyro_absolute,
        )

    # ---------------------------------------------------------------------------
    # calibration
    # ---------------------------------------------------------------------------

    def _calibration_step(self, accel_in_g, gyro_in_rad_s):
        self.accel_calibration_buffer.append(accel_in_g)
        self.gyro_calibration_buffer.append(gyro_in_rad_s)

        # placeholder array full of zeros except for where the quaternion need a one in the scalar
        self.x = np.eye(get_def("imu nominal state")["len"])[6]

        if self.elapsed_time < CALIBRATION_DT:
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
        a_b_init = accel_mean_without_gravity * GRAVITY_CONSTANT

        # gyro bias
        omega_b_init = gyro_mean

        self.x = np.zeros(get_def("imu nominal state")["len"])
        # position and velocity start out at zero
        self.x[self.q_slice] = q_init
        self.x[self.a_b_slice] = a_b_init
        self.x[self.omega_b_slice] = omega_b_init

    # ---------------------------------------------------------------------------
    # prediction
    # ---------------------------------------------------------------------------

    def _prediction_step(self, a_tilde, omega_tilde, dt):
        self.x = esekf_funcs["f_x_imu"](
            *self.x, *a_tilde, *omega_tilde, GRAVITY_CONSTANT, dt
        ).flatten()
        self.x[self.q_slice] = normalize_vector(self.x[self.q_slice])

        # hier ist die Frage ob ich das vorher mache oder nachher? Also
        F_imu = esekf_funcs["F_delta_x"](
            *self.x[self.q_slice],
            *self.x[self.a_b_slice],
            *self.x[self.omega_b_slice],
            *a_tilde,
            *omega_tilde,
            dt,
        )

        self.P = F_imu @ self.P @ F_imu.T + self.J @ self.Q @ self.J.T

    # ---------------------------------------------------------------------------
    # correction
    # ---------------------------------------------------------------------------

    def _correction_step(self, a_tilde):
        # print(f"det of P before correction step {np.linalg.det(self.P)}")

        H = esekf_funcs["H_delta_x"](*self.x[6:10], GRAVITY_CONSTANT)
        h = esekf_funcs["h"](
            *self.x[self.q_slice], *self.x[self.a_b_slice], GRAVITY_CONSTANT
        ).flatten()

        # fmt: off
        # print(f"\norientation quaternion: {np.array2string(self.x[self.q_slice], precision=3)}")
        # print(f"{np.array2string(h, precision=3)} <- my prediction of accelerometer measurement")
        # print(f"{np.array2string(a_tilde, precision = 3)} <- accelerometer measurement")
        # fmt: on

        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.W)
        delta_x = K @ (a_tilde - h)

        # lesbarer print von delta_x
        # print("delta x = K * (h(x) - g^h):")
        # for comp in get_def("imu error state")["components"]:
        #     print(
        #         f"{comp['name']}:\n{np.array2string(delta_x[comp['slice']], precision=3)}"
        #     )

        self.x = esekf_funcs["x_imu_true"](*self.x, *delta_x).flatten()
        self.x[self.q_slice] = normalize_vector(self.x[self.q_slice])

        # standard form
        # self.P = (np.eye(15) - K @ H) @ self.P

        # joseph form
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.W @ K.T
        self.P = (self.P + self.P.T) / 2

        # print(f"det of P after correction step {np.linalg.det(self.P)}\n")
