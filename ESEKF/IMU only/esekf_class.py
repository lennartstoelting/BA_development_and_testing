import numpy as np
import pandas as pd
import dill

from esekf_plot import *


folder_path = "../notebook theory/"
with open(f"{folder_path}filter_funcs_IMU_only.pkl", "rb") as f:
    esekf_funcs = dill.load(f)


output_columns = [
    "p_x",
    "p_y",
    "p_z",
    "v_x",
    "v_y",
    "v_z",
    "q_w",
    "q_x",
    "q_y",
    "q_z",
    "a_b_x",
    "a_b_y",
    "a_b_z",
    "omega_b_x",
    "omega_b_y",
    "omega_b_z",
]


class Filter:
    def __init__(self, imu_data, calibration_dt=3):

        self.x = None
        self.P = None
        self.Q = None
        self.J = None
        self.W = None

        # this is only to test the basic functionality.
        # Since I will not have it always be the case that there is zero speakers and the only measurement is gravity,
        # I need to construct the matrices individually but also ideally automatically
        self._initialize_error_state_covariance_matrix_P()
        self._initialize_process_noise_covariance_matrix_Q()
        self._initialize_process_noise_jacobian_matrix_J()
        self._initialize_measurement_noise_covariance_matrix_W()
        # process noise jacobian matrix is only needed because not every error state kinematic has added noise
        # for now, all measurement noise has added noise for every aspect until decided differently

        self.calibration_dt = calibration_dt
        self.calibration_complete = False

        self.accel_calibration_buffer = []
        self.gyro_calibration_buffer = []

        self.state_history = []
        self.input_and_state_history = imu_data
        self.input_and_state_history.loc[:, output_columns] = None

        # wie sieht diese csv denn aus?
        self.input_and_state_history.to_csv("test_output.csv", index=False)

    def calibration_step(self, accel_in_g, omega_tilde, elapsed_time):
        self.accel_calibration_buffer.append(accel_in_g)
        self.gyro_calibration_buffer.append(omega_tilde)

        if elapsed_time < self.calibration_dt:
            # keep calibarting if we are not over the calibration time window
            return

        self._initialize_state_x()

        self.calibration_complete = True
        # print(
        #     f"\nCalibration ended after {elapsed_time:.6f} seconds | {len(self.accel_calibration_buffer)} samples used | Initial state after calibration:\n"
        # )
        # for i in self.x:
        #     print(i)

        self.accel_calibration_buffer = []
        self.gyro_calibration_buffer = []

    def _initialize_state_x(self):
        # calibration data
        accel_array = np.array(self.accel_calibration_buffer)
        gyro_array = np.array(self.gyro_calibration_buffer)

        mean_accel = np.mean(accel_array, axis=0)
        mean_gyro = np.mean(gyro_array, axis=0)

        # position and velocity
        p_init = np.zeros(3)
        v_init = np.zeros(3)

        # orientation
        gravity_in_g = np.array([0.0, 0.0, 1.0])
        qw_init = 1.0 + np.dot(self.normalized(mean_accel), gravity_in_g)
        qx_init, qy_init, qz_init = np.cross(self.normalized(mean_accel), gravity_in_g)

        q_init = self.normalized([qw_init, qx_init, qy_init, qz_init])

        # accel bias
        a_b_init = mean_accel - self.normalized(mean_accel)

        # gyro bias
        omega_b_init = mean_gyro

        self.x = np.array([*p_init, *v_init, *q_init, *a_b_init, *omega_b_init])

    def _initialize_error_state_covariance_matrix_P(self):
        # for now, these values are trial and error
        # (low number -> low variance -> high certainty)
        initial_certainties = {
            "position": 1e-10,
            "velocity": 1e-10,
            "orientation": 1e-7,
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
        measurement_noise = {"accel_gravity": 1e-3}

        measurement_noise_order = ["accel_gravity"]

        self.W = np.diag(
            np.repeat([measurement_noise[key] for key in measurement_noise_order], 3)
        )

    def _initialize_process_noise_jacobian_matrix_J(self):
        self.J = np.vstack((np.zeros((3, 12)), np.eye(12)))

    def prediction_step(self, a_tilde, omega_tilde, gravity_constant, dt):
        f_x_IMU = esekf_funcs["f_x_IMU"](
            *self.x, *a_tilde, *omega_tilde, gravity_constant, dt
        ).flatten()
        # list(q) + list(a_b) + list(omega_b) + list(u_tilde) + [dt]
        F_IMU = esekf_funcs["F_delta_x"](*self.x[6:16], *a_tilde, *omega_tilde, dt)

        self.x = f_x_IMU
        self.x[6:10] = self.normalized(self.x[6:10])

        self.P = F_IMU @ self.P @ F_IMU.T + self.J @ self.Q @ self.J.T
        # I originally called the lambdified functions here with the neccessary arguments
        # but in the ekf I didn't have all the kinematics modeled, only for the orientation_certainty
        # and then I did the vector multiplication in code like this
        # self.P = A @ self.P @ A.T + self.Q
        # probably actually smart because it is after all only multiplication and some transposes

    # def plot_data(self):
    #     if not self.state_history:
    #         print("No states to save.")
    #         return

    #     df = pd.DataFrame(
    #         self.input_and_state_history, columns=output_columns + ["time"]
    #     )
    #     df.to_csv("IMU_input_and_states_over_time.csv")

    #     pyplot_input_and_state_grid(
    #         "IMU_input_and_states_over_time.csv", "placeholder title"
    #     )

    def plot_data(self):
        if not self.state_history:
            print("No states to save.")
            return

        # Save the complete history (inputs + estimated states) directly
        self.input_and_state_history.to_csv(
            "IMU_input_and_states_over_time.csv", index=False
        )

        pyplot_input_and_state_grid(
            "IMU_input_and_states_over_time.csv", "placeholder title"
        )

    def correction_step(self):
        # only to be implemented when I have succesfully done the prediction step to a satisfactory level
        pass

    @staticmethod
    def normalized(vec):
        return vec / np.linalg.norm(vec)
