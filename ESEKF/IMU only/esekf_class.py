import numpy as np
import pandas as pd
import dill

folder_path = "../notebook theory/"
with open(f"{folder_path}filter_funcs_IMU_only.pkl", "rb") as f:
    esekf_funcs = dill.load(f)


class Filter:
    def __init__(self, calibration_dt=3):
        self.calibration_dt = calibration_dt
        self.calibration_complete = False

        self.accel_calibration_buffer = []
        self.gyro_calibration_buffer = []

        self.x = None
        self.P = None
        self.Q = None
        self.J = None  # can be initialized right away from the esekf_funcs
        self.W = None

        # this is only to test the basic functionality.
        # Since I will not have it always be the case that there is zero speakers and the only measurement is gravity,
        # I need to construct the matrices individually but also ideally automatically
        self._initialize_error_state_covariance_matrix_P()
        self._initialize_process_noise_covariance_matrix_Q()
        self._initialize_process_noise_jacobian_matrix_J()
        self._initialize_measurement_noise_covariance_matrix_W()

        # TODO this is where I will initialize the array that saves each state
        # for plotting and evaluation later

    def calibration_step(self, accel_in_g, omega_tilde, elapsed_time):
        self.accel_calibration_buffer.append(accel_in_g)
        self.gyro_calibration_buffer.append(omega_tilde)

        # keep calibarting if we are not over the calibration time window
        if elapsed_time < self.calibration_dt:
            return

        self._initialize_state_x()

        self.calibration_complete = True
        print(
            f"\nCalibration ended after {elapsed_time:.6f} seconds | {len(self.accel_calibration_buffer)} samples used | Initial state after calibration:\n"
        )
        for i in self.x:
            print(i)

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

    def prediction_step(self):
        # I originally called the lambdified functions here with the neccessary arguments
        # but in the ekf I didn't have all the kinematics modeled, only for the orientation_certainty
        # and then I did the vector multiplication in code like this
        # self.P = A @ self.P @ A.T + self.Q
        # probably actually smart because it is after all only multiplication and some transposes
        pass

    def correction_step(self):
        # only to be implemented when I have succesfully done the prediction step to a satisfactory level
        pass

    @staticmethod
    def normalized(vec):
        return vec / np.linalg.norm(vec)
