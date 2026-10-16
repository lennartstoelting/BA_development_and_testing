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
        self.Q = None  # can be initialized right away from the esekf_funcs
        self.J = None  # same here, pobably very simple since no input
        self.V = None  # same here

        # TODO this is where I will initialize the array that save each state
        # for plotting and evaluation later

    def calibration_step(self, accel_in_g, omega_tilde, elapsed_time):
        self.accel_calibration_buffer.append(accel_in_g)
        self.gyro_calibration_buffer.append(omega_tilde)

        if elapsed_time < self.calibration_dt:
            return

        self._initialize_state_x()
        # self._initialize_state_covariance_matrix_P()
        # self._initialize_process_noise_covariance_matrix_Q()
        # self._initialize_measurement_noise_covariance_matrix_R()

        self.calibration_complete = True
        print(
            f"\nCalibration ended after {elapsed_time:.6f} seconds | {len(self.accel_calibration_buffer)} samples used | Initial state after calibration:\n"
        )
        for i in self.x:
            print(i)

        self.accel_calibration_buffer = []
        self.gyro_calibration_buffer = []

    def _initialize_state_x(self):
        # initial position and velocity
        p_init = np.zeros(3)
        v_init = np.zeros(3)

        # initial orientation
        accel_array = np.array(self.accel_calibration_buffer)
        mean_accel = np.mean(accel_array, axis=0)

        gravity_in_g = np.array([0.0, 0.0, 1.0])
        qw_init = 1.0 + np.dot(self.normalized(mean_accel), gravity_in_g)
        qx_init, qy_init, qz_init = np.cross(self.normalized(mean_accel), gravity_in_g)

        q_init = self.normalized([qw_init, qx_init, qy_init, qz_init])

        # initial accel bias
        a_b_init = mean_accel - self.normalized(mean_accel)

        # initial gyro bias
        gyro_array = np.array(self.gyro_calibration_buffer)
        mean_gyro = np.mean(gyro_array, axis=0)
        omega_b_init = mean_gyro

        self.x = np.array([*p_init, *v_init, *q_init, *a_b_init, *omega_b_init])

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
