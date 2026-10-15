import numpy as np
from scipy import constants
import pandas as pd
import dill

folder_path = "../notebook theory/"
with open(f"{folder_path}filter_funcs_IMU_only.pkl", "rb") as f:
    esekf_funcs = dill.load(f)


class Filter:
    def __init__(self, calibration__time_length_in_sec=2):
        self.calibration__time_length_in_sec = calibration__time_length_in_sec
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

    def calibration_step(self):
        # TODO append acceleration and gyro data to the respective self.[...]buffer

        # if the calibration time limit is reached, initialize the filter
        # by initializing state, covariance matrix and maybe more like maybe noise but maybe also that one will get set earlier
        pass

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
