import sympy as sym
import numpy as np
import time
import os
import sys


def normalize_vector(vec):
    return vec / np.linalg.norm(vec)


def R(q):
    """
    turns input quaternion into rotation matrix to be multiplied from the left to a vector

    :param q: quaternion as sympy or numpy array
    :return: sympy or numpy Matrix, depending on the input type
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rotation_matrix = [
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ]
    return _match_output_to_input_type(q, rotation_matrix)


def quat_from_axis_angle(Phi, small_angle=True):
    """
    definition see Solà 2017
    """
    if isinstance(Phi, sym.Matrix):
        if small_angle:
            return sym.Matrix([1, *(0.5 * Phi)])

        norm = sym.sqrt(Phi[0] ** 2 + Phi[1] ** 2 + Phi[2] ** 2)
        return sym.Matrix(
            [sym.cos(0.5 * norm), *(Phi * (1 / norm) * sym.sin(0.5 * norm))]
        )

    else:
        if small_angle:
            return np.array([1, *(0.5 * Phi)])

        norm = np.linalg.norm(Phi)
        return np.array([np.cos(0.5 * norm), *(Phi * (1 / norm) * np.sin(0.5 * norm))])


def skew_symmetric_matrix(omega):
    """
    definition see Solà 2017
    """
    omega_x, omega_y, omega_z = omega[0], omega[1], omega[2]
    matrix = [[0, -omega_z, omega_y], [omega_z, 0, -omega_x], [-omega_y, omega_x, 0]]
    return _match_output_to_input_type(omega, matrix)


def quat_inv(q):
    """
    construct the inverse of a quaternion

    :param q: quaternion
    :return: quaternion inverse
    """
    inverse = [q[0], -q[1], -q[2], -q[3]]
    return _match_output_to_input_type(q, inverse)


def left_quat_matrix(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    matrix = [
        [qw, -qx, -qy, -qz],
        [qx, qw, -qz, qy],
        [qy, qz, qw, -qx],
        [qz, -qy, qx, qw],
    ]
    return _match_output_to_input_type(q, matrix)


def _match_output_to_input_type(var, output):
    """
    helps return the correct type, based on the input type

    :param var: possible input types: sympy Matrix, numpy array or default python array
    """
    if isinstance(var, sym.Matrix):
        return sym.Matrix(output)
    if isinstance(var, np.ndarray):
        return np.array(output)
    return output


# ==================
# generally usefull helper methods
# ==================


class Timer:
    def __init__(self):
        self.starting_time = time.time()
        self.last_checkpoint = self.starting_time

    def check(self):
        t = time.time()
        time_string = f"(after {t - self.last_checkpoint:.2f}s, total {t - self.starting_time:.2f}s)"
        self.last_checkpoint = t
        return time_string
