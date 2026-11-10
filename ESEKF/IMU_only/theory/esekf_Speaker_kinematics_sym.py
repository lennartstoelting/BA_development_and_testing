import sympy as sym
import numpy as np

from .esekf_definitions import *
from .esekf_IMU_kinematics_sym import *

# ===
# fixed values
# ===

# \Delta p_{I \to c}
IMU_cam_x_translation = 0
IMU_cam_y_translation = sym.Symbol(r"\Delta{p}_y", negative=True)
IMU_cam_z_translation = sym.Symbol(r"\Delta{p}_z", negative=True)
IMU_cam_translation = sym.Matrix(
    [IMU_cam_x_translation, IMU_cam_y_translation, IMU_cam_z_translation]
)


# \Delta q_{I \to c}
IMU_cam_w_rotation = sym.Symbol(r"\Delta{q}_w", real=True)
IMU_cam_x_rotation = sym.Symbol(r"\Delta{q}_x", real=True)
IMU_cam_y_rotation = 0
IMU_cam_z_rotation = 0
IMU_cam_rotation = sym.Matrix(
    [IMU_cam_w_rotation, IMU_cam_x_rotation, IMU_cam_y_rotation, IMU_cam_z_rotation]
)

# ===
# nominal state
# ===

p_s = get_dims_as_sym("p_s")
v_s = get_dims_as_sym("v_s")
q_s = get_dims_as_sym("q_s")

x_s = sym.Matrix.vstack(p_s, v_s, q_s)

# kinematics

# continuous
p_s_dot = v_s
v_s_dot = sym.zeros(3, 1)
q_s_dot = sym.zeros(4, 1)

x_s_dot = sym.Matrix.vstack(p_s_dot, v_s_dot, q_s_dot)

# discrete
f_p_s = p_s + v_s * dt
f_v_s = v_s
f_q_s = q_s

f_x_s = sym.Matrix.vstack(f_p_s, f_v_s, f_q_s)

# ===
# error state
# ===

delta_p_s = get_dims_as_sym("delta_p_s")
delta_v_s = get_dims_as_sym("delta_v_s")
delta_theta_s = get_dims_as_sym("delta_theta_s")

delta_x_s = sym.Matrix.vstack(delta_p_s, delta_v_s, delta_theta_s)

# kinematics

# dontinuous
delta_a_s_per = get_dims_as_sym("delta_a_s_per")
delta_omega_s_per = get_dims_as_sym("delta_omega_s_per")

delta_p_s_dot = delta_v_s
delta_v_s_dot = delta_a_s_per
delta_theta_s_dot = delta_omega_s_per  # not defined, das muss ich noch ausarbeiten

delta_x_s_dot = sym.Matrix.vstack(delta_p_s_dot, delta_v_s_dot, delta_theta_s_dot)

# discrete
delta_v_s_n = get_dims_as_sym("delta_v_s_n")
delta_theta_s_n = get_dims_as_sym("delta_theta_s_n")

delta_x_s_n = sym.Matrix.vstack(delta_v_s_n, delta_theta_s_n)

f_delta_p_s = delta_p_s + delta_v_s * dt
f_delta_v_s = delta_v_s + delta_v_s_n
f_delta_theta_s = delta_theta_s + delta_theta_s_n

f_delta_x_s = sym.Matrix.vstack(f_delta_p_s, f_delta_v_s, f_delta_theta_s)

# ===
# true state
# ===

p_s_true = p_s + delta_p_s
v_s_true = v_s + delta_v_s
q_s_true = left_quat_matrix(q_s) * quat_from_axis_angle(delta_theta_s)

x_s_true = sym.Matrix.vstack(p_s_true, v_s_true, q_s_true)

# ===
# measurements
# ===

q_h_s = R(IMU_cam_rotation) * (R(q) * (p_s - p) - IMU_cam_translation)
p_h_s = left_quat_matrix(quat_inv(left_quat_matrix(q) * IMU_cam_rotation)) * q_s
