import sympy as sym
import numpy as np
import time
import random
from scipy.linalg import null_space


from esekf_definitions import *

# ===
# fixed values
# ===

dt = sym.Symbol(r"\Delta t", real=True, positive=True)

gravity_sym = sym.Symbol(r"g", real=True, positive=True)
gravity_vector_sym = sym.Matrix([0, 0, -1 * gravity_sym])

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
# input
# ===

a_tilde = get_dims_as_sym("a_tilde")
omega_tilde = get_dims_as_sym("omega_tilde")

u_tilde = sym.Matrix.vstack(a_tilde, omega_tilde)


# ===
# nominal state
# ===

# IMU

p = get_dims_as_sym("p")
v = get_dims_as_sym("v")
q = get_dims_as_sym("q")
a_b = get_dims_as_sym("a_b")
omega_b = get_dims_as_sym("omega_b")

x_IMU = sym.Matrix.vstack(p, v, q, a_b, omega_b)

# IMU kinematics

# Continuous
p_dot = v
v_dot = R(q) * (a_tilde - a_b) + gravity_vector_sym
q_dot = 0.5 * (left_quat_matrix(q) * sym.Matrix([0, *(omega_tilde - omega_b)]))
a_b_dot = sym.zeros(3, 1)
omega_b_dot = sym.zeros(3, 1)

x_IMU_dot = sym.Matrix.vstack(p_dot, v_dot, q_dot, a_b_dot, omega_b_dot)

# discrete
f_p = p + v * dt + 0.5 * (R(q) * (a_tilde - a_b) + gravity_vector_sym) * dt**2
f_v = v + (R(q) * (a_tilde - a_b) + gravity_vector_sym) * dt
f_q = left_quat_matrix(q) * quat_from_axis_angle((omega_tilde - omega_b) * dt)
f_a_b = a_b
f_omega_b = omega_b

f_x_IMU = sym.Matrix.vstack(f_p, f_v, f_q, f_a_b, f_omega_b)

# Speakers

p_s = get_dims_as_sym("p_s")
v_s = get_dims_as_sym("v_s")
q_s = get_dims_as_sym("q_s")

x_s = sym.Matrix.vstack(p_s, v_s, q_s)

# Speaker kinematics

p_s_dot = v_s
v_s_dot = sym.zeros(3, 1)
q_s_dot = sym.zeros(4, 1)

x_s_dot = sym.Matrix.vstack(p_s_dot, v_s_dot, q_s_dot)

f_p_s = p_s + v_s * dt
f_v_s = v_s
f_q_s = q_s

f_x_s = sym.Matrix.vstack(f_p_s, f_v_s, f_q_s)

# ===
# error state
# ===

# IMU

delta_p = get_dims_as_sym("delta_p")
delta_v = get_dims_as_sym("delta_v")
delta_theta = get_dims_as_sym("delta_theta")
delta_a_b = get_dims_as_sym("delta_a_b")
delta_omega_b = get_dims_as_sym("delta_omega_b")

delta_x_IMU = sym.Matrix.vstack(delta_p, delta_v, delta_theta, delta_a_b, delta_omega_b)

# IMU kinematics

# Continuous
a_tilde_n = get_dims_as_sym("a_tilde_n")
omega_tilde_n = get_dims_as_sym("omega_tilde_n")

a_b_tilde_n = get_dims_as_sym("a_b_tilde_n")
omega_b_tilde_n = get_dims_as_sym("omega_b_tilde_n")

delta_p_dot = delta_v
delta_v_dot = (
    -R(q) * skew_symmetric_matrix(a_tilde - a_b) * delta_theta
    - R(q) * delta_a_b
    - R(q) * a_tilde_n
)
delta_theta_dot = (
    -1 * skew_symmetric_matrix(omega_tilde - omega_b) * delta_theta
    - delta_omega_b
    - omega_tilde_n
)
# TODO a_b_n existiert noch gar nicht weil ichs bisher nicht brauchte
# ich könnte fürs Erste einfach die noise Werte vom discrete case benutzen, ich will es ja nur in der observability nachweisen
delta_a_b_dot = a_b_tilde_n  # technically wrong
delta_omega_b_dot = omega_b_tilde_n  # technically wrong

delta_x_IMU_dot = sym.Matrix.vstack(
    delta_p_dot, delta_v_dot, delta_theta_dot, delta_a_b_dot, delta_omega_b_dot
)

# Discrete
delta_v_n = get_dims_as_sym("delta_v_n")
delta_theta_n = get_dims_as_sym("delta_theta_n")
delta_a_b_n = get_dims_as_sym("delta_a_b_n")
delta_omega_b_n = get_dims_as_sym("delta_omega_b_n")

delta_x_n_IMU = sym.Matrix.vstack(
    delta_v_n, delta_theta_n, delta_a_b_n, delta_omega_b_n
)

f_delta_p = delta_p + delta_v * dt
f_delta_v = (
    delta_v
    + (-R(q) * skew_symmetric_matrix(a_tilde - a_b) * delta_theta - R(q) * delta_a_b)
    * dt
    + delta_v_n
)
f_delta_theta = (
    R(quat_from_axis_angle((omega_tilde - omega_b) * dt)).T * delta_theta
    - delta_omega_b * dt
    + delta_theta_n
)
f_delta_a_b = delta_a_b + delta_a_b_n
f_delta_omega_b = delta_omega_b + delta_omega_b_n

f_delta_x_IMU = sym.Matrix.vstack(
    f_delta_p, f_delta_v, f_delta_theta, f_delta_a_b, f_delta_omega_b
)

# Speakers

delta_p_s = get_dims_as_sym("delta_p_s")
delta_v_s = get_dims_as_sym("delta_v_s")
delta_theta_s = get_dims_as_sym("delta_theta_s")

delta_x_s = sym.Matrix.vstack(delta_p_s, delta_v_s, delta_theta_s)

# Speaker kinematics

# Continuous
delta_a_s_per = get_dims_as_sym("delta_a_s_per")
delta_omega_s_per = get_dims_as_sym("delta_omega_s_per")

delta_p_s_dot = delta_v_s
delta_v_s_dot = delta_a_s_per
delta_theta_s_dot = delta_omega_s_per  # not defined, das muss ich noch ausarbeiten

delta_x_s_dot = sym.Matrix.vstack(delta_p_s_dot, delta_v_s_dot, delta_theta_s_dot)

# Discrete
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

p_true = p + delta_p
v_true = v + delta_v
q_true = left_quat_matrix(q) * quat_from_axis_angle(delta_theta)
a_b_true = a_b + delta_a_b
omega_b_true = omega_b + delta_omega_b

x_IMU_true = sym.Matrix.vstack(p_true, v_true, q_true, a_b_true, omega_b_true)

p_s_true = p_s + delta_p_s
v_s_true = v_s + delta_v_s
q_s_true = left_quat_matrix(q_s) * quat_from_axis_angle(delta_theta_s)

x_s_true = sym.Matrix.vstack(p_s_true, v_s_true, q_s_true)

# ===
# measurements
# ===

# IMU
g_h = R(q).T * gravity_vector_sym + a_b

# face detection
q_h_s = R(IMU_cam_rotation) * (R(q) * (p_s - p) - IMU_cam_translation)
p_h_s = left_quat_matrix(quat_inv(left_quat_matrix(q) * IMU_cam_rotation)) * q_s

# fake measurements

p_h = p

psi_h = sym.Matrix(
    [sym.atan2(2 * (q[0] * q[3] - q[1] * q[2]), 1 - 2 * (q[1] ** 2 + q[3] ** 2))]
)
