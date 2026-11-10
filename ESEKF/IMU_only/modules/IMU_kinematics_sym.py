import sympy as sym
import numpy as np

from modules import *

# ===
# fixed values
# ===

dt = sym.Symbol(r"\Delta t", real=True, positive=True)

gravity_sym = sym.Symbol(r"g", real=True, positive=True)
gravity_vector_sym = sym.Matrix([0, 0, -1 * gravity_sym])

# ===
# input
# ===

a_tilde = get_dims_as_sym("a_tilde")
omega_tilde = get_dims_as_sym("omega_tilde")

u_tilde = sym.Matrix.vstack(a_tilde, omega_tilde)

# ===
# nominal state
# ===

# state

p = get_dims_as_sym("p")
v = get_dims_as_sym("v")
q = get_dims_as_sym("q")
a_b = get_dims_as_sym("a_b")
omega_b = get_dims_as_sym("omega_b")

x_IMU = sym.Matrix.vstack(p, v, q, a_b, omega_b)

# kinematics

# continuous
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

# ===
# error state
# ===

delta_p = get_dims_as_sym("delta_p")
delta_v = get_dims_as_sym("delta_v")
delta_theta = get_dims_as_sym("delta_theta")
delta_a_b = get_dims_as_sym("delta_a_b")
delta_omega_b = get_dims_as_sym("delta_omega_b")

delta_x_IMU = sym.Matrix.vstack(delta_p, delta_v, delta_theta, delta_a_b, delta_omega_b)

# kinematics

# continuous
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
delta_a_b_dot = a_b_tilde_n
delta_omega_b_dot = omega_b_tilde_n

delta_x_IMU_dot = sym.Matrix.vstack(
    delta_p_dot, delta_v_dot, delta_theta_dot, delta_a_b_dot, delta_omega_b_dot
)

# discrete
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

# ===
# true state
# ===

p_true = p + delta_p
v_true = v + delta_v
q_true = left_quat_matrix(q) * quat_from_axis_angle(delta_theta)
a_b_true = a_b + delta_a_b
omega_b_true = omega_b + delta_omega_b

x_IMU_true = sym.Matrix.vstack(p_true, v_true, q_true, a_b_true, omega_b_true)

# ===
# measurements
# ===

g_h = R(q).T * gravity_vector_sym + a_b

p_h = p

psi_h = sym.Matrix(
    [sym.atan2(2 * (q[0] * q[3] - q[1] * q[2]), 1 - 2 * (q[1] ** 2 + q[3] ** 2))]
)
