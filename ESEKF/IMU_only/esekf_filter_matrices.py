import sympy as sym
import numpy as np
import dill

from theory.esekf_definitions import *
from theory.esekf_IMU_kinematics_sym import *
from theory.esekf_Speaker_kinematics_sym import *


def assert_inputs_match(expr, input_symbols, name):
    input_set = set(input_symbols)

    # in expr, but not lambdified -> would become a global
    missing = expr.free_symbols - input_set
    # lambdified, but not in expr -> suspicious
    unused = input_set - expr.free_symbols

    if missing:
        raise ValueError(
            f"CRITICAL: {name} forgot to include these symbols in lambdify: {missing}"
        )
    if unused:
        raise ValueError(
            f"WARNING-level strict: {name} has unused lambdify inputs "
            f"(not in {name} expression): {unused}"
        )


# nominal state kinematics function $f$
# ---
# I need to set the order of inputs that will later be used when calling the function.
# In this case I decide on f(*x, *u, gravity, dt)
f_x_IMU_input_symbols = list(x_IMU) + list(u_tilde) + [gravity_sym] + [dt]
assert_inputs_match(f_x_IMU, f_x_IMU_input_symbols, "f_x_IMU")
f_x_IMU_lambdified = sym.lambdify(f_x_IMU_input_symbols, f_x_IMU, "numpy")

# error state kinematics jacobian $F_{\delta x}{}$
# ---
F_delta_x = f_delta_x_IMU.jacobian(delta_x_IMU)
F_delta_x = F_delta_x.subs({var: 0 for var in delta_x_IMU})
F_delta_x = sym.simplify(F_delta_x)
F_delta_x_input_symbols = list(q) + list(a_b) + list(omega_b) + list(u_tilde) + [dt]
assert_inputs_match(F_delta_x, F_delta_x_input_symbols, "F_delta_x")
F_delta_x_lambdified = sym.lambdify(F_delta_x_input_symbols, F_delta_x, "numpy")

# true state in order to perform error injection later
# ---
x_IMU_true_symbols = list(x_IMU) + list(delta_x_IMU)
assert_inputs_match(x_IMU_true, x_IMU_true_symbols, "x_IMU_true")
x_IMU_true_lambdified = sym.lambdify(x_IMU_true_symbols, x_IMU_true, "numpy")


# error noise jacobian $J$
# ---
# jacobian of error function with respect to pertubation impulses $j$
# its always some form of almost identity matrix because of the addititve error
J = f_delta_x_IMU.jacobian(delta_x_n_IMU)
J_lambdified = sym.lambdify([], J, "numpy")

# error noise covariance $Q{}$
# ---
# only temporary implementation approach, this will later need to be more complex
# as it will not just have purely diagonal elements but the diagonal is comprised of covariancematrices itself
Q_diagonal = sym.Matrix([])
for i in range(len(delta_x_n_IMU)):
    Q_diagonal = sym.Matrix.vstack(Q_diagonal, sym.Matrix([sym.Symbol(f"Q_{i}")]))
Q = sym.diag(*Q_diagonal)
Q_lambdified = sym.lambdify(Q_diagonal, Q, "numpy")

# measurement function $h$ ✔
# ---
# similar to the observability tests, I need to decide which measurements to include
h = sym.Matrix.vstack(g_h)  # ,p_h, psi_h
h_input_symbols = list(q) + list(a_b) + [gravity_sym]
assert_inputs_match(h, h_input_symbols, "h")
h_lambdified = sym.lambdify(h_input_symbols, h, "numpy")

# measurement jacobian $H$
H_delta_x = h.jacobian(x_IMU) * x_IMU_true.jacobian(delta_x_IMU)
H_delta_x = H_delta_x.subs({var: 0 for var in delta_x_IMU})
H_delta_x = sym.simplify(H_delta_x)
H_delta_x_input_symbols = list(q) + [gravity_sym]
assert_inputs_match(H_delta_x, H_delta_x_input_symbols, "H_delta_x")
H_delta_x_lambdified = sym.lambdify(H_delta_x_input_symbols, H_delta_x, "numpy")

# measurement noise $V$
# ---
# Same as Q, only temporary because I haven't yet decided on the correct conventions as well
V_diagonal = sym.Matrix([])
for i in range(len(h)):
    V_diagonal = sym.Matrix.vstack(V_diagonal, sym.Matrix([sym.Symbol(f"V_{i}")]))
V = sym.diag(*V_diagonal)
V_lambdified = sym.lambdify(V_diagonal, V, "numpy")

with open("filter_funcs_IMU_only.pkl", "wb") as f:
    dill.dump(
        {
            "f_x_IMU": f_x_IMU_lambdified,
            "F_delta_x": F_delta_x_lambdified,
            "x_IMU_true": x_IMU_true_lambdified,
            "J": J_lambdified,
            "Q": Q_lambdified,
            "h": h_lambdified,
            "H_delta_x": H_delta_x_lambdified,
            "V": V_lambdified,
        },
        f,
    )
