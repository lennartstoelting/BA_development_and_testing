import sympy as sym
import numpy as np

"""
this file tries to collect all definitions
together with useful functions to apply thos definitions in code

this includes:
- variables and vectors
- helper functions
- constants

the goal is for all files going forward to share the same definitions,
in case aspects, naming or other concepts change later in the process
"""


# ==================
# constants
# ==================

# physical constants

gravity_constant = 9.80665
gravity_vector = [0, 0, -1 * gravity_constant]

pi = 3.141592653589793

# system constants

# \Delta p_{I \to c}
IMU_cam_translation_value = [0, -0.0013, -0.00662]
# \Delta q_{I \to c}
Phi_in_rad = 102 * (pi / 180)
IMU_cam_rotation_value = [np.cos(-Phi_in_rad / 2), np.sin(-Phi_in_rad / 2), 0, 0]

# length of timewindow at the start for IMU calibration in seconds
calibration_dt = 3


# ==================
# dictionary of variables
# ==================

create_component = lambda name, var, tex, dims: {
    "name": name if isinstance(name, str) else None,
    "var": var if isinstance(var, str) else None,
    "latex": tex if isinstance(tex, str) else None,
    "dims": dims if isinstance(dims, int) else None,
}
create_vector = lambda name, var, tex, components: {
    "name": name if isinstance(name, str) else None,
    "var": var if isinstance(var, str) else None,
    "latex": tex if isinstance(tex, str) else None,
    "components": (
        [create_component(*c) for c in components]
        if isinstance(components, list)
        else None
    ),
}

esekf_vectors = [
    create_vector(*v)
    for v in [
        [
            "IMU nominal state",
            "x_IMU",
            r"x_{IMU}",
            [
                ["IMU position", "p", r"p", 3],
                ["IMU velocity", "v", r"v", 3],
                ["IMU orientation", "q", r"q", 4],
                ["IMU accel bias", "a_b", r"a_{b}", 3],
                ["IMU gyro bias", "omega_b", r"\omega_{b}", 3],
            ],
        ],
        [
            "Speaker nominal state",
            "x_s",
            r"x_{s}",
            [
                ["Speaker position", "p_s", r"p_{s}", 3],
                ["Speaker velocity", "v_s", r"v_{s}", 3],
                ["Speaker orientation", "q_s", r"q_{s}", 4],
            ],
        ],
        [
            "IMU error state",
            "delta_x_IMU",
            r"\delta{x}_{IMU}",
            [
                ["IMU position error", "delta_p", r"\delta{p}", 3],
                ["IMU velocity error", "delta_v", r"\delta{v}", 3],
                ["IMU orientation error", "delta_theta", r"\delta{\theta}", 3],
                ["IMU accel bias error", "delta_a_b", r"\delta{a}_{b}", 3],
                ["IMU gyro bias error", "delta_omega_b", r"\delta{\omega}_{b}", 3],
            ],
        ],
        [
            "Speaker error state",
            "delta_x_s",
            r"\delta{x}_{s}",
            [
                ["Speaker position error", "delta_p_s", r"\delta{p}_{s}", 3],
                ["Speaker velocity error", "delta_v_s", r"\delta{v}_{s}", 3],
                [
                    "Speaker orientation error",
                    "delta_theta_s",
                    r"\delta{\theta}_{s}",
                    3,
                ],
            ],
        ],
        [
            "IMU true state",
            "x_hat_IMU",
            r"\hat{x}_{IMU}",
            [
                ["IMU position true", "p_hat", r"\hat{p}", 3],
                ["IMU velocity true", "v_hat", r"\hat{v}", 3],
                ["IMU orientation true", "q_hat", r"\hat{q}", 4],
                ["IMU accel bias true", "a_hat_b", r"\hat{a}_{b}", 3],
                ["IMU gyro bias true", "omega_hat_b", r"\hat{\omega}_{b}", 3],
            ],
        ],
        [
            "Speaker true state",
            "x_hat_s",
            r"\hat{x}_{s}",
            [
                ["Speaker position true", "p_hat_s", r"\hat{p}_{s}", 3],
                ["Speaker velocity true", "v_hat_s", r"\hat{v}_{s}", 3],
                ["Speaker orientation true", "q_hat_s", r"\hat{q}_{s}", 4],
            ],
        ],
        [
            "IMU process noise",
            "delta_x_n_IMU",
            r"\delta{x}_{IMUn}",
            [
                # fmt: off
                ["IMU velocity error noise", "delta_v_n", r"\delta{v}_{n}", 3],
                ["IMU orientation error noise", "delta_theta_n", r"\delta{\theta}_{n}", 3],
                ["IMU accel bias error noise", "delta_a_b_n", r"\delta{a}_{bn}", 3],
                ["IMU gyro bias error noise", "delta_omega_b_n", r"\delta{\omega}_{bn}", 3],
                # fmt: on
            ],
        ],
        [
            "Speaker process noise",
            "delta_x_n_s",
            r"\delta{x}_{sn}",
            [
                # fmt: off
                ["Speaker velocity error noise", "delta_v_s_n", r"\delta{v}_{sn}", 3],
                ["Speaker orientation error noise", "delta_theta_s_n", r"\delta{\theta}_{sn}", 3],
                # fmt: on
            ],
        ],
        [
            "IMU input",
            "u_tilde",
            r"\tilde{u}",
            [
                ["IMU accelerometer (m/s^2)", "a_tilde", r"\tilde{a}", 3],
                ["IMU gyroscope (rad/s)", "omega_tilde", r"\tilde{\omega}", 3],
            ],
        ],
        [
            "IMU input noise",
            "u_tilde_n",
            r"\tilde{u}_{n}",
            [
                # fmt: off
                ["IMU accelerometer noise (m/s^2)", "a_tilde_n", r"\tilde{a}_{n}", 3],
                ["IMU gyroscope noise (rad/s)", "omega_tilde_n", r"\tilde{\omega}_{n}", 3],
                # fmt: on
            ],
        ],
        [
            "predicted maesurements",
            "h",
            r"h",
            [
                # fmt: off
                ["predicted gravity measurement", "a_g_h", r"a^{h}_{g}", 3],
                ["predicted speaker position relative to camera", "p_s_h", r"p^{h}_{s}", 3],
                ["predicted speaker orientation relative to camera", "q_s_h", r"q^{h}_{s}", 3],
                ["predicted IMU yaw", "psi_h", r"\psi^{h}", 1],
                # fmt: on
            ],
        ],
    ]
]


# ==================
# variables dictionary helper functions
# ==================


def get_def(target_value, target_key="name", node=esekf_vectors):
    """
    Recursively search the schema tree for an element matching target_key == target_value
    """
    if isinstance(node, dict):
        if node.get(target_key) == target_value:
            return node
        for val in node.values():
            result = get_def(target_value, target_key, val)
            if result:
                return result
    elif isinstance(node, list):
        for item in node:
            result = get_def(target_value, target_key, item)
            if result:
                return result
    return None


def tex_with_dims(tex_symbol, dim):
    """
    adds the dimensions for each tex symbol, example:
    input: (r"p", 3), output: [r"p_{x}", r"p_{y}", r"p_{z}"]

    adds x,y,z to inputs with three dimensions
    adds w,x,y,z to inputs with four dimensions

    :return: array of strings, formatted to fit standard latex notation
    """
    if dim not in (3, 4):
        return [tex_symbol]
    dimensions = ["x", "y", "z"] if dim == 3 else ["w", "x", "y", "z"]
    if "_" in tex_symbol:
        return [tex_symbol[:-1] + r"\," + i + r"}" for i in dimensions]
    else:
        return [tex_symbol + r"_{" + i + r"}" for i in dimensions]


def get_tex_vec(target_value):
    item = get_def(target_value)
    return tex_with_dims(item["latex"], item["dims"])


def add_components_index():
    for vec in esekf_vectors:
        current_index = 0
        for component in vec["components"]:
            dims = component["dims"]
            component["slice"] = slice(current_index, current_index + dims)
            current_index += dims
        vec["length"] = current_index


add_components_index()


def add_vec_sympy_matrices():
    for vec in esekf_vectors:
        definition = get_def(vec["name"])
        array = [
            vec for c in definition["components"] for vec in get_tex_vec(c["name"])
        ]
        vec["sympy array"] = sym.Matrix(sym.symbols(array))


add_vec_sympy_matrices()


def get_sym_subarray(target_value, target_key="var"):
    """
    get sympy symbols for the components of each vector, just by inputing the variable.
    useful because the vectors sympy matrix is only stored as one big array, getting the matching subarray is difficult
    """
    item = get_def(target_value, target_key)
    for vec in esekf_vectors:
        res = next(
            (c for c in vec["components"] if c == item),
            None,
        )
        if res:
            return sym.Matrix(vec["sympy array"][item["slice"]])


# ==================
# math helper functions
# ==================


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
    return match_output_to_input_type(q, rotation_matrix)


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
    return match_output_to_input_type(omega, matrix)


def quat_inv(q):
    """
    construct the inverse of a quaternion

    :param q: quaternion
    :return: quaternion inverse
    """
    inverse = [q[0], -q[1], -q[2], -q[3]]
    return match_output_to_input_type(q, inverse)


def left_quat_matrix(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    matrix = [
        [qw, -qx, -qy, -qz],
        [qx, qw, -qz, qy],
        [qy, qz, qw, -qx],
        [qz, -qy, qx, qw],
    ]
    return match_output_to_input_type(q, matrix)


def match_output_to_input_type(var, output):
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
