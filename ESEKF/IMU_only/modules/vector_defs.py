import sympy as sym
import numpy as np

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


# physical constants
GRAVITY_CONSTANT = 9.80665
g2ms2 = lambda g: g * GRAVITY_CONSTANT
GRAVITY_VEC = [0, 0, g2ms2(-1)]

# system constants
# \Delta p_{I \to c}
IMU_CAM_TRANSLATION = [0, -0.0013, -0.00662]
# \Delta q_{I \to c}
IMU_CAM_ROTATION = [np.cos(-np.deg2rad(102) / 2), np.sin(-np.deg2rad(102) / 2), 0, 0]

# length of timewindow at the start for imu calibration in seconds
CALIBRATION_DT = 3

# estimate of running average over multiple recordings
IMU_SAMPLE_FREQ = 103.5


# ---------------------------------------------------------------------------
# dictionary of variables
# ---------------------------------------------------------------------------


def _create_vector(name: str, var: str, tex: str, components: list):
    """one time call to assemble the ESEKF_VECTORS into a list of dictionaries

    Args:
        name (str): Name of the vector
        var (str): how the vector should be called in code
        tex (str): string of latex math for display purposes
        components (list): most vectors are stacked subcomponents or vectors

    Returns:
        _type_: the written out dictionary for this vector
    """
    return {
        "name": name,
        "var": var,
        "tex": tex,
        "components": ([_create_component(*c) for c in components]),
    }


def _create_component(name: str, unit: str, var: str, tex: str, dims: int):
    sub_components = [tex]

    if dims in (3, 4):
        dimensions = ["x", "y", "z"] if dims == 3 else ["w", "x", "y", "z"]

        if "_" in tex:
            sub_components = [tex[:-1] + r"\," + d + r"}" for d in dimensions]
        else:
            sub_components = [tex + r"_{" + d + r"}" for d in dimensions]

    return {
        "name": name,
        "unit": unit,
        "var": var,
        "tex": tex,
        "dims": sub_components,
    }


ESEKF_VECTORS = [
    _create_vector(*v)
    for v in [
        [
            "imu nominal state",
            "x_imu",
            r"x_{imu}",
            [
                ["imu position", "m", "p", r"p", 3],
                ["imu velocity", "m/s", "v", r"v", 3],
                ["imu orientation", "", "q", r"q", 4],
                ["imu accel bias", "m/s^2", "a_b", r"a_{b}", 3],
                ["imu gyro bias", "rad/s", "omega_b", r"\omega_{b}", 3],
            ],
        ],
        [
            "speaker nominal state",
            "x_s",
            r"x_{s}",
            [
                ["speaker position", "m", "p_s", r"p_{s}", 3],
                ["speaker velocity", "m/s", "v_s", r"v_{s}", 3],
                ["speaker orientation", "", "q_s", r"q_{s}", 4],
            ],
        ],
        [
            "imu error state",
            "delta_x_imu",
            r"\delta{x}_{imu}",
            [
                # fmt: off
                ["imu position error", "m", "delta_p", r"\delta{p}", 3],
                ["imu velocity error", "m/s", "delta_v", r"\delta{v}", 3],
                ["imu orientation error", "rad", "delta_theta", r"\delta{\theta}", 3],
                ["imu accel bias error", "m/s^2", "delta_a_b", r"\delta{a}_{b}", 3],
                ["imu gyro bias error", "rad/s", "delta_omega_b", r"\delta{\omega}_{b}", 3],
                # fmt: on
            ],
        ],
        [
            "speaker error state",
            "delta_x_s",
            r"\delta{x}_{s}",
            [
                # fmt: off
                ["speaker position error", "m", "delta_p_s", r"\delta{p}_{s}", 3],
                ["speaker velocity error", "m/s", "delta_v_s", r"\delta{v}_{s}", 3],
                ["speaker orientation error", "rad", "delta_theta_s", r"\delta{\theta}_{s}", 3],
                # fmt: on
            ],
        ],
        [
            "imu true state",
            "x_hat_imu",
            r"\hat{x}_{imu}",
            [
                ["imu position true", "m", "p_hat", r"\hat{p}", 3],
                ["imu velocity true", "m/s", "v_hat", r"\hat{v}", 3],
                ["imu orientation true", "", "q_hat", r"\hat{q}", 4],
                ["imu accel bias true", "m/s^2", "a_hat_b", r"\hat{a}_{b}", 3],
                ["imu gyro bias true", "rad/s", "omega_hat_b", r"\hat{\omega}_{b}", 3],
            ],
        ],
        [
            "speaker true state",
            "x_hat_s",
            r"\hat{x}_{s}",
            [
                ["speaker position true", "m", "p_hat_s", r"\hat{p}_{s}", 3],
                ["speaker velocity true", "m/s", "v_hat_s", r"\hat{v}_{s}", 3],
                ["speaker orientation true", "", "q_hat_s", r"\hat{q}_{s}", 4],
            ],
        ],
        [
            "imu input noise & perturbations (continuous time)",
            "u_tilde_n",
            r"\tilde{u}_{n}",
            [
                # fmt: off
                ["imu accelerometer noise", "m/s^2", "a_tilde_n", r"\tilde{a}_{n}", 3],
                ["imu gyroscope noise", "rad/s", "omega_tilde_n", r"\tilde{\omega}_{n}", 3],
                ["imu accel bias perturbations", "m/s^3", "a_b_tilde_n", r"\tilde{a}_{bn}", 3],
                ["imu gyro bias perturbations", "rad/s^2", "omega_b_tilde_n", r"\tilde{\omega}_{bn}", 3],
                # fmt: on
            ],
        ],
        [
            "imu process noise (discrete time)",
            "delta_x_n_imu",
            r"\delta{x}_{IMUn}",
            [
                # fmt: off
                ["imu velocity error noise", "m/s", "delta_v_n", r"\delta{v}_{n}", 3],
                ["imu orientation error noise", "rad", "delta_theta_n", r"\delta{\theta}_{n}", 3],
                ["imu accel bias error noise", "m/s^2", "delta_a_b_n", r"\delta{a}_{bn}", 3],
                ["imu gyro bias error noise", "rad/s", "delta_omega_b_n", r"\delta{\omega}_{bn}", 3],
                # fmt: on
            ],
        ],
        [
            "speaker perturbations (continuous time)",
            "delta_x_s_per",
            r"\delta{x}_{s\text{per}}",
            [
                # fmt: off
                ["speaker velocity error perturbations", "m/s^2", "delta_a_s_per", r"\delta{a}_{s\text{per}}", 3],
                ["speaker orientation error perturbations", "rad/s", "delta_omega_s_per", r"\delta{\omega}_{s\text{per}}", 3],
                # fmt: on
            ],
        ],
        [
            "speaker process noise (discrete time)",
            "delta_x_s_n",
            r"\delta{x}_{sn}",
            [
                # fmt: off
                ["speaker velocity error noise", "m/s", "delta_v_s_n", r"\delta{v}_{sn}", 3],
                ["speaker orientation error noise", "rad", "delta_theta_s_n", r"\delta{\theta}_{sn}", 3],
                # fmt: on
            ],
        ],
        [
            "imu input",
            "u_tilde",
            r"\tilde{u}",
            [
                ["imu accelerometer", "m/s^2", "a_tilde", r"\tilde{a}", 3],
                ["imu gyroscope", "rad/s", "omega_tilde", r"\tilde{\omega}", 3],
            ],
        ],
        [
            "predicted measurements",
            "h",
            r"h",
            [
                # fmt: off
                ["predicted gravity measurement", "m/s^2", "a_g_h", r"a^{h}_{g}", 3],
                ["predicted speaker position relative to camera", "m", "p_s_h", r"p^{h}_{s}", 3],
                ["predicted speaker orientation relative to camera", "", "q_s_h", r"q^{h}_{s}", 4],
                ["predicted imu yaw", "rad", "psi_h", r"\psi^{h}", 1],
                # fmt: on
            ],
        ],
    ]
]


def _add_components_slice():
    """
    adds the slice key to each (sub-)component dictionary
    together with the start/stop index of the slice (range). \n
    Also adds the length of each vector, adding up all the (sub-)components lengths (dims)
    """
    for vec in ESEKF_VECTORS:
        current_index = 0
        for component in vec["components"]:
            dims = len(component["dims"])
            component["slice"] = slice(current_index, current_index + dims)
            current_index += dims
        vec["len"] = current_index


_add_components_slice()


# ---------------------------------------------------------------------------
# variables dictionary helper functions
# ---------------------------------------------------------------------------


def get_def(target_value, target_key="name", node=ESEKF_VECTORS):
    """
    Recursively search the schema definition tree for an element matching target_key == target_value
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


def get_sympy_vec(target_value, target_key="var"):
    item = get_def(target_value, target_key)
    return sym.Matrix(sym.symbols(item["dims"]))


def main():
    """prints out assembled dictionary for debugging"""
    for vec in ESEKF_VECTORS:
        print(f"\n{vec["name"]}\n")
        for comp in vec["components"]:
            for index, (key, value) in enumerate(comp.items()):
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
