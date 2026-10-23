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
            "nominal state",
            "x",
            r"x",
            [
                ["IMU position", "p", r"p", 3],
                ["IMU velocity", "v", r"v", 3],
                ["IMU orientation", "q", r"q", 4],
                ["IMU accel bias", "a_b", r"a_{b}", 3],
                ["IMU gyro bias", "omega_b", r"\omega_{b}", 3],
                ["Speaker position", "p_s", r"p_{s}", 3],
                ["Speaker velocity", "v_s", r"v_{s}", 3],
                ["Speaker orientation", "q_s", r"q_{s}", 4],
            ],
        ],
        [
            "error state",
            "delta_x",
            r"\delta{x}",
            [
                ["IMU position error", "delta_p", r"\delta{p}", 3],
                ["IMU velocity error", "delta_v", r"\delta{v}", 3],
                ["IMU orientation error", "delta_q", r"\delta{\theta}", 3],
                ["IMU accel bias error", "delta_a_b", r"\delta{a}_{b}", 3],
                ["IMU gyro bias error", "delta_omega_b", r"\delta{\omega}_{b}", 3],
                ["Speaker position error", "delta_p_s", r"\delta{p}_{s}", 3],
                ["Speaker velocity error", "delta_v_s", r"\delta{v}_{s}", 3],
                ["Speaker orientation error", "delta_q_s", r"\delta{\theta}_{s}", 3],
            ],
        ],
        [
            "true state",
            "x_hat",
            r"\hat{x}",
            [
                ["IMU position true", "p_hat", r"\hat{p}", 3],
                ["IMU velocity true", "v_hat", r"\hat{v}", 3],
                ["IMU orientation true", "q_hat", r"\hat{q}", 4],
                ["IMU accel bias true", "a_hat_b", r"\hat{a}_{b}", 3],
                ["IMU gyro bias true", "omega_hat_b", r"\hat{\omega}_{b}", 3],
                ["Speaker position true", "p_hat_s", r"\hat{p}_{s}", 3],
                ["Speaker velocity true", "v_hat_s", r"\hat{v}_{s}", 3],
                ["Speaker orientation true", "q_hat_s", r"\hat{q}_{s}", 4],
            ],
        ],
        [
            "Process noise",
            "delta_x_n",
            r"\delta{x}_{n}",
            [
                # fmt: off
                ["IMU velocity error noise", "delta_v_n", r"\delta{v}_{n}", 3],
                ["IMU orientation error noise", "delta_q_n", r"\delta{\theta}_{n}", 3],
                ["IMU accel bias error noise", "delta_a_b_n", r"\delta{a}_{bn}", 3],
                ["IMU gyro bias error noise", "delta_omega_b_n", r"\delta{\omega}_{bn}", 3],
                ["Speaker velocity error noise", "delta_v_s_n", r"\delta{v}_{sn}", 3],
                ["Speaker orientation error noise", "delta_q_s_n", r"\delta{\theta}_{sn}", 3],
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
            "Predicted Maesurements",
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


print(esekf_vectors)
