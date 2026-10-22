"""
ESEKF Schema Definition Script
------------------------------
Defines the structure, symbols, code representation, formulas, units,
and descriptions for all vectors and components in the ESEKF filter system.
"""


def get_schema_element(target_key, target_value, node=ESEKF_SCHEMA):
    """Recursively search the schema tree for an element matching target_key == target_value."""
    if isinstance(node, dict):
        if node.get(target_key) == target_value:
            return node
        for val in node.values():
            result = get_schema_element(target_key, target_value, val)
            if result:
                return result
    elif isinstance(node, list):
        for item in node:
            result = get_schema_element(target_key, target_value, item)
            if result:
                return result
    return None


ESEKF_SCHEMA = {
    "nominal state": {
        "name": "nominal state",
        "latex symbol": r"x",
        "code variable": "x",
        "definition": r"x = [x_{IMU}^T, x_{s}^T]^T",
        "description": "Large non-linear nominal state maintained by the filter.",
        "components": [
            {
                "name": "IMU (nominal state)",
                "latex symbol": r"x_{IMU}",
                "code variable": "x_IMU",
                "components": [
                    {
                        "name": "IMU position (nominal state)",
                        "latex symbol": r"p",
                        "code variable": "p",
                        "dimensionality": 3,
                        "unit": "m",
                        "components": [r"p_x", r"p_y", r"p_z"],
                    },
                    {
                        "name": "IMU velocity (nominal state)",
                        "latex symbol": r"v",
                        "code variable": "v",
                        "dimensionality": 3,
                        "unit": "m/s",
                        "components": [r"v_x", r"v_y", r"v_z"],
                    },
                    {
                        "name": "IMU orientation (nominal state)",
                        "latex symbol": r"q",
                        "code variable": "q",
                        "dimensionality": 4,
                        "unit": "unit quaternion",
                        "components": [r"q_w", r"q_x", r"q_y", r"q_z"],
                    },
                    {
                        "name": "IMU accelerometer bias (nominal state)",
                        "latex symbol": r"a_b",
                        "code variable": "a_b",
                        "dimensionality": 3,
                        "unit": "m/s^2",
                        "components": [r"a_{b\,x}", r"a_{b\,y}", r"a_{b\,z}"],
                    },
                    {
                        "name": "IMU gyroscope bias (nominal state)",
                        "latex symbol": r"\omega_b",
                        "code variable": "omega_b",
                        "dimensionality": 3,
                        "unit": "rad/s",
                        "components": [
                            r"\omega_{b\,x}",
                            r"\omega_{b\,y}",
                            r"\omega_{b\,z}",
                        ],
                    },
                ],
            },
            {
                "name": "Speaker (nominal state)",
                "latex symbol": r"x_{s}",
                "code variable": "x_s",
                "components": [
                    {
                        "name": "Speaker position (nominal state)",
                        "latex symbol": r"p_s",
                        "code variable": "p_s",
                        "dimensionality": 3,
                        "unit": "m",
                        "components": [r"p_{s\,x}", r"p_{s\,y}", r"p_{s\,z}"],
                    },
                    {
                        "name": "Speaker velocity (nominal state)",
                        "latex symbol": r"v_s",
                        "code variable": "v_s",
                        "dimensionality": 3,
                        "unit": "m/s",
                        "components": [r"v_{s\,x}", r"v_{s\,y}", r"v_{s\,z}"],
                    },
                    {
                        "name": "Speaker orientation (nominal state)",
                        "latex symbol": r"q_s",
                        "code variable": "q_s",
                        "dimensionality": 4,
                        "unit": "unit quaternion",
                        "components": [
                            r"q_{s\,w}",
                            r"q_{s\,x}",
                            r"q_{s\,y}",
                            r"q_{s\,z}",
                        ],
                    },
                ],
            },
        ],
    },
    "error state": {
        "name": "error state",
        "latex symbol": r"\delta{x}",
        "code variable": "delta_x",
        "definition": r"\delta{x} = [\delta{x}_{IMU}^T, \delta{x}_{s}^T]^T",
        "description": "Minimal linear state representing errors, reset to 0 after injection.",
        "components": [
            {
                "name": "IMU (error state)",
                "latex symbol": r"\delta{x}_{IMU}",
                "code variable": "delta_x_IMU",
                "components": [
                    {
                        "name": "IMU position (error state)",
                        "latex symbol": r"\delta{p}",
                        "code variable": "delta_p",
                        "dimensionality": 3,
                        "unit": "m",
                        "components": [r"\delta{p}_x", r"\delta{p}_y", r"\delta{p}_z"],
                    },
                    {
                        "name": "IMU velocity (error state)",
                        "latex symbol": r"\delta{v}",
                        "code variable": "delta_v",
                        "dimensionality": 3,
                        "unit": "m/s",
                        "components": [r"\delta{v}_x", r"\delta{v}_y", r"\delta{v}_z"],
                    },
                    {
                        "name": "IMU orientation (error state)",
                        "latex symbol": r"\delta{\theta}",
                        "code variable": "delta_theta",
                        "dimensionality": 3,
                        "unit": "rad",
                        "components": [
                            r"\delta{\theta}_x",
                            r"\delta{\theta}_y",
                            r"\delta{\theta}_z",
                        ],
                    },
                    {
                        "name": "IMU accelerometer bias (error state)",
                        "latex symbol": r"\delta{a}_b",
                        "code variable": "delta_a_b",
                        "dimensionality": 3,
                        "unit": "m/s^2",
                        "components": [
                            r"\delta{a}_{b\,x}",
                            r"\delta{a}_{b\,y}",
                            r"\delta{a}_{b\,z}",
                        ],
                    },
                    {
                        "name": "IMU gyroscope bias (error state)",
                        "latex symbol": r"\delta{\omega}_b",
                        "code variable": "delta_omega_b",
                        "dimensionality": 3,
                        "unit": "rad/s",
                        "components": [
                            r"\delta{\omega}_{b\,x}",
                            r"\delta{\omega}_{b\,y}",
                            r"\delta{\omega}_{b\,z}",
                        ],
                    },
                ],
            },
            {
                "name": "Speaker (error state)",
                "latex symbol": r"\delta{x}_s",
                "code variable": "delta_x_s",
                "components": [
                    {
                        "name": "Speaker position (error state)",
                        "latex symbol": r"\delta{p}_s",
                        "code variable": "delta_p_s",
                        "dimensionality": 3,
                        "unit": "m",
                        "components": [
                            r"\delta{p}_{s\,x}",
                            r"\delta{p}_{s\,y}",
                            r"\delta{p}_{s\,z}",
                        ],
                    },
                    {
                        "name": "Speaker velocity (error state)",
                        "latex symbol": r"\delta{v}_s",
                        "code variable": "delta_v_s",
                        "dimensionality": 3,
                        "unit": "m/s",
                        "components": [
                            r"\delta{v}_{s\,x}",
                            r"\delta{v}_{s\,y}",
                            r"\delta{v}_{s\,z}",
                        ],
                    },
                    {
                        "name": "Speaker orientation error (angle-axis)",
                        "latex symbol": r"\delta{\theta}_s",
                        "code variable": "delta_theta_s",
                        "dimensionality": 3,
                        "unit": "rad",
                        "components": [
                            r"\delta{\theta}_{s\,x}",
                            r"\delta{\theta}_{s\,y}",
                            r"\delta{\theta}_{s\,z}",
                        ],
                    },
                ],
            },
        ],
    },
    "true state": {
        "name": "true state",
        "latex symbol": r"\hat{x}",
        "code variable": "x_hat",
        "definition": r"\hat{x} = x \oplus \delta{x}",
        "description": "Concrete composition of nominal and error states (used for Jacobian linearization).",
        "components": [
            {
                "name": "IMU (true state)",
                "latex symbol": r"\hat{x}_{IMU}",
                "code variable": "x_hat_IMU",
                "components": [
                    {
                        "name": "IMU position (true state)",
                        "latex symbol": r"\hat{p}",
                        "code variable": "p_hat",
                        "definition": r"\hat{p} = p + \delta{p}",
                        "dimensionality": 3,
                        "unit": "m",
                        "components": [r"\hat{p}_x", r"\hat{p}_y", r"\hat{p}_z"],
                    },
                    {
                        "name": "IMU velocity (true state)",
                        "latex symbol": r"\hat{v}",
                        "code variable": "v_hat",
                        "definition": r"\hat{v} = v + \delta{v}",
                        "dimensionality": 3,
                        "unit": "m/s",
                        "components": [r"\hat{v}_x", r"\hat{v}_y", r"\hat{v}_z"],
                    },
                    {
                        "name": "IMU orientation (true state)",
                        "latex symbol": r"\hat{q}",
                        "code variable": "q_hat",
                        "definition": r"\hat{q} = q \otimes q\{\delta{\theta}\}",
                        "dimensionality": 4,
                        "unit": "unit quaternion",
                        "components": [
                            r"\hat{q}_w",
                            r"\hat{q}_x",
                            r"\hat{q}_y",
                            r"\hat{q}_z",
                        ],
                    },
                    {
                        "name": "IMU accelerometer bias (true state)",
                        "latex symbol": r"\hat{a}_b",
                        "code variable": "a_b_hat",
                        "definition": r"\hat{a}_b = a_b + \delta{a}_b",
                        "dimensionality": 3,
                        "unit": "m/s^2",
                        "components": [
                            r"\hat{a}_{b\,x}",
                            r"\hat{a}_{b\,y}",
                            r"\hat{a}_{b\,z}",
                        ],
                    },
                    {
                        "name": "IMU gyroscope bias (true state)",
                        "latex symbol": r"\hat{\omega}_b",
                        "code variable": "omega_b_hat",
                        "definition": r"\hat{\omega}_b = \omega_b + \delta{\omega}_b",
                        "dimensionality": 3,
                        "unit": "rad/s",
                        "components": [
                            r"\hat{\omega}_{b\,x}",
                            r"\hat{\omega}_{b\,y}",
                            r"\hat{\omega}_{b\,z}",
                        ],
                    },
                ],
            },
            {
                "name": "Speaker (true state)",
                "latex symbol": r"\hat{x}_s",
                "code variable": "x_hat_s",
                "components": [
                    {
                        "name": "Speaker position (true state)",
                        "latex symbol": r"\hat{p}_s",
                        "code variable": "p_hat_s",
                        "definition": r"\hat{p}_s = p_s + \delta{p}_s",
                        "dimensionality": 3,
                        "unit": "m",
                        "components": [
                            r"\hat{p}_{s\,x}",
                            r"\hat{p}_{s\,y}",
                            r"\hat{p}_{s\,z}",
                        ],
                    },
                    {
                        "name": "Speaker velocity (true state)",
                        "latex symbol": r"\hat{v}_s",
                        "code variable": "v_hat_s",
                        "definition": r"\hat{v}_s = v_s + \delta{v}_s",
                        "dimensionality": 3,
                        "unit": "m/s",
                        "components": [
                            r"\hat{v}_{s\,x}",
                            r"\hat{v}_{s\,y}",
                            r"\hat{v}_{s\,z}",
                        ],
                    },
                    {
                        "name": "Speaker orientation (true state)",
                        "latex symbol": r"\hat{q}_s",
                        "code variable": "q_hat_s",
                        "definition": r"\hat{q}_s = q_s \otimes q\{\delta{\theta}_s\}",
                        "dimensionality": 4,
                        "unit": "unit quaternion",
                        "components": [
                            r"\hat{q}_{s\,w}",
                            r"\hat{q}_{s\,x}",
                            r"\hat{q}_{s\,y}",
                            r"\hat{q}_{s\,z}",
                        ],
                    },
                ],
            },
        ],
    },
    "input": {
        "name": "IMU input",
        "latex symbol": r"\tilde{u}",
        "code variable": "u_tilde",
        "description": "Processed system input used in kinematic propagation.",
        "components": [
            {
                "name": "accelerometer",
                "latex symbol": r"\tilde{a}",
                "code variable": "a_tilde",
                "dimensionality": 3,
                "unit": "m/s^2",
                "components": [r"\tilde{a}_x", r"\tilde{a}_y", r"\tilde{a}_z"],
            },
            {
                "name": "gyroscope",
                "latex symbol": r"\tilde{\omega}",
                "code variable": "omega_tilde",
                "dimensionality": 3,
                "unit": "rad/s",
                "components": [
                    r"\tilde{\omega}_x",
                    r"\tilde{\omega}_y",
                    r"\tilde{\omega}_z",
                ],
            },
        ],
    },
    "raw sensor input": {
        "name": "Raw Sensor Input & Detector Measurements",
        "latex symbol": r"\tilde{u}_{raw}",
        "code variable": "u_tile_raw",
        "description": "Direct hardware readings and external tracking observations before filtering/conversion.",
        "components": [
            {
                "name": "raw accelerometer reading",
                "latex symbol": r"\tilde{a}_{raw}",
                "code variable": "accel_raw",
                "dimensionality": 3,
                "unit": "g",
                "definition": r"\tilde{a} = \tilde{a}_{raw} \cdot 9.80665",
                "components": [
                    r"\tilde{a}_{raw\,x}",
                    r"\tilde{a}_{raw\,y}",
                    r"\tilde{a}_{raw\,z}",
                ],
            },
            {
                "name": "raw gyroscope reading",
                "latex symbol": r"\tilde{\omega}_{raw}",
                "code variable": "gyro_raw",
                "dimensionality": 3,
                "unit": "deg/s",
                "definition": r"\tilde{\omega} = \tilde{\omega}_{raw} \cdot \frac{\pi}{180}",
                "components": [
                    r"\tilde{\omega}_{raw\,x}",
                    r"\tilde{\omega}_{raw\,y}",
                    r"\tilde{\omega}_{raw\,z}",
                ],
            },
            {
                "name": "raw face detection cartesian position",
                "latex symbol": r"\tilde{p}_{s}",
                "code variable": "p_s_tilde",
                "dimensionality": 3,
                "unit": "m",
                "description": "Raw 3D cartesian coordinates detected for the faces in the scene.",
                "components": [
                    r"\tilde{p}_{s\,x}",
                    r"\tilde{p}_{s\,y}",
                    r"\tilde{p}_{s\,z}",
                ],
            },
            {
                "name": "raw face detection orientation quaternion",
                "latex symbol": r"\tilde{q}_{s}",
                "code variable": "q_s_tilde",
                "dimensionality": 4,
                "unit": "unit quaternion",
                "description": "Raw unit quaternion representing face orientation.",
                "components": [
                    r"\tilde{q}_{s\,w}",
                    r"\tilde{q}_{s\,x}",
                    r"\tilde{q}_{s\,y}",
                    r"\tilde{q}_{s\,z}",
                ],
            },
            {
                "name": "fake global origin reset measurement",
                "latex symbol": r"\tilde{p}",
                "code variable": "p_tilde",
                "dimensionality": 3,
                "unit": "m",
                "components": [
                    r"\tilde{p}_{x}",
                    r"\tilde{p}_{z}",
                    r"\tilde{p}_{z}",
                ],
            },
            {
                "name": "fake optitrack yaw angle measurement",
                "latex symbol": r"\psi_{opt}",
                "code variable": "psi_optitrack",
                "dimensionality": 1,
                "unit": "rad",
                "definition": r"\psi = \text{atan2}\left(2(q_{opt\,w} q_{opt\,z} - q_{opt\,x} q_{opt\,y}), 1 - 2(q_{opt\,x}^2 + q_{opt\,z}^2)\right)",
                "description": "Fake yaw extracted from the external OptiTrack system's IMU orientation output to bound global heading drift",
                "components": [r"\psi_{optitrack}"],
            },
        ],
    },
    "process noise": {
        "name": "process noise",
        "latex symbol": r"\delta{x}_n",
        "code variable": "delta_x_n",
        "description": "Additive process noise perturbations per step.",
        "components": [
            {
                "name": "IMU process noise",
                "latex symbol": r"\delta{x}_{IMUn}",
                "code variable": "delta_x_n_IMU",
                "components": [
                    {
                        "name": "velocity noise",
                        "latex symbol": r"\delta{v}_n",
                        "code variable": "delta_v_n",
                        "dimensionality": 3,
                        "unit": "m/s",
                        "components": [
                            r"\delta{v}_{n\,x}",
                            r"\delta{v}_{n\,y}",
                            r"\delta{v}_{n\,z}",
                        ],
                    },
                    {
                        "name": "orientation noise",
                        "latex symbol": r"\delta{\theta}_n",
                        "code variable": "delta_theta_n",
                        "dimensionality": 3,
                        "unit": "rad",
                        "components": [
                            r"\delta{\theta}_{n\,x}",
                            r"\delta{\theta}_{n\,y}",
                            r"\delta{\theta}_{n\,z}",
                        ],
                    },
                    {
                        "name": "accelerometer bias random walk",
                        "latex symbol": r"\delta{a}_n",
                        "code variable": "delta_a_n",
                        "dimensionality": 3,
                        "unit": "m/s^2",
                        "components": [
                            r"\delta{a}_{n\,x}",
                            r"\delta{a}_{n\,y}",
                            r"\delta{a}_{n\,z}",
                        ],
                    },
                    {
                        "name": "gyroscope bias random walk",
                        "latex symbol": r"\delta{\omega}_n",
                        "code variable": "delta_omega_n",
                        "dimensionality": 3,
                        "unit": "rad/s",
                        "components": [
                            r"\delta{\omega}_{n\,x}",
                            r"\delta{\omega}_{n\,y}",
                            r"\delta{\omega}_{n\,z}",
                        ],
                    },
                ],
            },
            {
                "name": "Speaker process noise",
                "latex symbol": r"\delta{x}_{sn}",
                "code variable": "delta_x_n_s",
                "components": [
                    {
                        "name": "speaker velocity noise",
                        "latex symbol": r"\delta{v}_{sn}",
                        "code variable": "delta_v_s_n",
                        "dimensionality": 3,
                        "unit": "m/s",
                        "components": [
                            r"\delta{v}_{sn\,x}",
                            r"\delta{v}_{sn\,y}",
                            r"\delta{v}_{sn\,z}",
                        ],
                    },
                    {
                        "name": "speaker orientation noise",
                        "latex symbol": r"\delta{\theta}_{sn}",
                        "code variable": "delta_theta_s_n",
                        "dimensionality": 3,
                        "unit": "rad",
                        "components": [
                            r"\delta{\theta}_{sn\,x}",
                            r"\delta{\theta}_{sn\,y}",
                            r"\delta{\theta}_{sn\,z}",
                        ],
                    },
                ],
            },
        ],
    },
    "measurements": {
        "name": "measurement predictions",
        "latex symbol": r"h",
        "code variable": "h",
        "components": [
            {
                "name": "IMU gravity vector measurement",
                "latex symbol": r"g^h",
                "code variable": "g_h",
                "definition": r"g^h = R(q)^T g + a_b",
                "dimensionality": 3,
                "unit": "m/s^2",
            },
            {
                "name": "Face position relative to IMU",
                "latex symbol": r"p_s^h",
                "code variable": "p_s_h",
                "definition": r"p_s^b = R^{\Delta q_{I \to c}}(R^q (p_s - p) - \Delta p_{I \to c})",
                "dimensionality": 3,
                "unit": "m",
            },
            {
                "name": "Face Detection relative speaker orientation",
                "latex symbol": r"q_s^{h}",
                "code variable": "q_s_h",
                "definition": r"q_s^{hc} = (q \otimes \Delta q_{I \to c})^{-1} \otimes q_s",
                "dimensionality": 4,
                "unit": "unit quaternion",
            },
            {
                "name": "Fake IMU position measurement",
                "latex symbol": r"p^h",
                "code variable": "p_h",
                "definition": r"p^h = p",
                "dimensionality": 3,
                "unit": "m",
            },
            {
                "name": "Fake OptiTrack yaw angle measurement",
                "latex symbol": r"\psi^h",
                "code variable": "psi_h",
                "definition": r"\psi^h = \text{atan2}(2(q_w q_z - q_x q_y), 1 - 2(q_x^2 + q_z^2))",
                "dimensionality": 1,
                "unit": "rad",
            },
        ],
    },
    "extra variables": {
        "name": "extra fixed parameters & constants",
        "components": [
            {
                "name": "sample step delta t",
                "latex symbol": r"\Delta t",
                "code variable": "dt",
                "dimensionality": 1,
                "unit": "s",
            },
            {
                "name": "gravity constant",
                "latex symbol": r"g",
                "code variable": "gravity_contstant",
                "dimensionality": 1,
                "definition": "g = 9.80665",
                "unit": "m/s^2",
            },
            {
                "name": "gravity vector",
                "latex symbol": r"\vec{g}",
                "code variable": "g_vec",
                "dimensionality": 3,
                "definition": "[0, 0, gravity_constant]",
                "description": "the reading of a perfectly level accelerometer, pointing up with 1g",
            },
            {
                "name": "IMU to camera spatial translation",
                "latex symbol": r"\Delta p_{I \to c}",
                "code variable": "IMU_cam_translation",
                "dimensionality": 3,
                "unit": "m",
            },
            {
                "name": "IMU to camera rotation quaternion",
                "latex symbol": r"\Delta q_{I \to c}",
                "code variable": "IMU_cam_rotation",
                "dimensionality": 4,
                "unit": "unit quaternion",
            },
        ],
    },
}
