# every entry can have the following attributes
# - name/title:
# - latex symbol: for printing in notebook, uses raw string type (e.g. r"\omega_x")
# - code variable: when defined in code, the entry will be named like this (e.g. "omega_x")
# - definition: is it defined by some formula (e.g true state = (nominal state) \oplus error state)
# - description: if there is more context needed, this can be like a text section to add more info
# - dimensionality: if it is fixed. I could also only allow this on the lowest child objects and then infer it for the higher ones
# - unit: (e.g. m, m/s, g, deg, radian)

ESEKF_SCHEMA = {
    "nominal state": {
        "name": "nominal state",
        "latex symbol": r"x",
        "code variable": "x",
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
                        "components": [r"p_x", r"p_y", r"p_z"],
                    },
                    {"name": "IMU velocity (nominal state)"},
                    {"name": "IMU orientation (nominal state)"},
                    {"name": "IMU accelerometer bias (nominal state)"},
                    {"name": "IMU gyroscope bias (nominal state)"},
                ],
            },
            {
                "name": "Speaker (nominal state)",
                "latex symbol": r"x_{s}",
                "code variable": "x_s",
                "components": [{"name": "Speaker position (nominal state)"}],
            },
        ],
    },
    "error state": {},
    "true state": {"name": "true state", "latex symbol": r"\hat{x}"},
    "input": {
        "name": "IMU input",
        "latex symbol": r"\tilde{u}",
        "code variable": "u_tilde",
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
                "unit": "deg/s",
                "components": [
                    r"\tilde{\omega}_x",
                    r"\tilde{\omega}_y",
                    r"\tilde{\omega}_z",
                ],
            },
        ],
    },
    "measurement": {},
    "extra components": {},
}
