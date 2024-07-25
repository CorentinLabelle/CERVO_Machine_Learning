import scipy


def save(variable: dict, filename: str) -> None:
    if not isinstance(variable, dict):
        variable = {"variable": variable}
    scipy.io.savemat(filename, variable)
