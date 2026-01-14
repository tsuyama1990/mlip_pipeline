from mace.calculators import mace_mp


def get_calculator(name: str):
    if name == "mace_mp":
        return mace_mp(model="medium", device="cpu")
    unknown_calculator_error = f"Unknown calculator: {name}"
    raise ValueError(unknown_calculator_error)
