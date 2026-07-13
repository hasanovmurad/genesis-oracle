import json
import re
from pathlib import Path


def extract_parameters_from_text(text: str) -> dict:
    """
    Extracts simple simulation parameters from scientific text.

    Args:
        text: Scientific abstract or paper text.

    Returns:
        Dictionary containing extracted parameters.
    """

    parameters = {
        "thermal_conductivity": None,
        "doi": None,
    }

    doi = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)

    if doi:
        parameters["doi"] = doi.group(0)

    if "thermal conductivity" in text.lower():
        parameters["thermal_conductivity"] = "mentioned"

    return parameters


if __name__ == "__main__":

    sample_text = """
    DOI: 10.1016/j.nucengdes.2019.04.023

    Thermal conductivity is an important material property for
    advanced fission reactor simulations.
    """

    result = extract_parameters_from_text(sample_text)

    Path("data").mkdir(exist_ok=True)

    with open("data/simulation_parameters.json", "w") as f:
        json.dump(result, f, indent=4)

    print(json.dumps(result, indent=4))