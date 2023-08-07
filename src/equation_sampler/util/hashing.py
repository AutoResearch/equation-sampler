import json
import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources


def load_adjusted_probabilities(hash_id):
    pkg = importlib_resources.files("equation_sampler")
    hash_file = pkg / "data" / "adjusted_probabilities.json"
    with open(hash_file, "r") as file:
        data = json.load(file)
    if hash_id in data.keys():
        datum = data[hash_id]
        if (
            "adjusted_probabilities_functions" in datum.keys()
            and "adjusted_probabilities_operators" in datum.keys()
        ):
            return (
                datum["adjusted_probabilities_functions"],
                datum["adjusted_probabilities_operators"],
            )
    return None, None


def store_adjusted_probabilities(
    hash_id, adjusted_probabilities_functions, adjusted_probabilities_operators
):
    pkg = importlib_resources.files("equation_sampler")
    hash_file = pkg / "data" / "adjusted_probabilities.json"
    with open(hash_file, "r") as file:
        data = json.load(file)
    data[hash_id] = {
        "adjusted_probabilities_functions": adjusted_probabilities_functions,
        "adjusted_probabilities_operators": adjusted_probabilities_operators,
    }
    with open(hash_file, "w") as file:
        json.dump(data, file)
