import itertools
import warnings
from typing import List

import numpy as np
from sympy import I, simplify, symbols, sympify

from .equation_tree import EquationTree, is_binary_tree, rooted_tree_iterator
from .measure import get_frequencies
from .util.hashing import load_adjusted_probabilities, store_adjusted_probabilities
from .util.unary_minus_to_binary import unary_minus_to_binary

padding = "<PAD>"

BURN_SAMPLE_SIZE = 20000
LEARNING_RATE = 0.03
SAVE_TO_HASH = False  # Set to true if updating hashes

PRINT_MOD = 1000


def prefix_to_infix(prefix, function_space, operation_space):
    stack = []
    for i in range(len(prefix) - 1, -1, -1):
        if prefix[i] in function_space:
            # symbol in unary operator
            stack.append(prefix[i] + "(" + stack.pop() + ")")
        elif prefix[i] in operation_space:
            # symbol is binary operator
            str = "(" + stack.pop() + prefix[i] + stack.pop() + ")"
            stack.append(str)
        else:
            # symbol is operand
            stack.append(prefix[i])
    return stack.pop()


# Function to get the priority of operators and functions
def get_priority(c):
    if c == "-" or c == "+":
        return 1
    elif c == "*" or c == "/":
        return 2
    elif c == "^":
        return 3
    elif c[0].isalpha():
        return 4
    return 0


# Function to convert the infix expression to postfix
def infix_to_postfix(infix, function_space, operation_space):
    infix = "(" + infix + ")"
    n = len(infix)
    char_stack = []
    output = []
    i = 0
    while i < n:
        # Check if the character is alphabet or digit
        if infix[i].isdigit() and infix[i + 1] == "_":
            output.append(infix[i : i + 3][::-1])
            i += 2
        elif infix[i].isdigit():
            output.append(infix[i])

        # If the character is '(' push it in the stack
        elif infix[i] == "(":
            char_stack.append(infix[i])

        # If the character is ')' pop from the stack
        elif infix[i] == ")":
            while char_stack[-1] != "(":
                output.append(char_stack.pop())
            char_stack.pop()
        # Found an operator
        else:
            if (
                char_stack[-1] in function_space
                or char_stack[-1] in operation_space
                or char_stack[-1] in [")", "("]
            ):
                if infix[i] == "^":
                    while get_priority(infix[i]) <= get_priority(char_stack[-1]):
                        output.append(char_stack.pop())
                    char_stack.append(infix[i])
                elif infix[i].isalpha():
                    fct = ""
                    while infix[i].isalpha() and i < n - 1:
                        fct += infix[i]
                        i += 1
                    i -= 1
                    while get_priority(fct) < get_priority(char_stack[-1]):
                        output.append(char_stack.pop())
                    char_stack.append(fct[::-1])
                else:  # + - * / ( )
                    while get_priority(infix[i]) < get_priority(char_stack[-1]):
                        output.append(char_stack.pop())
                    char_stack.append(infix[i])

        i += 1

    while len(char_stack) != 0:
        output.append(char_stack.pop())
    return output


# Function to convert infix expression to prefix
def infix_to_prefix(infix, function_space, operation_space):
    n = len(infix)

    infix = list(infix[::-1].lower())

    for i in range(n):
        if infix[i] == "(":
            infix[i] = ")"
        elif infix[i] == ")":
            infix[i] = "("

    infix = "".join(infix)
    prefix = infix_to_postfix(infix, function_space, operation_space)
    prefix = prefix[::-1]

    return prefix


def sample_equations(
    num_samples: int,
    max_depth: int,
    max_num_variables: int,
    max_num_constants: int,
    function_space: list = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"],
    operator_space: list = ["+", "-", "*", "/", "^"],
    function_priors: dict = {},
    operator_priors: dict = {},
    force_full_domain: bool = False,
    with_replacement: bool = True,
    fix_num_variables_to_max: bool = False,
    include_zero_as_constant=False,
    min_input_value: float = -1,
    max_input_value: float = 1,
    min_constant_value: float = -1,
    max_constant_value: float = 1,
    num_input_points: int = 100,
    num_constant_points: int = 100,
    num_evaluation_samples: int = 100,
    max_iter: int = 1000000,
    num_burns: int = 0,
    require_simplify: bool = True,
    is_real_domain: bool = True,
    verbose: bool = False,
):
    """
    Generate data for the equation generator.

    Arguments:
        num_samples: Number of samples to generate.
        max_depth: Maximum depth of the equation tree.
        max_num_variables: Number of variables in the equation tree.
        max_num_constants: Maximum number of constants in the equation tree.
        function_space: List of functions to use in the equation tree.
        operator_space: List of operations to use in the equation tree.
        function_priors: Dict with priors for the functions.
        operator_priors: Dict with priors for the operators.
        force_full_domain: If true only equations that are defined on full R are sampled.
        with_replacement: Whether to sample with replacement.
        fix_num_variables_to_max: Whether to fix the number of variables.
        include_zero_as_constant: Whether to include zero as a constant.
        min_input_value: Minimum value of the input variables.
        max_input_value: Maximum value of the input variables.
        min_constant_value: Minimum value of the constants.
        max_constant_value: Maximum value of the constants.
        num_input_points: Number of points to sample for each input variable and constant.
        num_constant_points: Number of points to sample for each constant.
        num_evaluation_samples: ...,
        max_iter: ...,
        num_burns: Number of times before sampling to adjust the probabilities
        require_simplify: Defines if the equations are simplified
        is_real_domain: Defines if the variables and constants are real or complex numbers
        verbose: Defines if additional output is generated
    """
    local_dict = locals()
    del local_dict["num_burns"]
    del local_dict["verbose"]
    del local_dict["num_samples"]
    del local_dict["num_evaluation_samples"]
    del local_dict["operator_priors"]
    del local_dict["function_priors"]

    if max_depth < 3:
        raise ValueError("max_depth must be at least 3")
    if force_full_domain:
        warnings.warn(
            "Forcing equations to be defined on full domain may lead "
            "to larger discrepancies between priors and frequencies in sample."
        )

    tokenized_equation_list: List[List[str]] = list()
    sympy_equation_list: List[str] = list()
    evaluation_list: List[float] = list()
    max_equation_elements = 0

    # Generate Feature Space
    feature_space = [f"x_{i + 1}" for i in range(max_num_variables)] + [
        f"c_{i + 1}" for i in range(max_num_constants)
    ]

    if include_zero_as_constant:
        feature_space.append("0")

    # Get the configuration
    config = {
        "max_num_variables": max_num_variables,
        "max_num_constants": max_num_constants,
        "feature_space": feature_space,
        "function_space": function_space,
        "operator_space": operator_space,
        "force_full_domain": force_full_domain,
        "with_replacement": with_replacement,
        "fix_num_variables_to_max": fix_num_variables_to_max,
        "include_zero_as_constant": include_zero_as_constant,
        "min_input_value": min_input_value,
        "max_input_value": max_input_value,
        "min_constant_value": min_constant_value,
        "max_constant_value": max_constant_value,
        "num_input_points": num_input_points,
        "num_constant_points": num_constant_points,
        "num_evaluation_samples": num_evaluation_samples,
        "max_iter": max_iter,
        "require_simplify": require_simplify,
        "is_real_domain": is_real_domain,
        "verbose": verbose,
    }

    # Generate all possible trees
    tree_structures = [
        tree.copy()
        for depth in range(3, max_depth + 1)
        for tree in rooted_tree_iterator(depth)
    ]

    # set target priors
    target_probabilities_functions = _set_priors(function_priors, function_space)
    target_probabilities_operators = _set_priors(operator_priors, operator_space)
    hash_id_backup = str(local_dict)
    if function_priors != {} or operator_priors != {}:
        local_dict["functon_priors"] = target_probabilities_functions
        local_dict["operator_priors"] = target_probabilities_operators
        if num_burns > 0:
            warnings.warn(
                f"Storing non default priors. "
                f"Please make sure BURN_SAMPLE_SIZE {BURN_SAMPLE_SIZE} is large enough"
            )
    hash_id = str(local_dict)

    # load adjusted probabilities from hash
    function_probabilities, operator_probabilities = load_adjusted_probabilities(
        hash_id
    )

    if function_probabilities is None:
        function_probabilities_, operator_probabilities_ = load_adjusted_probabilities(
            hash_id_backup
        )
        if function_probabilities_ is not None:
            warnings.warn(
                "Load backup probabilities from default priors and adjusting them. "
                "This may lead to discrepancies between priors and sampled frequencies."
            )
            _f = {
                key: target_probabilities_functions[key]
                * len(target_probabilities_functions)
                for key in target_probabilities_functions.keys()
            }
            _o = {
                key: target_probabilities_operators[key]
                * len(target_probabilities_operators)
                for key in target_probabilities_operators.keys()
            }
            function_probabilities = {
                key: function_probabilities_[key] * _f[key]
                for key in target_probabilities_functions.keys()
            }
            operator_probabilities = {
                key: operator_probabilities_[key] * _o[key]
                for key in target_probabilities_operators.keys()
            }
        else:
            function_probabilities = target_probabilities_functions
            operator_probabilities = target_probabilities_operators
            if num_burns <= 0:
                warnings.warn(
                    "Using raw priors without burn. This may lead to discrepancies "
                    "between priors and sampled frequencies."
                )
            else:
                warnings.warn(
                    "Using raw priors. This may lead to discrepancies "
                    "between priors and sampled frequencies."
                )
    # burn (sample and adjust on samples)
    for burn in range(num_burns):
        tokenized_equation_list_burn: List[List[str]] = list()
        # sample an equation
        for i in range(BURN_SAMPLE_SIZE):
            tokenized_equation_burn, _, __ = _sample_full_equation(
                tree_structures,
                tokenized_equation_list_burn,
                function_probabilities,
                operator_probabilities,
                **config,
            )
            tokenized_equation_list_burn.append(tokenized_equation_burn)

            nr_burns = burn * BURN_SAMPLE_SIZE + i + 1

            # print progress
            if nr_burns % PRINT_MOD == 0:
                print(f"{nr_burns} equations burned")
        function_frequencies, operator_frequencies = get_frequencies(
            tokenized_equation_list_burn, **config
        )
        for key in function_space:
            diff = target_probabilities_functions[key] - function_frequencies[key]
            function_probabilities[key] += LEARNING_RATE * diff
            if function_probabilities[key] <= 0:
                function_probabilities[key] = 0
        for key in operator_space:
            diff = target_probabilities_operators[key] - operator_frequencies[key]
            operator_probabilities[key] += LEARNING_RATE * diff
            if operator_probabilities[key] <= 0:
                operator_probabilities[key] = 0
        function_probabilities = _normalize_priors(function_probabilities)
        operator_probabilities = _normalize_priors(operator_probabilities)
        if SAVE_TO_HASH:
            store_adjusted_probabilities(
                hash_id, function_probabilities, operator_probabilities
            )

    # sample
    for sample in range(num_samples):
        # sample an equation
        tokenized_equation, sympy_expression, evaluation = _sample_full_equation(
            tree_structures,
            tokenized_equation_list,
            function_probabilities,
            operator_probabilities,
            **config,
        )

        # add to lists
        tokenized_equation_list.append(tokenized_equation)
        sympy_equation_list.append(sympy_expression)
        evaluation_list.append(evaluation)
        max_equation_elements = max(max_equation_elements, len(tokenized_equation))

        # print progress
        if (sample + 1) % PRINT_MOD == 0:
            print(f"{sample + 1} equations generated")

    # pad the equations and evaluations
    for idx, equation in enumerate(tokenized_equation_list):
        num_equation_elements = len(equation)
        for i in range(max_equation_elements - num_equation_elements):
            tokenized_equation_list[idx].append(padding)
            evaluation_list[idx] = np.append(
                evaluation_list[idx], np.zeros((num_evaluation_samples, 1)), axis=1
            )

    # transpose each evaluation
    # (this is temporary to work with the autoencoder model and may be removed in the future)
    for idx, evaluation in enumerate(evaluation_list):
        evaluation_list[idx] = evaluation.T

    print("all equations generated")
    return {
        "tokenized_equations": tokenized_equation_list,
        "sympy_equations": sympy_equation_list,
        "evaluations": evaluation_list,
    }


def create_crossings(num_inputs, num_constants, **kwargs):
    min_input_value = kwargs.get("min_input_value")
    max_input_value = kwargs.get("max_input_value")
    min_constant_value = kwargs.get("min_constant_value")
    max_constant_value = kwargs.get("max_constant_value")
    num_input_points = kwargs.get("num_input_points")
    num_constant_points = kwargs.get("num_constant_points")
    num_evaluation_samples = kwargs.get("num_evaluation_samples")
    grids = []

    for variable in range(num_inputs):
        # Create an evenly spaced grid for each variable
        grid = np.linspace(min_input_value, max_input_value, num_input_points)
        grids.append(grid)

    for constant in range(num_constants):
        # Create an evenly spaced grid for each constant
        grid = np.linspace(min_constant_value, max_constant_value, num_constant_points)
        grids.append(grid)

    # Generate combinations of variables
    crossings = np.array(list(itertools.product(*grids)))

    # Randomly sample M crossings if the total number of crossings is greater than M
    if len(crossings) > num_evaluation_samples:
        indices = np.random.choice(
            len(crossings), num_evaluation_samples, replace=False
        )
        crossings = crossings[indices]

    return crossings


def get_evaluation(
    crossings,
    tree,
    num_evaluation_samples=100,
    include_zero_as_constant=False,
    max_nodes_for_evaluation=None,
    transpose=False,
):
    evaluation = np.zeros((num_evaluation_samples, len(tree.expr)))

    # get all the x labels
    x_labels = list()
    for element in tree.expr:
        # if element in feature_space:
        if "x_" in element:
            x_labels.append(element)

    # get all the c labels
    c_labels = list()
    for element in tree.expr:
        if "c_" in element:
            c_labels.append(element)

    for idx, crossing in enumerate(crossings):
        eqn_input = dict()
        # adding constants to the feature set
        for key in tree.expr:
            if is_numeric(key):
                eqn_input[key] = float(key)
            elif key == "e":
                eqn_input[key] = 2.71828182846
        if include_zero_as_constant:
            eqn_input["0"] = 0
        for i in range(tree.num_x):
            eqn_input[x_labels[i]] = crossing[i]
        for i in range(tree.num_c):
            eqn_input[c_labels[i]] = crossing[tree.num_x + i]

        evaluation[idx, :] = tree.evaluate(eqn_input)

    if max_nodes_for_evaluation is not None:
        for i in range(max_nodes_for_evaluation - len(tree.expr)):
            evaluation = np.append(
                evaluation, np.zeros((num_evaluation_samples, 1)), axis=1
            )

    if transpose:
        evaluation = evaluation.T

    return evaluation


def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _tree_expr_to_sympy(expr, function_space, operator_space, is_real_domain, verbose):
    """
    Takes a tree expression as input and returns a sympy expression
    """
    current_infix = prefix_to_infix(expr, function_space, operator_space)
    if verbose:
        print("_________")
        print("infix initial", current_infix)
        print("initial tree", expr)
    # create a sympy expression from string
    sympy_expr = sympify(current_infix)
    if sympy_expr.free_symbols:
        symbol_names = [str(symbol) for symbol in sympy_expr.free_symbols]
        real_symbols = symbols(" ".join(symbol_names), real=is_real_domain)
        if not isinstance(real_symbols, list) and not isinstance(real_symbols, tuple):
            real_symbols = [real_symbols]
        subs_dict = {old: new for old, new in zip(symbol_names, real_symbols)}
        sympy_expr = sympy_expr.subs(subs_dict)
    return sympy_expr


def _simplified_tree(
    expr, feature_space, function_space, operator_space, is_real_domain, verbose
):
    """
    Takes a tree expression as input and returns a simplified tree
    """
    current_infix = prefix_to_infix(expr, function_space, operator_space)
    if verbose:
        print("_________")
        print("infix initial", current_infix)
        print("initial tree", expr)

    sympy_expr = _tree_expr_to_sympy(
        expr, function_space, operator_space, is_real_domain, verbose
    )

    simplified_equation = simplify(sympy_expr)

    if I in simplified_equation.free_symbols:
        return None
    simplified_equation = unary_minus_to_binary(
        str(simplified_equation), operator_space
    )

    simplified_equation = simplified_equation.replace(" ", "")
    simplified_equation = simplified_equation.replace("**", "^")

    prefix = infix_to_prefix(simplified_equation, function_space, operator_space)
    if verbose:
        print("prefix", simplified_equation)
        print("prefix tree", prefix)
    if len(prefix) > len(expr):
        prefix = expr
    if "re" in prefix:
        prefix.remove("re")
    if "zoo" in prefix or "oo" in prefix:
        return None
    tree = EquationTree([], feature_space, function_space, operator_space)
    tree.instantiate_from_prefix_notation(prefix)
    return tree


def _set_priors(priors=None, space=None):
    """
    Set the priors

    Examples:
        >>> default_priors = _set_priors(space=['a', 'b', 'c', 'd'])
        >>> default_priors
        {'a': 0.25, 'b': 0.25, 'c': 0.25, 'd': 0.25}

        >>> custom_priors_full = _set_priors({'a' : .3, 'b': .7}, ['a', 'b'])
        >>> custom_priors_full
        {'a': 0.3, 'b': 0.7}

        >>> custom_priors_partial = _set_priors({'a' : .5}, ['a', 'b', 'c'])
        >>> custom_priors_partial
        {'a': 0.5, 'b': 0.25, 'c': 0.25}
    """
    if space is None:
        space = []
    if priors is None:
        priors = {}

    n = len(space)
    default_prior = 1 / n

    # Set all to default to begin with
    _priors = {el: default_prior for el in space}

    # If the user provides priors
    if priors:
        if not set(priors.keys()).issubset(set(space)):
            raise Exception(f"Priors {priors} are not subset of space {space}")
        total_custom_prior = sum(priors.values())
        if total_custom_prior > 1:
            raise ValueError(f"Sum of custom priors {priors} is greater than 1")

        # Update the priors dict with custom values
        for key, value in priors.items():
            _priors[key] = value

        # Adjust the other priors
        remaining_probability = 1 - total_custom_prior
        num_unset_possibilities = n - len(priors)
        for key in _priors:
            if key not in priors:
                _priors[key] = remaining_probability / num_unset_possibilities
    return _normalize_priors(_priors)


def _normalize_priors(priors):
    """normalize priors"""
    total = sum(priors.values())
    if total <= 0:
        warnings.warn(
            f"Sum of priors {priors} is less then 0. Falling back to default priors."
        )
        n = len(priors.keys)
        default_prior = 1 / n
        return {el: default_prior for el in priors.keys()}
    return {el: priors[el] / total for el in priors.keys()}


def _sample_single_tree(
    tree_structures, function_probabilities, operator_probabilities, **kwargs
):
    feature_space = kwargs.get("feature_space")
    function_space = kwargs.get("function_space")
    operator_space = kwargs.get("operator_space")
    require_simplify = kwargs.get("require_simplify")
    is_real_domain = kwargs.get("is_real_domain")
    verbose = kwargs.get("verbose")

    # sample a tree structure
    idx_sample = np.random.randint(0, len(tree_structures))
    tree_structure = tree_structures[idx_sample]
    # check if tree is binary, else continue
    if is_binary_tree(tree_structure) is False:
        return None

    # sample a tree
    tree = EquationTree(
        tree_structure,
        feature_space,
        function_space,
        operator_space,
        function_priors=function_probabilities,
        operation_priors=operator_probabilities,
    )
    # sample a valid equation
    tree.sample_valid()

    # simplify the tree
    if require_simplify:  # simplify equation
        simplified_tree = _simplified_tree(
            tree.expr,
            feature_space,
            function_space,
            operator_space,
            is_real_domain,
            verbose,
        )
        if simplified_tree is None:
            return None
        tree = simplified_tree
    return tree


def _validate_tree_on_conditions(tree, tokenized_equation_list, **kwargs):
    max_num_variables = kwargs.get("max_num_variables")
    max_num_constants = kwargs.get("max_num_constants")
    with_replacement = kwargs.get("with_replacement")
    fix_num_variables_to_max = kwargs.get("fix_num_variables_to_max")

    if not with_replacement and tree.expr in tokenized_equation_list:
        return False
    if tree.num_x > max_num_variables or tree.num_c > max_num_constants:
        return False
    if fix_num_variables_to_max and len(np.unique(tree.variables)) < max_num_variables:
        return False

    return True


def _sample_full_equation(
    tree_structures,
    tokenized_equation_list,
    function_probabilities,
    operator_probabilities,
    **kwargs,
):
    function_space = kwargs.get("function_space")
    operator_space = kwargs.get("operator_space")
    include_zero_as_constant = kwargs.get("include_zero_as_constant")
    num_evaluation_samples = kwargs.get("num_evaluation_samples")
    max_iter = kwargs.get("max_iter")
    is_real_domain = kwargs.get("is_real_domain")
    force_full_domain = kwargs.get("force_full_domain")

    for i in range(max_iter):
        # sample a tree structure
        idx_sample = np.random.randint(0, len(tree_structures))
        tree_structure = tree_structures[idx_sample]
        if not is_binary_tree(tree_structure):
            continue
        # sample a tree structure
        tree = _sample_single_tree(
            tree_structures, function_probabilities, operator_probabilities, **kwargs
        )
        if tree is None:
            continue

        if not _validate_tree_on_conditions(tree, tokenized_equation_list, **kwargs):
            continue

        # now we evaluate each node in the tree for a grid of inputs and constants
        crossings = create_crossings(tree.num_x, tree.num_c, **kwargs)

        evaluation = get_evaluation(
            crossings,
            tree,
            num_evaluation_samples,
            include_zero_as_constant=include_zero_as_constant,
        )

        # if any of the evaluations are infinite or nan, continue
        if force_full_domain:
            if np.any(np.isinf(evaluation)) or np.any(np.isnan(evaluation)):
                continue
            else:
                break
        else:
            if np.all(np.isinf(evaluation) | np.isnan(evaluation)):
                continue
            else:
                break
    return (
        tree.expr,
        _tree_expr_to_sympy(
            tree.expr, function_space, operator_space, is_real_domain, False
        ),
        evaluation,
    )
