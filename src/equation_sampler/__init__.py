import itertools

import numpy as np
from sympy import simplify, symbols, sympify, I

from .equation_tree import EquationTree, is_binary_tree, rooted_tree_iterator
from .util.unary_minus_to_binary import unary_minus_to_binary

padding = "<PAD>"


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
            output.append(infix[i: i + 3][::-1])
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
        without_replacement: bool = True,
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
        require_simplify=True,
        verbose=False,
        is_real_domain=True,
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
        without_replacement: Whether to sample without replacement.
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
        require_simplify: Defines if the equations are simplified
        verbose: Defines if additional output is generated
        is_real_domain: Defines if the variables and constants are real or complex numbers
    """
    if max_depth < 3:
        raise ValueError("max_depth must be at least 3")

    tokenized_equation_list = list()
    sympy_equation_list = list()
    evaluation_list = list()
    max_equation_elements = 0

    # Generate Feature Space
    feature_space = [f"x_{i + 1}" for i in range(max_num_variables)] + \
                    [f"c_{i + 1}" for i in range(max_num_constants)]

    if include_zero_as_constant:
        feature_space.append("0")

    # Generate all possible trees
    tree_structures = [
        tree.copy() for depth in range(3, max_depth + 1) for tree in rooted_tree_iterator(depth)
    ]

    # sample
    for sample in range(num_samples):
        for i in range(max_iter):
            # sample a tree structure
            idx_sample = np.random.randint(0, len(tree_structures))
            tree_structure = tree_structures[idx_sample]
            # check if tree is binary, else continue
            if is_binary_tree(tree_structure) is False:
                continue

            # sample a tree
            tree = EquationTree(
                tree_structure, feature_space, function_space, operator_space
            )
            # sample a valid equation
            tree.sample_valid()

            # simplify the tree
            if require_simplify:  # simplify equation
                simplified_tree = _simplified_tree(
                    tree.expr, feature_space, function_space,
                    operator_space, is_real_domain, verbose
                )
                if simplified_tree is None:
                    continue
                tree = simplified_tree

            # if we want to sample without replacement and if tree is already sampled, continue
            if without_replacement and tree.expr in tokenized_equation_list:
                continue

            # if tree has too many variables or constants, continue
            if tree.num_x > max_num_variables or tree.num_c > max_num_constants:
                continue

            if (
                    fix_num_variables_to_max
                    and len(np.unique(tree.variables)) < max_num_variables
            ):
                continue

            # now we evaluate each node in the tree for a grid of inputs and constants
            crossings = create_crossings(
                tree.num_x,
                tree.num_c,
                min_input_value,
                max_input_value,
                min_constant_value,
                max_constant_value,
                num_input_points,
                num_constant_points,
                num_evaluation_samples,
            )

            evaluation = get_evaluation(
                crossings,
                tree,
                num_evaluation_samples,
                include_zero_as_constant=include_zero_as_constant,
            )

            # if any of the evaluations are infinite or nan, continue
            if np.any(np.isinf(evaluation)) or np.any(np.isnan(evaluation)):
                continue
            else:
                break

        # add to lists
        tokenized_equation_list.append(tree.expr)
        sympy_equation_list.append(
            _tree_expr_to_sympy(tree.expr, function_space, operator_space, is_real_domain, False))
        evaluation_list.append(evaluation)
        max_equation_elements = max(max_equation_elements, len(tree.expr))

        # print progress
        if sample % 10 == 0:
            print(f"{sample} equations generated")

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
        'tokenized_equations': tokenized_equation_list,
        'sympy_equations': sympy_equation_list,
        'evaluations': evaluation_list
    }


def create_crossings(
        num_inputs,
        num_constants,
        min_input_value: float = -1,
        max_input_value: float = 1,
        min_constant_value: float = -1,
        max_constant_value: float = 1,
        num_input_points: int = 100,
        num_constant_points: int = 100,
        num_evaluation_samples: int = 100,
):
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


def _simplified_tree(expr, feature_space, function_space, operator_space, is_real_domain, verbose):
    """
    Takes a tree expression as input and returns a simplified tree
    """
    current_infix = prefix_to_infix(
        expr, function_space, operator_space
    )
    if verbose:
        print("_________")
        print("infix initial", current_infix)
        print("initial tree", expr)

    sympy_expr = _tree_expr_to_sympy(expr, function_space, operator_space, is_real_domain,
                                     verbose)

    simplified_equation = simplify(sympy_expr)

    if I in simplified_equation.free_symbols:
        return None

    simplified_equation = unary_minus_to_binary(str(simplified_equation),
                                                operator_space)

    simplified_equation = simplified_equation.replace(" ", "")
    simplified_equation = simplified_equation.replace("**", "^")

    prefix = infix_to_prefix(
        simplified_equation, function_space, operator_space
    )
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


#np.random.seed(42)
sample_equations(100, 6, 3, 5)
