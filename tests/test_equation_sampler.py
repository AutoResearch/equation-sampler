import numpy as np

from src import infix_to_prefix, prefix_to_infix, simplify


def test_simplification():
    np.random.seed(42)
    function_space = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"]
    operation_space = ["+", "-", "*", "/"]
    tree = ["*", "sqrt", "x_1", "x_1"]
    current_infix = prefix_to_infix(tree, function_space, operation_space)
    assert current_infix == "(sqrt(x_1)*x_1)"
    assert tree == ["*", "sqrt", "x_1", "x_1"]
    simplified_equation = simplify(current_infix)
    simplified_equation = str(simplified_equation)
    simplified_equation = simplified_equation.replace(" ", "")
    simplified_equation = simplified_equation.replace("**", "^")
    prefix = infix_to_prefix(simplified_equation, function_space, operation_space)
    assert prefix == ["^", "x_1", "/", "3", "2"]
