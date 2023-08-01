import numpy as np

from autora.experiment_runner.synthetic.generator.equations.equation_tree import (
    EquationTree,
)


def test_equation_sampler():
    np.random.seed(42)
    tree_structure = [0, 1, 2, 3, 2, 1]
    operation_space = ["+", "-", "*", "/"]
    function_space = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"]
    feature_space = ["x_1", "x_2", "c_i"]

    features = {"x_1": 1, "x_2": 2, "c_1": -1, "c_2": -2, "c_3": -3}

    sampler = EquationTree(
        tree_structure, feature_space, function_space, operation_space
    )
    root = sampler.sample_valid()
    expr = sampler.expr
    assert expr == ["-", "/", "sqrt", "x_2", "x_1", "x_1"]
    validity = sampler.check_tree_validity(root)
    assert validity
    output = sampler.evaluate_node(features, sampler.root)
    assert output == 0.41421356237309515
    eval = sampler.evaluate(features)
    assert eval == [
        0.41421356237309515,
        1.4142135623730951,
        1.4142135623730951,
        2,
        1,
        1,
    ]


def test_prefix_instantiation():
    prefix_notation = ["log", "abs", "x_1", "<PAD>"]
    np.random.seed(42)
    function_space = [
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "abs",
    ]  # functions allowed in the equations
    operation_space = ["+", "-", "*", "/"]
    feature_space = [
        "x_1",
        "x_2",
        "c_1",
        "c_2",
        "0",
    ]  # variables allowed in the equations
    tree = EquationTree([], feature_space, function_space, operation_space)
    tree.instantiate_from_prefix_notation(prefix_notation)
    assert tree.expr == ["log", "abs", "x_1"]
    features = {"x_1": -4}
    output = tree.evaluate_node(features, tree.root)
    assert output == 1.3862943611198906


def test_sampler():
    operation_space = ["+", "-", "*", "/"]
    function_space = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"]
    feature_space = ["x_1", "x_2", "c_i"]
    tree_structure = [0, 1, 2, 3, 2, 1]

    for i in range(10):
        sampler = EquationTree(
            tree_structure, feature_space, function_space, operation_space
        )
        sampler.sample_valid()
        expr = sampler.expr
        num_x = sampler.num_x
        validity = sampler.check_tree_validity()
        assert validity
        assert expr.count("x_1") + expr.count("x_2") == num_x
