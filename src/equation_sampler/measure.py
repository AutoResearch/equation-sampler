def get_frequencies(equations, **kwargs):
    """
    Get the frequencies of operators and equations

    Examples:
        # set the spaces
        >>> config = {
        ... "function_space": ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"],
        ... "operator_space": ["+", "-", "*", "/", "^"],
        ... }

        # a list of equations in tokenized form:
        >>> equations = [
        ... ['*', '2', 'x_1'],
        ... ['^', 'x_1', '2'],
        ... ['sin', 'cos', 'x_1'],
        ... ['log', 'cos', 'x_1']
        ... ]

        # get the frequencies:
        >>> get_frequencies(equations, **config)
        ({'sin': 0.25, 'cos': 0.5, 'tan': 0.0, 'exp': 0.0, 'log': 0.25, 'sqrt': 0.0, 'abs': 0.0}, \
{'+': 0.0, '-': 0.0, '*': 0.5, '/': 0.0, '^': 0.5})
    """
    function_space = kwargs.get("function_space")
    operator_space = kwargs.get("operator_space")
    functions_total = 0
    operators_total = 0
    functions_appearances = {fct: 0 for fct in function_space}
    operators_appearances = {op: 0 for op in operator_space}
    for e in equations:
        functions_total += _count_appearances(e, function_space)
        operators_total += _count_appearances(e, operator_space)
        for key in e:
            if key in function_space:
                functions_appearances[key] += 1
            if key in operator_space:
                operators_appearances[key] += 1

    if functions_total > 0:
        functions_frequencies = {
            key: functions_appearances[key] / functions_total
            for key in functions_appearances.keys()
        }
    else:
        functions_frequencies = functions_appearances

    if operators_total > 0:
        operators_frequencies = {
            key: operators_appearances[key] / operators_total
            for key in operators_appearances.keys()
        }
    else:
        operators_frequencies = operators_appearances

    return functions_frequencies, operators_frequencies


def _count_appearances(list1, list2):
    """
    Number of appearances in list1 that are also present in another list2

    Examples:
        >>> _3 = _count_appearances([1, 2, 3, 1], [1, 2])
        >>> _3
        3

        >>> _1 = _count_appearances(['a', 'b', 'cd'], [1, 'cd'])
        >>> _1
        1

        >>> _0 = _count_appearances(['a', 'b'], [1, 2])
        >>> _0
        0

    """
    return sum(1 for item in list1 if item in list2)
