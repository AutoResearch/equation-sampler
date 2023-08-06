from sympy import simplify


def find_unary(symbol, expression, operators):
    expression = expression.replace(' ', '')
    result = []
    prev_token = None

    for token in expression:
        if token == symbol:
            if prev_token is None or prev_token in ['('] + operators:
                # Unary minus found, convert it to binary minus
                token = '#'
            # If it's not a unary minus, leave it as it is (e.g., negative number)

        result.append(token)
        prev_token = token

    return ''.join(result)


def move_placeholder(expression, operators):
    expression = expression.replace(' ', '')  # Remove any whitespaces
    i = 0
    has_to_close = False
    start = None
    end = None
    while i < len(expression):
        char = expression[i]
        if char == '(':
            has_to_close = True
        if char == '#':
            start = i
            open_brackets = 0
            # Move the expression after % to the end of the equation
            j = i + 1
            while j < len(expression) and expression[j] not in operators or open_brackets >= 1:
                j += 1
                if j < len(expression) - 1 and expression[j] == '(':
                    open_brackets += 1
                if j < len(expression) and expression[j] == ')':
                    open_brackets -= 1
                    if open_brackets == 0:
                        j += 1
                        break

            end = j
            break
        i += 1
    if (start is None or end is None):
        return
    if has_to_close:
        insert = find_next_closing_parenthesis(expression, end)
    else:
        insert = len(expression)

    replacement = '-' + expression[start:end][1:]
    new_expression = insert_string_at_position(expression, replacement, insert)
    new_expression = delete_chars_between_indexes(new_expression, start, end - 1)
    return new_expression


def replace_with_zero_minus(expression, operators):
    expression = expression.replace(' ', '')  # Remove any whitespaces
    i = 0
    has_to_close = False
    start = None
    end = None
    while i < len(expression):
        char = expression[i]
        if char == '(':
            has_to_close = True
        if char == '#':
            start = i
            open_brackets = 0
            # Move the expression after % to the end of the equation
            j = i + 1
            while j < len(expression) and expression[j] not in operators or open_brackets >= 1:
                j += 1
                if j < len(expression) - 1 and expression[j] == '(':
                    open_brackets += 1
                if j < len(expression) and expression[j] == ')':
                    open_brackets -= 1
                    if open_brackets == 0:
                        j += 1
                        break

            end = j
            break
        i += 1
    if (start is None or end is None):
        return
    if has_to_close:
        insert = find_next_closing_parenthesis(expression, end) - 1
    else:
        insert = len(expression)

    replacement = '(0-' + expression[start:end][1:] + ')'
    new_expression = insert_string_at_position(expression, replacement, insert)
    new_expression = delete_chars_between_indexes(new_expression, start, end - 1)

    return new_expression


def delete_chars_between_indexes(input_string, i, j):
    if i < 0:
        i = 0
    if j >= len(input_string):
        j = len(input_string)

    return input_string[:i] + input_string[j + 1:]


def remove_character_from_string(input_string, character):
    return input_string.replace(character, '')


def insert_string_at_position(original_string, string_to_insert, position):
    return original_string[:position] + string_to_insert + original_string[position:]


def find_next_closing_parenthesis(input_string, j):
    for i in range(j, len(input_string)):
        if input_string[i] == ')':
            return i
    return -1  # Return -1 if closing parenthesis is not found after index j


def unary_minus_to_binary(expr, operators):
    _temp = find_unary('-', str(expr), operators)
    while '#' in _temp:
        _temp = move_placeholder(_temp, operators)
    _temp = find_unary('+', _temp, operators)
    _temp = remove_character_from_string(_temp, '#')
    _temp = find_unary('-', _temp, operators)
    while '#' in _temp:
        _temp = replace_with_zero_minus(_temp, operators)
    return _temp
