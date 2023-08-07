from enum import Enum
from typing import Callable, Dict, List, Union

import numpy as np

UnaryOperator = Callable[[Union[int, float]], Union[int, float]]
BinaryOperator = Callable[[Union[int, float], Union[int, float]], Union[int, float]]

operators: Dict[str, BinaryOperator] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "^": lambda a, b: a**b,
}

functions: Dict[str, UnaryOperator] = {
    "sin": lambda a: np.sin(a),
    "cos": lambda a: np.cos(a),
    "tan": lambda a: np.tan(a),
    "exp": lambda a: np.exp(a),
    "log": lambda a: np.log(a),
    "sqrt": lambda a: np.sqrt(a),
    "abs": lambda a: np.abs(a),
}


class NodeType(Enum):
    NONE = 0
    FUNCTION = 1
    OPERATION = 2
    INPUT = 3
    CONSTANT = 4


class TreeNode:
    def __init__(
        self, val=0, left=None, right=None, parent=None, attribute="", evaluation=0
    ):
        self.val = val
        self.children = []
        self.parent = parent
        self.left = left
        self.right = right
        self.attribute = attribute
        self.evaluation = evaluation
        self.type = NodeType.NONE
        self.is_leaf = False


def generate_parent_pointers(levels, values):
    n = len(levels)
    if n == 0:
        return None
    root = TreeNode(val=values[0])
    parent = None
    stack = [(root, levels[0])]
    for i in range(1, n):
        level = levels[i]
        node = TreeNode(val=values[i])
        while stack and level <= stack[-1][1]:
            parent = stack.pop()[0]
        parent = stack[-1][0]

        if parent:
            parent.children.append(node)
            node.parent = parent

        stack.append((node, level))
    return root


def dfs_parents(node, parents_count):
    if not node:
        return

    if node.parent is not None:
        parents_count[node.parent.val] += 1

    for child in node.children:
        dfs_parents(child, parents_count)


def is_binary_tree(tree_structure):
    parents_count = [0] * len(tree_structure)
    root = generate_parent_pointers(tree_structure, range(len(tree_structure)))
    dfs_parents(root, parents_count)
    return all(count <= 2 for count in parents_count)


def rooted_tree_iterator(n, verbose=False):
    r"""
    Iterator through regular level sequences of rooted trees.
    (only works for n >= 3)

    EXAMPLES::

        sage: from surface_dynamics.misc.plane_tree import rooted_tree_iterator
        sage: for t in rooted_tree_iterator(4): print(t)
        [0, 1, 2, 3]
        [0, 1, 2, 2]
        [0, 1, 2, 1]
        [0, 1, 1, 1]
        sage: for t in rooted_tree_iterator(5): print(t)
        [0, 1, 2, 3, 4]
        [0, 1, 2, 3, 3]
        [0, 1, 2, 3, 2]
        [0, 1, 2, 3, 1]
        [0, 1, 2, 2, 2]
        [0, 1, 2, 2, 1]
        [0, 1, 2, 1, 2]
        [0, 1, 2, 1, 1]
        [0, 1, 1, 1, 1]

        sage: for t in rooted_tree_iterator(5,verbose=True): pass
          p =    4
          prev = [0, 1, 2, 3, 4]
          save = [0, 0, 0, 0, 0]
        [0, 1, 2, 3, 4]
          p =    4
          prev = [0, 1, 2, 3, 4]
          save = [0, 0, 0, 0, 0]
        [0, 1, 2, 3, 3]
          p =    4
          prev = [0, 1, 2, 3, 4]
          save = [0, 0, 0, 0, 0]
        [0, 1, 2, 3, 2]
          p =    3
          prev = [0, 1, 2, 0, 4]
          save = [0, 0, 0, 0, 0]
        [0, 1, 2, 3, 1]
          p =    4
          prev = [0, 1, 3, 0, 4]
          save = [0, 0, 0, 2, 0]
        [0, 1, 2, 2, 2]
          p =    3
          prev = [0, 1, 2, 0, 4]
          save = [0, 0, 0, 2, 0]
        [0, 1, 2, 2, 1]
          p =    4
          prev = [0, 3, 2, 0, 4]
          save = [0, 0, 0, 1, 0]
        [0, 1, 2, 1, 2]
          p =    2
          prev = [0, 1, 0, 0, 4]
          save = [0, 0, 0, 1, 0]
        [0, 1, 2, 1, 1]
          p =    0
          prev = [0, 0, 0, 0, 4]
          save = [0, 0, 0, 1, 0]
        [0, 1, 1, 1, 1]
    """
    assert n >= 3

    levels = list(range(n))
    prev = list(range(n))  # function: level -> ?
    save = [0] * n
    p = n - 1

    if verbose:
        print("  p =    %s" % p)
        print("  prev = %s" % prev)
        print("  save = %s" % save)
        print(levels)
    root = generate_parent_pointers(levels, np.arange(n))
    parents_count = {i: 0 for i in range(n)}
    dfs_parents(root, parents_count)
    # print("PARENTS", parents_count)

    yield levels

    while p > 0:
        levels[p] = levels[p] - 1
        if p < n and (levels[p] != 1 or levels[p - 1] != 1):
            diff = p - prev[levels[p]]  # = p-q
            while p < n - 1:
                save[p] = prev[levels[p]]
                prev[levels[p]] = p
                p += 1
                levels[p] = levels[p - diff]
        while levels[p] == 1:
            p -= 1
            prev[levels[p]] = save[p]

        if verbose:
            print("  p =    %s" % p)
            print("  prev = %s" % prev)
            print("  save = %s" % save)
            print(levels)
        root = generate_parent_pointers(levels, np.arange(n))
        parents_count = {i: 0 for i in range(n)}
        dfs_parents(root, parents_count)

        # parents_count represents how many children a parent has
        # for binary we want at most 2 children per parent
        if any(value > 2 for value in parents_count.values()):
            continue
        yield levels


# https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
def bfs(s):
    # Create a queue for BFS
    queue = []

    # enqueue source node
    queue.append(s)

    while queue:
        # Dequeue a vertex from
        # queue and print it
        s = queue.pop(0)
        print("parent", s.val)

        # Get all adjacent vertices of the
        # dequeued vertex s.
        # If an adjacent has not been visited,
        # then mark it visited and enqueue it
        print("children")
        for v in s.children:
            print(v.val)
            queue.append(v)


def get_children(tree_structure, index):
    parent = tree_structure[index]
    children = []
    for i in range(index + 1, len(tree_structure)):
        if tree_structure[i] == parent:
            break
        if tree_structure[i] == parent + 1:
            children.append(i)

    return children


def count_children(tree_structure, index):
    parent = tree_structure[index]
    children = 0
    for i in range(index + 1, len(tree_structure)):
        if tree_structure[i] == parent:
            break
        if tree_structure[i] == parent + 1:
            children += 1

    return children


class EquationTree:
    def __init__(
        self,
        tree_structure: np.array,
        feature_space: List[str],
        function_space: List[str],
        operation_space: List[str],
        feature_priors: Dict[str, float] = dict(),
        function_priors: Dict[str, float] = dict(),
        operation_priors: Dict[str, float] = dict(),
    ):

        if is_binary_tree(tree_structure) is False:
            raise Exception("Tree structure must be a binary tree: %s" % tree_structure)

        self.tree_structure = tree_structure
        self.feature_space = feature_space
        self.operating_feature_space = feature_space.copy()
        self.function_space = function_space
        self.operation_space = operation_space
        self.feature_priors = feature_priors
        self.function_priors = function_priors
        self.operation_priors = operation_priors

        self.root = None
        self.num_x = 0
        self.num_c = 0
        self.num_leafs = 0
        self.variables: List[str] = list()
        self.constants: List[str] = list()
        self.expr: List[str] = list()

    def sample(self, tree_structure=None):
        if tree_structure is not None:
            self.tree_structure = tree_structure
        self.operating_feature_space = self.feature_space.copy()
        self.root, self.num_x, self.num_c, self.expr = self.sample_equation_tree(0)
        return self.root

    def sample_valid(self, max_tries=1000, verbose=False):
        for i in range(max_tries):
            self.operating_feature_space = self.feature_space.copy()
            self.root, self.num_x, self.num_c, self.expr = self.sample_equation_tree(0)
            self.variables = list()
            self.constants = list()
            self.num_leafs = 0
            if self.check_tree_validity(self.root, verbose=verbose):
                self.collect_variables(self.root)
                self.collect_constants(self.root)
                return self.root
        raise Exception(
            "Could not sample a valid equation tree after %s tries" % max_tries
        )

    def sample_equation_tree(self, index, num_x=0, num_c=0, parent_attribute=""):
        expr = list()
        attribute = self.sample_attribute_from_tree(index, parent_attribute)

        if attribute in self.function_space:
            type = NodeType.FUNCTION
        elif attribute in self.operation_space:
            type = NodeType.OPERATION
        elif attribute in self.feature_space:
            if "x_" in attribute:
                num_x += 1
                type = NodeType.INPUT
            elif "c_" in attribute:
                num_c += 1
                attribute = "c_" + str(num_c)
                type = NodeType.CONSTANT
            elif "0" in attribute:
                type = NodeType.CONSTANT
        else:
            type = NodeType.NONE

        self.type = type

        node = TreeNode(val=self.tree_structure[index], attribute=attribute)

        expr.append(attribute)

        children = get_children(self.tree_structure, index)

        if len(children) >= 1:
            node.left, num_x, num_c, expr_add = self.sample_equation_tree(
                children[0], num_x, num_c, parent_attribute=attribute
            )
            for expr_element in expr_add:
                expr.append(expr_element)

        if len(children) == 2:
            node.right, num_x, num_c, expr_add = self.sample_equation_tree(
                children[1], num_x, num_c, parent_attribute=attribute
            )
            for expr_element in expr_add:
                expr.append(expr_element)

        return node, num_x, num_c, expr

    def sample_attribute_from_tree(self, index, parent_attribute=""):
        num_children = count_children(self.tree_structure, index)
        if num_children == 0:
            return self.sample_attribute(
                self.operating_feature_space, self.feature_priors, parent_attribute
            )
        elif num_children == 1:
            return self.sample_attribute(
                self.function_space, self.function_priors, parent_attribute
            )
        elif num_children == 2:
            return self.sample_attribute(
                self.operation_space, self.operation_priors, parent_attribute
            )
        else:
            raise Exception("Invalid number of children: %s" % num_children)

    def sample_attribute(
        self, attribute_list: List[str], priors: Dict, parent_attribute=""
    ):
        probabilities = np.ones(len(attribute_list))
        for idx, attribute in enumerate(attribute_list):
            if parent_attribute != "":
                key = parent_attribute + "_" + attribute
            else:
                key = attribute
            if key in priors:
                probabilities[idx] = priors[key]

        probabilities = probabilities / np.sum(probabilities)
        sample_index = np.random.choice(len(attribute_list), p=probabilities)
        return attribute_list[sample_index]

    def collect_variables(self, node=None):
        if node is None:
            if self.root is not None:
                return self.collect_variables(self.root)
            else:
                return list()

        if "x_" in node.attribute:
            if node.attribute not in self.variables:
                self.variables.append(node.attribute)
        if node.left is not None:
            self.collect_variables(node.left)
        if node.right is not None:
            self.collect_variables(node.right)
        return self.variables

    def collect_constants(self, node=None):
        if node is None:
            if self.root is not None:
                return self.collect_constants(self.root)
            else:
                return list()

        if "c_" in node.attribute or "0" in node.attribute:
            if node.attribute not in self.constants:
                self.constants.append(node.attribute)
        if node.left is not None:
            self.collect_constants(node.left)
        if node.right is not None:
            self.collect_constants(node.right)
        return self.constants

    def check_tree_validity(self, node=None, verbose=False):
        if node is None:
            if self.root is not None:
                return self.check_tree_validity(self.root, verbose)
            else:
                return True

        if node.attribute in self.function_space:
            if node.left is None:
                return False
            if "log" in node.attribute and "0" in node.left.attribute:
                if verbose:
                    print(
                        "logarithm is applied to 0 which is results in not real number."
                    )
                return False
            if (
                "c_" in node.left.attribute
            ):  # function of a constant is a constant (unnecessary complexity)
                if verbose:
                    print(
                        "%s is a constant applied to a function %s"
                        % (node.left.attribute, node.attribute)
                    )
                return False
            return self.check_tree_validity(node.left, verbose)

        elif node.attribute in self.operation_space:
            if node.left is None or node.right is None:
                return False
            if "/" in node.attribute and "0" in node.right.attribute:
                if verbose:
                    print("division by 0 is not allowed.")
                return False
            if "c_" in node.left.attribute and "c_" in node.right.attribute:
                if verbose:
                    print(
                        "%s and %s are constants applied to the same operation %s"
                        % (node.left.attribute, node.right.attribute, node.attribute)
                    )
                return False  # operation of two constants is a constant (unnecessary complexity)
            return self.check_tree_validity(
                node.left, verbose
            ) and self.check_tree_validity(node.right, verbose)

        else:
            if node.left is None and node.right is None:
                self.num_leafs += 1
            return True

    def evaluate(self, features: Dict):
        eval: List[float] = list()

        if self.root is not None:
            self.evaluate_node(features, self.root)
            eval = self.get_full_evaluation(self.root)

        return eval

    def evaluate_node(self, features: Dict, node: TreeNode):
        if node is None:
            if self.root is not None:
                value = self.evaluate_node(features, self.root)
            else:
                value = 0

        if node.attribute in self.function_space:
            if node.left is None:
                raise Exception("Invalid tree: %s" % self.expr)
            value = functions[node.attribute](self.evaluate_node(features, node.left))

        elif node.attribute in self.operation_space:
            if node.left is None or node.right is None:
                raise Exception("Invalid tree: %s" % self.expr)
            value = operators[node.attribute](
                self.evaluate_node(features, node.left),
                self.evaluate_node(features, node.right),
            )

        elif node.attribute in features:
            value = features[node.attribute]

        else:
            raise Exception("Invalid attribute %s" % node.attribute)

        node.evaluation = value
        return value

    def get_full_evaluation(self, node: TreeNode):
        eval = list()
        eval.append(node.evaluation)

        if node.attribute in self.function_space:
            if node.left is None:
                raise Exception("Invalid tree: %s" % self.expr)
            eval_add = self.get_full_evaluation(node.left)
            for eval_element in eval_add:
                eval.append(eval_element)

        if node.attribute in self.operation_space:
            if node.left is None or node.right is None:
                raise Exception("Invalid tree: %s" % self.expr)
            eval_add = self.get_full_evaluation(node.left)
            for eval_element in eval_add:
                eval.append(eval_element)
            eval_add = self.get_full_evaluation(node.right)
            for eval_element in eval_add:
                eval.append(eval_element)

        return eval

    def instantiate_from_prefix_notation(self, prefix_notation: List):

        self.operating_prefix_notation = prefix_notation.copy()
        self.root, self.num_x, self.num_c, self.expr = self.generate_from_prefix(0)
        return self.root

    def generate_from_prefix(self, num_x=0, num_c=0):
        expr = list()
        attribute = self.operating_prefix_notation[0]

        if attribute in self.function_space:
            type = NodeType.FUNCTION
        elif attribute in self.operation_space:
            type = NodeType.OPERATION
        elif attribute in self.feature_space:
            if "x_" in attribute:
                num_x += 1
                type = NodeType.INPUT
            elif "c_" in attribute:
                num_c += 1
                attribute = "c_" + str(num_c)
                type = NodeType.CONSTANT
            elif "0" in attribute:
                type = NodeType.CONSTANT
        else:
            type = NodeType.NONE

        self.type = type

        node = TreeNode(val=len(self.operating_prefix_notation), attribute=attribute)

        expr.append(attribute)

        if type == NodeType.FUNCTION:
            children = 1
        elif type == NodeType.OPERATION:
            children = 2
        else:
            children = 0

        if len(self.operating_prefix_notation) <= 1:
            children = 0

        # remove attribute
        self.operating_prefix_notation.pop(0)

        if children >= 1:
            node.left, num_x, num_c, expr_add = self.generate_from_prefix(num_x, num_c)
            for expr_element in expr_add:
                expr.append(expr_element)

        if children == 2:
            node.right, num_x, num_c, expr_add = self.generate_from_prefix(num_x, num_c)
            for expr_element in expr_add:
                expr.append(expr_element)

        return node, num_x, num_c, expr
