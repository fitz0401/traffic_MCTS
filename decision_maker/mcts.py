"""
Author: haroldsultan
Modified by Licheng Wen
Date: 2022-08-22 14:33:58
Description:
A quick Monte Carlo Tree Search implementation.
For more details on MCTS see
http://www.incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf

Copyright (c) 2022 by PJLab, All Rights Reserved.
"""
import random
import math
import logging


# MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
SCALAR = 2 / (2 * math.sqrt(2.0))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')

EXPAND_NODE = 0


class Node:
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self, num_moves_lambda):
        num_moves = self.state.num_moves
        if num_moves_lambda is not None:
            num_moves = num_moves_lambda(self)
        if len(self.children) == num_moves:
            return True
        return False

    def __repr__(self):
        s = "Node: %s\n\tChildren: %d; visits: %d; reward: %f, exploit: %f" % (
            self.state,
            len(self.children),
            self.visits,
            self.reward,
            self.reward / self.visits,
        )
        return s


def uct_search(budget, root, num_moves_lambda=None):
    for iteration in range(int(budget)):
        if iteration % 100 == 0:
            logger.info("simulation: %d" % iteration)
            logger.info(root)
        front = tree_policy(root, num_moves_lambda)
        reward = default_policy(front.state)  # can parallelize here
        backpropagation(front, reward)

    return best_child(root, 0)


def default_policy(state):
    while not state.terminal():
        state = state.next_state()
    return state.reward()


def tree_policy(node, num_moves_lambda):
    # a hack to force 'exploitation' in a game where there are many options,
    # and you may never/not want to fully expand first
    while not node.state.terminal():
        if len(node.children) == 0:
            return expand(node)
        elif random.uniform(0, 1) < 0.5:
            node = best_child(node, SCALAR)
        else:
            if not node.fully_expanded(num_moves_lambda):
                return expand(node)
            else:
                node = best_child(node, SCALAR)
    return node


def expand(node):
    tried_children = [child.state for child in node.children]
    # TODO: 此处的代码逻辑有问题，已经在next_state中传入了tried_children，但仍会进入while循环
    new_state = node.state.next_state(tried_children)
    while new_state in tried_children and not new_state.terminal():
        # print("should not be here!!!")
        new_state = node.state.next_state()
    node.add_child(new_state)
    global EXPAND_NODE
    EXPAND_NODE += 1
    return node.children[-1]


def best_child(node, scalar):
    # current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
    best_score = 0.0
    best_children = []
    for child in node.children:
        exploit = child.reward / child.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / float(child.visits))
        score = exploit + scalar * explore
        if score == best_score:
            best_children.append(child)
        if score > best_score:
            best_children = [child]
            best_score = score
    if len(best_children) == 0:
        logger.warning("OOPS: no best child found, probably fatal")
    return random.choice(best_children)


def backpropagation(node, reward):
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent
    return
