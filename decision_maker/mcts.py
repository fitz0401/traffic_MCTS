'''
Author: haroldsultan
Modified by Licheng Wen
Date: 2022-08-22 14:33:58
Description: 
A quick Monte Carlo Tree Search implementation.
For more details on MCTS see http://www.incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf

Copyright (c) 2022 by PJLab, All Rights Reserved. 
'''
#!/usr/bin/env python
import random
import math
import hashlib
import logging
import argparse
import time


# MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
SCALAR = 2 / (2 * math.sqrt(2.0))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')


class State:
    NUM_TURNS = 10
    GOAL = 0
    MOVES = [2, -2, 3, -3]
    MAX_VALUE = (5.0 * (NUM_TURNS - 1) * NUM_TURNS) / 2
    num_moves = len(MOVES)

    def __init__(self, value=0, moves=[], turn=NUM_TURNS):
        self.value = value
        self.turn = turn
        self.moves = moves

    def next_state(self):
        nextmove = random.choice([x * self.turn for x in self.MOVES])
        next = State(self.value + nextmove, self.moves + [nextmove], self.turn - 1)
        return next

    def terminal(self):
        if self.turn == 0:
            return True
        return False

    def reward(self):  # reward have to have their support in [0, 1]
        r = 1 - (abs(self.value - self.GOAL) / self.MAX_VALUE)
        return r

    def __hash__(self):
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = "Value: %d; Moves: %s" % (self.value, self.moves)
        return s


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
        if num_moves_lambda != None:
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
    for iter in range(int(budget)):
        if iter % 10000 == 9999:
            logger.info("simulation: %d" % iter)
            logger.info(root)
        front = tree_policy(root, num_moves_lambda)
        reward = default_policy(front.state)  # can parallelize here
        backpropagation(front, reward)

    return best_child(root, 0)
    
def default_policy(state):
    while state.terminal() == False:
        state = state.next_state()
    return state.reward()

def tree_policy(node, num_moves_lambda):
    # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.state.terminal() == False:
        if len(node.children) == 0:
            return expand(node)
        elif random.uniform(0, 1) < 0.5:
            node = best_child(node, SCALAR)
        else:
            if node.fully_expanded(num_moves_lambda) == False:
                return expand(node)
            else:
                node = best_child(node, SCALAR)
    return node


def expand(node):
    tried_children = [c.state for c in node.children]
    new_state = node.state.next_state(tried_children)
    while new_state in tried_children and new_state.terminal() == False:
        print("should not be here!!!")
        new_state = node.state.next_state()
    node.add_child(new_state)
    return node.children[-1]


def best_child(
    node, scalar
):  # current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
    bestscore = 0.0
    bestchildren = []
    for c in node.children:
        exploit = c.reward / c.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
        score = exploit + scalar * explore
        if score == bestscore:
            bestchildren.append(c)
        if score > bestscore:
            bestchildren = [c]
            bestscore = score
    if len(bestchildren) == 0:
        logger.warn("OOPS: no best child found, probably fatal")
    return random.choice(bestchildren)





def backpropagation(node, reward):
    while node != None:
        node.visits += 1
        node.reward += reward
        node = node.parent
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCTS research code')
    parser.add_argument('--num_sims', action="store", required=True, type=int)
    parser.add_argument(
        '--levels',
        action="store",
        required=True,
        type=int,
        choices=range(State.NUM_TURNS + 1),
    )
    args = parser.parse_args()

    current_node = Node(State())  # root node
    for l in range(args.levels):
        start_time = time.time()
        old_node = current_node
        current_node = uct_search(args.num_sims / (l + 1), current_node)
        print("level %d" % l)
        print("Num Children: %d" % len(old_node.children))
        for i, c in enumerate(old_node.children):
            print(i, c)
        print("Best Child: %s" % current_node.state)

        temp_best = current_node
        while temp_best.children != []:
            temp_best = best_child(temp_best, 0)
        print("Temp Best Child: %s" % temp_best.state)

        print("-------------", time.time() - start_time, "-------------------")
