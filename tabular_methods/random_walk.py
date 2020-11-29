import random
import time


class RandomWalk(object):
    """the environment"""

    def __init__(self, road_len):
        """
        initial
        :param road_len: the length of road
        """
        self.road_len = road_len
        self.target = self.road_len

    def walk_onestep(self, s):
        """show the state"""
        if s==self.target:
            print('\r'+ '-' * s + '1', end='', flush=True)
        else:
            print('\r'+'-' * s + '1' + '-' * (self.road_len - s -1) + 'O', end='', flush=True)
        time.sleep(1)

    def walk(self, pai=None):
        """
        walk following by policy pai
        :param pai: a dict or a function. policy pai [default: random].
        """
        self.chain = {
            'S': {},
            'A': {},
            'R': {}
        }  # S,A,R
        t = 0
        s = 0
        g = 0
        self.chain['S'][t] = s
        self.walk_onestep(s)
        while True:
            if pai is None:
                a = self.random_pai(s)
            else:
                a = pai[s] if isinstance(pai, dict) else pai(s)
            self.chain['A'][t] = a
            t += 1
            s, r = self.next_state(s, a)
            self.chain['S'][t], self.chain['R'][t] = s, r
            g += r
            self.walk_onestep(s)
            if s == self.target:
                break

    def action(self, s):
        """
        correct actions under the given state
        :param s: the state
        :return: the action list

        s==0: # the left of road, person can only walk left.
            action_list = ['right']
        elif
        """
        if s==self.target: # the target, person should stand still.
            action_list = ['static']
        else: # other: person can choose to walk left or right.
            action_list = ['left', 'right']
        return action_list

    def next_state(self, s, a):
        """
        return the next state and reward by given the current state and action
        :param s: the current state
        :param a: the action which is took under the current state
        :return s_prime: the next state
        :return r: get r reward by taking action a under the current state s
        """
        if s==0 and a=='left':
            s_prime = s
        else:
            s_prime = s - 1 if a == 'left' else s + 1 if a == 'right' else s
        r = 1 if s_prime==self.target else 0
        return s_prime, r

    def random_pai(self, s):
        """
        Random Policy: pai(a|s)
        :param s: the index of person
        :return: the next action following by random policy
        """
        a = random.choice(self.action(s))
        return a


if __name__ == '__main__':
    walker = RandomWalk(6)
    walker.walk()
