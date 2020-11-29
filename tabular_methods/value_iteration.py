import numpy as np

from random_walk import RandomWalk


class ValueIteration(object):
    """
    -----------------------------   Pseudo Code   ------------------------------------------
        Initialize v(s), the hyperparameter theta
        While Loop [value evaluate]:
            delta <- 0
            For each state:
                v <- v(s)
                v(s) <- \max_a \sum{s',r} p(s',r|s,a)(r + gamma*v(s'))  [Bellman optimality equation]
                delta <- \max(delta, |v-v(s)|)
        Until delta < theta
        Output a policy: pai(a|s) <- \max_a \sum{s',r} p(s',r|s,a)(r + gamma*v(s'))  [Bellman optimality equation]
    """
    def __init__(self, env, gamma, theta):
        self.gamma = gamma
        self.theta = theta
        self.env = env
        self.state_num = self.env.road_len + 1  # include target state
        self.init()

    def init(self):
        """initialize v(s)"""
        self.v = np.zeros(self.state_num)

    def value_evaluate(self):
        """
        Value evaluate is the process of updating v(s) by using bellman optimality equation.
        bellman optimality equation is v(s) = \max_a \sum{s',r} p(s',r|s,a)(r + gamma*v(s'))
        """
        while True:
            v_old = self.v.copy()
            for s in range(self.state_num):
                max_q = 0
                for a in self.env.action(s):
                    q_val = self.q(s, a)
                    if max_q < q_val:
                        max_q = q_val
                self.v[s] = max_q
            print(self.v)

            delta = max(abs((v_old-self.v)))
            if delta < self.theta:
                break

    def q(self, s, a):
        """
        Definition q(s,a) = \sum{s',r} p(s',r|s,a)(r + gamma*v(s'))
        :param s: the current state
        :param a: the action which is choose under the current state
        :return: the value of q(s,a)
        """
        s_prime, r = self.env.next_state(s, a)
        q_val = r + self.gamma*self.v[s_prime]
        return q_val

    def get_policy(self):
        """get the optimality policy"""
        pai = {}
        for s in range(self.state_num-1):
            max_q = 0
            max_a = None
            for a in self.env.action(s):
                q_val = self.q(s, a)
                if max_q < q_val:
                    max_q = q_val
                    max_a = a
            pai[s] = max_a
        return pai


if __name__ == '__main__':
    env = RandomWalk(5)
    value_iter = ValueIteration(env, 0.5, 1e-8)
    value_iter.value_evaluate()
    best_pai = value_iter.get_policy()
    env.walk(best_pai)
