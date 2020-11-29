import numpy as np
import random

from random_walk import RandomWalk


class MonteCarlo(object):
    """
    the first-visit MC of on-policy method
    ---------------------------------   Pseudo Code   --------------------------------------
    Initialize pai(epsilon-greedy policy), q(s,a) and returns(s,a)
    While Loop with each episode:
        generate a chain by pai: S_0, A_0, R_1, S_1, A_1, ..., S_{T-1}, A_{T-1}, R_T, S_T
        G <- 0
        For t = T-1, T-2, ..., 0:
            G <- gamma*G + R_{t+1}
            if (S_t, A_t) is the first visit:
                append G into returns(S_t,A_t)
                q(S_t, A_t) <- average(returns(S_t,A_t))
                A^* <- \argmax_a q(S_t,a)
        For each action in A(S_t):  [A(S_t) is the set of correct actions under the state S_t]
            pai(a|S_t) = 1-epsilon+epsilon/|A(S_t)|  if a=A^*
            pai(a|S_t) = epsilon/|A(S_t)|            if a<>A^*
    """
    def __init__(self, walker, gamma, epsilon):
        self.walker = walker
        self.gamma = gamma
        self.epsilon = epsilon
        self.init()

    def init(self):
        """Initialize q table and returns list"""
        self.q = {}
        self.returns = {}
        for s in range(self.walker.road_len+1):
            self.q[s] = self.q.get(s, {})
            self.returns[s] = self.returns.get(s, {})
            for a in self.walker.action(s):
                self.q[s][a] = self.q[s].get(a, 0)
                self.returns[s][a] =self.returns.get(a, [])

    def is_first_visit(self, t):
        """
        check whether (s, a) is the first visit in chain
        :param t: the t-th step
        :return: boolean
        """
        s_t, a_t = [self.walker.chain['S'][t], self.walker.chain['A'][t]]
        if self.visit.index([s_t, a_t]) < t:    # the earliest time is smaller than t
            return False
        else:
            return True

    def epsilon_greedy_policy(self, s):
        """
        epsilon-greedy policy:
            if random.random > epsilon (default:0.9):
                a = random.choice(action_list)
            else:
                a = argmax_a q(s,a)
        :param s: the current state
        :return: action which is choose by policy
        """
        if random.random() > 1-self.epsilon+self.epsilon/len(self.walker.action(s)): # random
            a = random.choice(list(self.q[s].keys()))
        else: # greedy
            if np.diff(np.array(list(self.q[s].values()))).sum() == 0: ## if q(s,a) is the same,choose action randomly
                a = random.choice(list(self.q[s].keys()))
            else:
                q_sort = sorted(self.q[s].items(), key=lambda x: -x[1])
                a = q_sort[0][0]
        return a

    def learn(self, max_episode):
        """
        the main part of MC which is the process for learning a best epsilon-greedy policy
        :param max_episode: the maximum of episode number
        """
        episode_num = 0
        while True:
            episode_num += 1
            pai = self.epsilon_greedy_policy
            self.walker.walk(pai)   # generate a chain
            self.visit = [[s, a] for s, a in zip(list(self.walker.chain["S"].values())[:-1], list(self.walker.chain["A"].values()))]
            T = max(self.walker.chain['S'])
            g = 0
            for t in range(T-1, -1, -1):
                g = self.gamma*g + self.walker.chain['R'][t+1]
                s_t, a_t = self.walker.chain['S'][t], self.walker.chain['A'][t]
                if self.is_first_visit(t):
                    self.returns[s_t][a_t].append(g)
                    self.q[s_t][a_t] = sum(self.returns[s_t][a_t]) / len(self.returns[s_t][a_t])
            if episode_num >= max_episode:
                break


if __name__ == '__main__':
    walker = RandomWalk(3)
    fv_mc = MonteCarlo(walker, 0.5, 0.9)
    fv_mc.learn(10)