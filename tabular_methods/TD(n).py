import random
import numpy as np

from random_walk import RandomWalk


class SARSAStepN(object):

    def init(self):
        """Initialize q table"""
        self.q = {}  # state: action
        for s in range(self.env.road_len + 1):
            self.q[s] = self.q.get(s, {})
            for a in self.env.action(s):
                self.q[s][a] = self.q[s].get(a, 0)

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
        if random.random() < self.epsilon:  # random
            a = random.choice(list(self.q[s].keys()))
        else:  # greedy
            if np.diff(np.array(list(self.q[s].values()))).sum() == 0:  ## if q(s,a) is the same,choose action randomly
                a = random.choice(list(self.q[s].keys()))
            else:
                q_sort = sorted(self.q[s].items(), key=lambda x: -x[1])
                a = q_sort[0][0]
        return a

    def learn(self, max_episode):
        raise NotImplementedError


class SARSAStepNOnPolicy(SARSAStepN):
    """
    the advantage:
    for example
        a <-> b <-> c <-> d -> target
    in the first episode
    - TD(0): only update q(d,action)
    - TD(1): can update q(c,action) and q(d,action)
    - and so on
    ----------------------------   Pseudo Code   -----------------------------------
    Initialize q(s,a) and a epsilon-greedy policy pai, hyperparameter alpha, epsilon, step n
    While loop with episodes:
        initialize and save state S_0
        select and save an action A_0 by pai
        T <- inf
        for t = 0, 1, 2, ...:
            if t < T:
                take action A_t
                get and save the reward R_{t+1} and the next state S_{t+1}
                if S_{t+1} is the target:
                    T <- t+1
                else:
                    select and save an action A_{t+1} by pai
            tao <- t-n+1
            if tao >= 0:
                G <- \sum_{i=tao+1}^{min(tao+n,T)} gamma^{i-tao-1} R_i
                if tao+n<T, then G <- G + gamma^n Q(S_{tao+n}, A_{tao+n})
                Q(S_{tao}, A_{tao}) <- Q(S_{tao}, A_{tao}) + alpha*(G - Q(S_{tao}, A_{tao}))
                if in the process of learning policy, make sure the policy is epsilon-greedy
        Until tao = T-1
    """

    def __init__(self, env, gamma, epsilon, n, alpha):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n = n
        self.init()

    def learn(self, max_episode):
        episode_num = 0
        while True:
            episode_num += 1
            self.chain = {
                'S': {},
                'A': {},
                'R': {}
            }
            self.chain['S'][0] = 0
            self.env.walk_onestep(self.chain['S'][0])
            self.chain['A'][0] = self.epsilon_greedy_policy(self.chain['S'][0])
            T = np.inf
            t = 0
            while True:
                if t < T:
                    self.chain['S'][t+1], self.chain['R'][t+1] = self.env.next_state(self.chain['S'][t], self.chain['A'][t])
                    self.env.walk_onestep(self.chain['S'][t+1])
                    if self.chain['S'][t+1] == self.env.target:
                        T = t+1
                    else:
                        self.chain['A'][t+1] = self.epsilon_greedy_policy(self.chain['S'][t+1])
                tao = t-self.n+1
                if tao >= 0:
                    g = sum([self.gamma**(i-tao-1)*self.chain['R'][i] for i in range(tao+1, min([tao+self.n, T])+1)])
                    if tao+self.n < T:
                        g += self.gamma**self.n * self.q[self.chain['S'][tao+self.n]][self.chain['A'][tao+self.n]]
                    self.q[self.chain['S'][tao]][self.chain['A'][tao]] += self.alpha*(g - self.q[self.chain['S'][tao]][self.chain['A'][tao]])
                t += 1
                if tao == T-1:
                    break
            if episode_num >= max_episode:
                break


class SARSAStepNOffPolicy(SARSAStepN):
    """
    ----------------------------   Pseudo Code   -----------------------------------
    -- the content in "<>"  is different from on-policy
    Initialize q(s,a) and a epsilon-greedy policy pai <and an arbitrary behavior policy b>, hyperparameter alpha, epsilon, step n
    While loop with episodes:
        initialize and save state S_0
        <select and save an action A_0 by b>
        T <- inf
        for t = 0, 1, 2, ...:
            if t < T:
                take action A_t
                get and save the reward R_{t+1} and the next state S_{t+1}
                if S_{t+1} is the target:
                    T <- t+1
                else:
                    <select and save an action A_{t+1} by b>
            tao <- t-n+1
            if tao >= 0:
                <pho <- \sum_{i=tao+1}^{min(tao+n-1,T-1} pai(A_{i}|S_{i}) / b(A_{i}|S_{i})>
                G <- \sum_{i=tao+1}^{min(tao+n,T)} gamma^{i-tao-1} R_i
                if tao+n<T, then G <- G + gamma^n Q(S_{tao+n}, A_{tao+n})
                <Q(S_{tao}, A_{tao}) <- Q(S_{tao}, A_{tao}) + alpha*pho*(G - Q(S_{tao}, A_{tao}))>
                if in the process of learning policy, make sure the policy is epsilon-greedy
        Until tao = T-1
    """

    def __init__(self, env, gamma, epsilon, n, alpha):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n = n
        self.init()

    def arbitrary_policy(self, s):
        """
        arbitrary policy
        :param s: the current state
        :return: action which is choose by arbitrary policy
        """
        return self.env.random_pai(s)

    def b(self, s, a):
        """
        calculate b(a|s)
        :param s: state
        :param a: action
        :return: b(a|s)
        """
        return 1/len(self.env.action(s))

    def pai(self, s, a):
        """
        calculate pai(a|s)
        :param s: state
        :param a: action
        :return: pai(a|s)
        """
        if np.diff(np.array(list(self.q[s].values()))).sum() == 0:  # not random, but q(s,a) is the same for all of a
            prob = 1/2
        elif a == sorted(self.q[s].items(), key=lambda x: -x[1])[0][0]:
            prob = (1-self.epsilon)/2 + self.epsilon
        else:
            prob = (1-self.epsilon)/2
        return prob

    def learn(self, max_episode):
        episode_num = 0
        while True:
            episode_num += 1
            self.chain = {
                'S': {},
                'A': {},
                'R': {}
            }
            self.w = {}
            self.chain['S'][0] = 0
            self.env.walk_onestep(self.chain['S'][0])   # show
            self.chain['A'][0] = self.arbitrary_policy(self.chain['S'][0])
            self.w[0] = self.pai(self.chain['S'][0], self.chain['A'][0]) / self.b(self.chain['S'][0], self.chain['A'][0])
            T = np.inf
            t = 0
            while True:
                if t < T:
                    self.chain['S'][t+1], self.chain['R'][t+1] = self.env.next_state(self.chain['S'][t], self.chain['A'][t])
                    self.env.walk_onestep(self.chain['S'][t+1])
                    if self.chain['S'][t+1] == self.env.target:
                        T = t+1
                    else:
                        self.chain['A'][t+1] = self.arbitrary_policy(self.chain['S'][t+1])
                        self.w[t+1] = self.pai(self.chain['S'][t+1], self.chain['A'][t+1]) / self.b(self.chain['S'][t+1], self.chain['A'][t+1])
                tao = t-self.n+1
                if tao >= 0:
                    pho = np.prod([self.w[i] for i in range(tao+1, min([tao+self.n-1, T-1])+1)])
                    g = sum([self.gamma**(i-tao-1)*self.chain['R'][i] for i in range(tao+1, min([tao+self.n, T])+1)])
                    if tao+self.n < T:
                        g += self.gamma**self.n * self.q[self.chain['S'][tao+self.n]][self.chain['A'][tao+self.n]]
                    self.q[self.chain['S'][tao]][self.chain['A'][tao]] += self.alpha*pho*(g - self.q[self.chain['S'][tao]][self.chain['A'][tao]])
                t += 1
                if tao == T-1:
                    break
            if episode_num >= max_episode:
                break


if __name__ == '__main__':
    env = RandomWalk(6)
    # sarsa_stepn = SARSAStepNOnPolicy(env, 0.9, 0.9, 3, 0.5)
    sarsa_stepn = SARSAStepNOffPolicy(env, 0.9, 0.1, 3, 0.5)
    sarsa_stepn.learn(3)
    print(sarsa_stepn.q)
