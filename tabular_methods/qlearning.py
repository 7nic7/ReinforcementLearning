import time
import numpy as np
import random

from random_walk import RandomWalk


class QLearning(object):
    """
    Off-Policy Method:
        Q Learning: learning with Q table.
    ------------------------------   Pseudo Code   --------------------------------
    Initialize q(s,a), hyperparameter epsilon, alpha and gamma
    While Loop with episodes:
        initialize s
        For each step in this episode:
            choose action a from epsilon-greedy policy under the state s
            take action a and then get the next state s' and reward r
            q(s,a) <- q(s,a) + alpha*(r + gamma*\max_a q(s',a) - q(s,a))  [this step is different from SARSA]
            s <- s'
        Until reach the target state
    """

    def __init__(self, env, epsilon, alpha, gamma):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.init()

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
        else: # greedy
            if np.diff(np.array(list(self.q[s].values()))).sum() == 0:  ## for a, q(s,a) is the same
                a = random.choice(list(self.q[s].keys()))
            else:
                q_sort = sorted(self.q[s].items(), key=lambda x: -x[1])
                a = q_sort[0][0]
        return a

    def learn(self, max_episode):
        """
        the most important part of Q learning, including the update process of q table
        :param max_episode: the maximum of episode number
        """
        episode_num = 0
        while True:
            episode_num += 1
            # print('episode %s:' % episode_num)
            s = 0
            self.env.walk_onestep(s)
            while s < self.env.target:
                a = self.epsilon_greedy_policy(s)
                s_prime, r = self.env.next_state(s, a)
                self.q[s][a] = self.q[s][a] + self.alpha * (r + self.gamma * max(list(self.q[s_prime].values())) - self.q[s][a])
                s = s_prime
                self.env.walk_onestep(s)
            time.sleep(3)
            if episode_num >= max_episode:
                break


class DoubleQLearning(object):
    """
    Double Q-learning can solve the problem of maximization bias
    ------------------------------   Pseudo Code   --------------------------------
    Initialize q1(s,a), q2(s,a), hyperparameter epsilon, alpha and gamma
    While Loop with episodes:
        initialize s
        For each step in this episode:
            choose action a from s using policy derived from q1 and q2 (e.g., epsilon-greedy in q1+q2)
            take action a and then get the next state s' and reward r
            with 0.5 probability:
                q1(s,a) <- q1(s,a) + alpha*(r + gamma*q2(s',\argmax_a' q1(s',a')) - q1(s,a))
            else:
                q2(s,a) <- q2(s,a) + alpha*(r + gamma*q1(s',\argmax_a' q2(s',a')) - q2(s,a))
            s <- s'
        Until reach the target state
    """

    def __init__(self, env, epsilon, alpha, gamma):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.init()

    def init(self):
        """Initialize q table"""
        self.q1 = {}  # state: action
        self.q2 = {}
        for s in range(self.env.road_len + 1):
            self.q1[s] = self.q1.get(s, {})
            self.q2[s] = self.q2.get(s, {})
            for a in self.env.action(s):
                self.q1[s][a] = self.q1[s].get(a, 0)
                self.q2[s][a] = self.q2[s].get(a, 0)

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
            a = random.choice(list(self.q1[s].keys()))
        else: # greedy
            q_sum = {a_: q1+q2 for (a_, q1), q2 in zip(self.q1[s].items(), self.q2[s].values())}
            if np.diff(np.array(list(q_sum.values()))).sum() == 0:  ## for a, q(s,a) is the same
                a = random.choice(list(q_sum.keys()))
            else:
                q_sort = sorted(q_sum.items(), key=lambda x: -x[1])
                a = q_sort[0][0]
        return a

    def learn(self, max_episode):
        """
        the most important part of Q learning, including the update process of q table
        :param max_episode: the maximum of episode number
        """
        episode_num = 0
        while True:
            episode_num += 1
            # print('episode %s:' % episode_num)
            s = 0
            self.env.walk_onestep(s)
            while s < self.env.target:
                a = self.epsilon_greedy_policy(s)
                s_prime, r = self.env.next_state(s, a)
                if random.random() < 0.5:
                    argmaxa_q1 = sorted(self.q1[s_prime].items(), key=lambda x: -x[1])[0][0]
                    self.q1[s][a] += self.alpha * (r + self.gamma * self.q2[s_prime][argmaxa_q1] - self.q1[s][a])
                else:
                    argmaxa_q2 = sorted(self.q2[s_prime].items(), key=lambda x: -x[1])[0][0]
                    self.q2[s][a] += self.alpha * (r + self.gamma * self.q1[s_prime][argmaxa_q2] - self.q2[s][a])
                s = s_prime
                self.env.walk_onestep(s)
            time.sleep(3)
            if episode_num >= max_episode:
                break


if __name__ == '__main__':
    env = RandomWalk(6)
    # ql = QLearning(env, 0.9, 0.5, 1)
    double_ql = DoubleQLearning(env, 0.1, 0.5, 1)
    start_time = time.time()
    # ql.learn(10)
    double_ql.learn(10)
    end_time = time.time()
    print(end_time - start_time)

