import random
import numpy as np
import time

from random_walk import RandomWalk


class SARSA(object):
    """
    On-Policy Method:
        SARSA -> (S_t, A_t, R_t, S_{t+1}, A_{t+1})
    ------------------------------   Pseudo Code   ---------------------------------
    Initialize q(s,a), hyperparameter epsilon, alpha and gamma
    While Loop with episodes:
        initialize s
        choose action a from epsilon-greedy policy in the state s [\max_a q(s,a) if random.random<epsilon else random.choice(action_list)]
        For each step in this episode:
            take action a and then get the next state s' and reward r
            choose action a' from epsilon-greedy policy in the next state s'
            q(s,a) <- q(s,a) + alpha*(r + gamma*q(s',a') - q(s,a))
            s <- s', a <- a'
        Until reach the target state
    """
    def __init__(self, env, epsilon, alpha, gamma):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.init()

    def init(self):
        """Initialize q table"""
        self.q = {}  # state: action
        for s in range(self.env.road_len+1):
            self.q[s] = self.q.get(s, {})
            for a in self.env.action(s):
                self.q[s][a] = self.q[s].get(a, 0)

    def epsilon_greedy_policy(self, s):
        """
        epsilon-greedy policy:
            if random.random < epsilon:
                a = random.choice(action_list)
            else:
                a = argmax_a q(s,a)
        :param s: the current state
        :return: action which is choose by policy
        """
        if random.random() < self.epsilon: # random
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
        the most important part of SARSA, including the update process of q table
        :param max_episode: the maximum of episode number
        """
        episode_num = 0
        while True:
            episode_num += 1
            # print('episode %s:' % episode_num)
            s = 0
            self.env.walk_onestep(s)
            a = self.epsilon_greedy_policy(s)
            while s < self.env.target:
                s_prime, r = self.env.next_state(s, a)
                a_prime = self.epsilon_greedy_policy(s_prime)
                self.q[s][a] = self.q[s][a] + self.alpha*(r + self.gamma*self.q[s_prime][a_prime] - self.q[s][a])
                s = s_prime
                a = a_prime
                self.env.walk_onestep(s)
            time.sleep(3)
            if episode_num >= max_episode:
                break


if __name__ == '__main__':
    env = RandomWalk(6)
    sarsa = SARSA(env, 0.1, 0.5, 1)
    start_time = time.time()
    sarsa.learn(10)
    end_time = time.time()
    print(end_time-start_time)

