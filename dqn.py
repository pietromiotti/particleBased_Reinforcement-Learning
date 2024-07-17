import torch
import torch.nn as nn
import numpy as np
import collections
from tensorboardX import SummaryWriter
from env import Env
from torch import nn

import random

random.seed(2)
BATCH_SIZE = 128
EPSILON_INITIAL = 1
EPSILON_FINAL = 0.02
EPSILON_DECAY_FINAL_STEP = 5000
REPLAY_BUFFER_CAPACITY = 200000
SYNC_NETWORKS_EVERY_STEP = 500
DISCOUNT_FACTOR = 0.999
LEARNING_RATE = 0.001

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# DQN NN
class DQN(nn.Module):
    def __init__(self, input_size, num_action):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(input_size, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 32)
        self.action = nn.Linear(32, num_action)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = self.action(x)
        return x

#Epsilon Greedy Implementation
class EpsilonGreedy:
    def __init__(self, start_value, final_value, final_step):
        self.start_value = start_value
        self.final_value = final_value
        self.final_step = final_step

    def decay(self, step):
        epsilon = 1 + step * (self.final_value - self.start_value) / self.final_step
        return max(self.final_value, epsilon)

#Replay Buffer for experience Replay
class ReplayBuffer:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, episode_step):
        self.buffer.append(episode_step)

    def sample(self, sample_size):
        # Note: replace=False makes random.choice O(n)
        indexes = np.random.choice(len(self.buffer), sample_size, replace=True)
        samples = [self.buffer[idx] for idx in indexes]
        return self.unroll(samples)

    def unroll(self, samples):
        states, actions, rewards, dones, next_states = [], [], [], [], []
        for episode_step in samples:
            states.append(episode_step[0])
            actions.append(episode_step[1])
            rewards.append(episode_step[2])
            dones.append(episode_step[3])
            next_states.append(episode_step[4])

        states = torch.FloatTensor(np.array(states, copy=False)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states, copy=False)).to(self.device)
        actions = torch.LongTensor(np.array(actions, copy=False)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards, copy=False)).to(self.device)
        dones = torch.BoolTensor(np.array(dones, copy=False)).to(self.device)
        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.buffer)

#Simulation class that handle the trainining
class Simulation:
    def __init__(self, env, buffer, net, target_net, epsilon_tracker, device, batch_size, sync_every, discount_factor,
                 learning_rate):
        self.env = env
        self.buffer = buffer
        self.net = net
        self.target_net = target_net
        self.epsilon_greedy = epsilon_tracker
        self.device = device
        self.batch_size = batch_size
        self.sync_steps = sync_every
        self.discount_factor = discount_factor
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(logdir='dqn/')
        self._reset()

    def _reset(self, maxpoint=0):
        self.state = self.env.reset(maxpoint)
        self.total_episode_reward = 0

    def train(self):
        step = 0
        episode_rewards = []
        self.iter = 0
        self.episode = 0
        self.total_reward = 0
        self.learn = 1
        while True:
            self.optimizer.zero_grad()
            epsilon = self.epsilon_greedy.decay(step)
            reward, done = self.take_step(epsilon)
            self.total_reward += reward*(self.discount_factor**self.iter)

            if len(self.buffer) < self.batch_size:
                print('\rFilling up the replay buffer...', end='')
                continue

            states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(states, actions, next_states, dones, rewards)
            self.learn = 1
            loss.backward()
            self.optimizer.step()
            self.sync_target_network(step)

            if done:
                print(self.total_reward)
                episode_rewards.append(self.total_reward)
                mean_reward = np.array(episode_rewards)[-100:].mean()

                self.print_progress(self.episode, loss.item(), self.total_reward, mean_reward, epsilon)

                if self.episode > self.env.episodes_dqn:
                    break

                self.total_reward = 0

            step += 1
            self.learn+=1;

    @torch.no_grad()
    def take_step(self, epsilon):
        state_t = torch.FloatTensor(np.array([self.state], copy=False)).to(self.device)
        q_actions = self.net(state_t)

        action = torch.argmax(q_actions, dim=1).item()

        if np.random.random() < epsilon:
            action = np.random.choice(self.env.num_action)

        next_state, reward, done, info = self.env.step(action)

        '''
        if (self.iter > self.env.iterationPerEpisode):
            print('ended here')
            with torch.no_grad():
                done = True
                reward = reward + self.discount_factor*q_actions[0][action].item()
        '''

        if not info:
            self.buffer.append([self.state, action, reward, done, next_state])

        if (self.iter > self.env.iterationPerEpisode):
            print('ended here')
            done = True

        if done:
            self.iter = 0
            self.episode+=1
            self._reset()

        elif info:
            self.iter = 0
            self.episode += 1
            self._reset()
        else:
            self.state = next_state
            self.iter = self.iter + 1

        return reward, done

    def compute_loss(self, states, actions, next_states, dones, rewards):
        state_q_all = self.net(states)
        state_q_taken_action = state_q_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_q_all = self.target_net(next_states)
            next_state_q_max = torch.max(next_state_q_all, dim=1)[0]
            next_state_q_max[dones] = 0

            state_q_expected = rewards + self.discount_factor * next_state_q_max
            state_q_expected = state_q_expected.detach()
            # LOSS Q-Learning
        return nn.functional.mse_loss(state_q_expected, state_q_taken_action)

    #Syncronyze the target network
    def sync_target_network(self, step):
        if step % self.sync_steps:
            self.target_net.load_state_dict(self.net.state_dict())

    #print the progress of the network
    def print_progress(self, step, loss, totalreword_epi, mean_reward, epsilon):

        self.writer.add_scalar('Episode Reward', totalreword_epi, step)
        self.writer.add_scalar('Reward', mean_reward, step)
        self.writer.add_scalar('loss', loss, step)
        self.writer.add_scalar('epsilon', epsilon, step)

    # in order to evaluate the policy I start from the bottom (with y coordinate between 1.5 and 2) and try to arrive at the end within
    # a reasonable time.
    def evalPolicy(self, episodes):
        rewardTotal = 0
        with torch.no_grad():
            for i in range(episodes):
                state = self.env.reset(2)
                rewardepi = 0
                currentiter = 0
                finish = False
                while not finish:
                    state_t = torch.FloatTensor(np.array([state], copy=False)).to(self.device)
                    q_actions = self.net(state_t)
                    action = torch.argmax(q_actions, dim=1).item()

                    state, reward, done, info = self.env.step(action)
                    rewardepi = rewardepi + reward*(self.discount_factor**currentiter)
                    currentiter+=1
                    if done or (currentiter >self.env.iterationPerEpisode):
                        finish = True
                rewardTotal += rewardepi
            avg = rewardTotal/episodes
        return avg



if __name__ == '__main__':
    env = Env(continuous=False)
    buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY, device=DEVICE)
    net = DQN(env.num_state, env.num_action).to(DEVICE)
    target_net = DQN(env.num_state,  env.num_action).to(DEVICE)
    epsilon_tracker = EpsilonGreedy(start_value=EPSILON_INITIAL, final_value=EPSILON_FINAL,
                                    final_step=EPSILON_DECAY_FINAL_STEP)
    simulation = Simulation(env=env, buffer=buffer, net=net, target_net=target_net, epsilon_tracker=epsilon_tracker,
                      device=DEVICE, batch_size=BATCH_SIZE, sync_every=SYNC_NETWORKS_EVERY_STEP, discount_factor=DISCOUNT_FACTOR,learning_rate=LEARNING_RATE)

    simulation.train()
    torch.save(net.state_dict(), './netstate_dqn_2_doors_1_obstacle')
    torch.save(target_net.state_dict(), './targetnetstate_dqn_2_doors_1_obstacle')
