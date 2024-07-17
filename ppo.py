import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import numpy as np
import os
from env import Env

writer = SummaryWriter(logdir='ppo_2/')

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.new_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.lastStep = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.new_states[:]
        del self.lastStep[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


def minibatch_number(size, batch_size):
    return int(np.ceil(size / batch_size))

def minibatch_generator(batch_size, *dataset):
    size = len(dataset[0])
    num_batches = minibatch_number(size, batch_size)
    indexes = np.arange(0, size, 1)
    np.random.shuffle(indexes)
    batches = [(i * batch_size, min(size, (i + 1) * batch_size))
               for i in range(0, num_batches)]

    for (batch_start, batch_end) in batches:
        batch = []
        for i in range(len(dataset)):
            batch.append(dataset[i][indexes[batch_start:batch_end]])
        yield batch


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.batch_size = 128
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):

        rewards = []

        discounted_reward = 0
        for reward, is_terminal, state, last in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals),
                                              reversed(self.buffer.states), reversed(self.buffer.lastStep),):
            if last:
                value = self.policy.critic(state.squeeze())
                discounted_reward = value
            elif is_terminal:
                discounted_reward = 0

            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            for old_states_batch, old_action_batch, old_logprobs_batch, rewards_batch in minibatch_generator(self.batch_size, old_states, old_actions, old_logprobs, rewards):

                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_batch, old_action_batch)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs_batch.detach())

                # Finding Surrogate Loss
                advantages = rewards_batch - state_values.detach()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                surr1 =torch.squeeze(surr1)
                surr2 = torch.squeeze(surr2)
                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, torch.squeeze(rewards_batch)) - 0.01 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

def main():

    # creating environment
    env = Env(continuous=False)

    log_interval = 27  # print avg reward in the interval
    max_episodes = env.episodes_ppo  # max training episodes
    max_timesteps = 1000  # max timesteps in one episode

    update_timestep = 4000  # update policy every n timesteps
    K_epochs = 50  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.999  # discount factor

    lr_actor = 0.001  # parameters for Adam optimizer
    lr_critic = 0.001  # parameters for Adam optimizer

    state_dim = env.num_state
    action_dim = env.num_action


    ppo = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):

        lenght = 0
        running_epi = 0

        state = env.reset()

        last_step = 0
        for t in range(max_timesteps):
            time_step += 1

            action = ppo.select_action(state)

            state, reward, done, info = env.step(action)
            ppo.buffer.new_states.append(torch.tensor(state, dtype=torch.float32))

            if (t == max_timesteps - 1):
                last_step = 1

            ppo.buffer.rewards.append(reward)
            ppo.buffer.lastStep.append(last_step)
            ppo.buffer.is_terminals.append(done)

            if time_step % update_timestep == 0:
                ppo.update()
                time_step = 0

            running_reward += reward
            running_epi += reward*(gamma**time_step)
            lenght += 1

            if done:
                break
        rewardEpi = running_epi
        avg_length += t

        writer.add_scalar('Reward', rewardEpi, i_episode)

        if i_episode % 50 == 0:
            torch.save(ppo.policy.state_dict(), 'ppo_state_v2')

        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = ((running_reward / log_interval))
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
