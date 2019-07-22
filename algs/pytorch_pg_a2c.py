# Inspired by: 
#   http://www.tuananhle.co.uk/notes/dqn-pg-a2c.html
#   https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#actor-critic 
#   https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py 

from torch.autograd import Variable
import sys 
import collections
import gym
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# PG 
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        # pi(a|s)
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.Tanh(),
            nn.Linear(24, 48),
            nn.Tanh(),
            nn.Linear(48, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        return self.network(state)

class PGAgent():
    def __init__(self, state_size, action_size, lr, lr_decay):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.action_size = action_size
        if lr_decay is not None:
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, weight_decay=lr_decay)
        else:
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
    def act(self, state):
        action_probs = self.policy_network(Variable(torch.from_numpy(state).float().unsqueeze(0))).squeeze(0).data.numpy()
        return np.random.choice(self.action_size, p=action_probs)

    def learn(self, cumulative_reward, states, actions):
        action_log_probs = torch.log(self.policy_network(Variable(torch.from_numpy(states).float())))
        actions = Variable(torch.from_numpy(actions).long())
        self.optimizer.zero_grad()
        loss = -torch.sum(torch.gather(action_log_probs, dim=1, index=actions.unsqueeze(-1))) * cumulative_reward
        loss.backward()
        self.optimizer.step()

# A2C 
class Actor(nn.Module):
    """A simple actor-critic for discrete action space. 
    """
    def __init__(self, state_size, action_size, hidden_size=20):
        # Computes both p(a_t | s_t) and v(s_t)
        super(Actor, self).__init__()
        # Dedicated nets 
        self.actor =  nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size), 
            nn.Softmax(dim=1)
        )
        
    def forward(self, states):
        """Compute p(a|s) for each s in `states`. 
        """
        return self.actor(torch.from_numpy(states) if type(states) is np.ndarray else states )

class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=20):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,1)
        )
    def forward(self, states):
        return self.critic(states)

class A2CAgent(object):
    def __init__(self, state_size, action_size, gamma,
                 actor_lr=0.001, critic_lr=0.01, hidden_size=20, lr_decay =0.1):
        self.actor = Actor(state_size, action_size, hidden_size)
        self.critic = Critic(state_size, hidden_size)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        if lr_decay is None:
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr )
        else:
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=lr_decay)
            self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=lr_decay)

    def act(self, state):
        action_probs = self.actor(Variable(torch.from_numpy(state).float().unsqueeze(0)))
        return np.random.choice(self.action_size, p=action_probs.data.numpy().squeeze(0))   

    def learn(self, s,a,r, done, s_):
        v = self.critic(Variable(torch.from_numpy(s).float().unsqueeze(0)))
        if not done:
            v_next = self.critic(Variable(torch.from_numpy(s_).float().unsqueeze(0))) 
            td_error = r + self.gamma * v_next.detach() - v
        else:
            td_error = r - v
        v_loss = td_error.pow(2).sum()  
        a_loss = -td_error.detach() * self.actor(Variable(torch.from_numpy(s).float().unsqueeze(0)))[0][a].log()

        # Update critic
        self.critic_opt.zero_grad()
        v_loss.backward()
        self.critic_opt.step()

        # Update actor 
        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()
        return v_loss.item(), a_loss.item()

if __name__ == "__main__":
    num_episodes = 1000
    num_timesteps_max = 1000
    state_size = 4
    action_size = 2 
    # Playing A2C 
    torch.manual_seed(0)
    np.random.seed(0)
    env_seed = 0
    env = gym.make('CartPole-v0')
    env.seed(env_seed)
    env = env.unwrapped # Reset the maximum time steps 
    actor_lr = 0.001 
    critic_lr = 0.01 
    lr_decay = None # Set None; otherwise it does not work!
    hidden_size = 20 
    
    gamma = 0.9
    ac_cumulative_rewards = np.zeros([num_episodes])  
    agent = A2CAgent(state_size, action_size, gamma,actor_lr, critic_lr, hidden_size, lr_decay = lr_decay) 

    for episode_idx in range(num_episodes):
        sys.stdout.write('ep: %d/%d\r'%(episode_idx +1, num_episodes))
        sys.stdout.flush()
        state = env.reset() # Reset the game 
        t = 0 
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            ac_cumulative_rewards[episode_idx] += reward
            # if done: reward = -20
            v_loss, a_loss = agent.learn(state, action, reward, done, next_state)
            state = next_state 
            t += 1
            if done or t >= num_timesteps_max: 
                print("episode:", episode_idx, "  reward:", ac_cumulative_rewards[episode_idx])
                break
            
    # Playing PG 
    torch.manual_seed(0)
    np.random.seed(0)
    env_seed = 0
    env = gym.make('CartPole-v0')
    env.seed(env_seed)
    env = env.unwrapped # Reset the maximum time steps
    lr = 0.001 
    lr_decay = 0.01
    
    gamma = 0.99 # Set to 0.9 does not work
    pg_cumulative_rewards = np.zeros([num_episodes])  
    agent = PGAgent(state_size, action_size, lr, lr_decay) 

    for episode_idx in range(num_episodes):
        sys.stdout.write('ep: %d/%d\r'%(episode_idx +1, num_episodes))
        sys.stdout.flush()
        state = env.reset() # Reset the game 
        t = 0 
        states = []
        actions = []
        discounted_return = 0.
        while True:
            action = agent.act(state)
            states.append(state)
            actions.append(action) 
            
            state, reward, done, _ = env.step(action)
            pg_cumulative_rewards[episode_idx] += reward
            discounted_return += gamma**t * reward
            t += 1
            if done or t >= num_timesteps_max: 
                print("episode:", episode_idx, "  reward:", pg_cumulative_rewards[episode_idx])
                agent.learn(discounted_return, np.array(states), np.array(actions))
                break
    
    # Plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(3, 2)
    ax.plot(range(num_episodes), pg_cumulative_rewards, label='PG')
    ax.plot(range(num_episodes), ac_cumulative_rewards, label='A2C')
    ax.axhline(195, color='black', linestyle='dashed', label='solved')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Cumulative Reward')
    ax.set_xlabel('Episode')
    ax.set_title('CartPole-v0')
    filenames = ['pg_a2c.png']

    for filename in filenames:
        fig.savefig(filename, bbox_inches='tight', dpi=200)
        print('Saved to {}'.format(filename))