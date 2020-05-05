import torch
import numpy as np
from ddpg import DDPGAgent
from buffer import ReplayBuffer

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 512
GAMMA = 0.99
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgent:
	def __init__(self, state_size, action_size, num_agents, random_seed):
		self.agents = [DDPGAgent(state_size, action_size, random_seed) for _ in range(num_agents)]       
		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device, random_seed)
		self.t_step = 0
	def step_all(self, states, actions, rewards, next_states, dones):
		# Save experience in replay memory
		for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
			self.memory.add(state, action, reward, next_state, done)
		
		# Learn every UPDATE_EVERY time steps.
		self.t_step = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			# If enough samples are available in memory, get random subset and learn
			if len(self.memory) > BATCH_SIZE:
				for agent in self.agents:
					experiences = self.memory.sample()
					agent.learn(experiences, GAMMA)
	def act_all(self, multi_states):
		actions = [agent.act(np.expand_dims(states, axis=0)) for agent, states in zip(self.agents, multi_states)]
		return actions
	
	def save_weights_all(self):
		for index, agent in enumerate(self.agents):
			torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(index+1))
			torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(index+1))
	
	def reset_all(self):
		for agent in self.agents:
			agent.reset()