from rlglue.agent import AgentLoader as AgentLoader

import numpy as np
import copy
import argparse

from util.ALEFeatures import BasicALEFeatures,RAMALEFeatures
from agents.ALESarsaAgent import ALESarsaAgent, RAMALESarsaAgent

class TransitionTable(object):
    def __init__(self, max_steps, num_features, rng, sparse=False,
                 buffer_size=10**9, buffer_growth=5*10**7, 
                 max_buffer_size=2*10**9):
        self.max_steps = max_steps      # Memory size
        self.num_features = num_features    # State dimension tuple
        self.rng = rng                   
        self.sparse = sparse

        # TODO question: Here I create a huge numpy buffer. I then fill it with
        # arrays. However, the arrays were already allocated. Does numpy copy
        # over the values from the old arrays? I think so. I might instead want
        # a large array of pointers.

        # These are circular buffers; they wrap around when looking for next
        if sparse:
            # In case of sparsity, arrays are of inequal length. 
            # Just pad the phis. That's easiest.
            self.state_lengths = np.zeros(max_steps, dtype=np.uint32)
        self.states = np.zeros((max_steps, num_features), dtype=np.uint8)
        self.actions = np.zeros(max_steps, dtype=np.uint8) # actually indices
        self.rewards = np.zeros(max_steps, dtype=np.float64)
        self.terminal = np.zeros(max_steps, dtype=np.bool)

        self.bottom = 0
        self.top = 0    # Points to free or at least usable index
        self.size = 0   # Will start wrapping once max_steps == size

    def add(self, state, action, reward, terminal):
        if self.sparse:
            self.state_lengths[self.top] = state.size
            state = np.pad(state, (0, self.num_features - state.size),
                                  mode='constant')
        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        return self.size

    def sample(self, amount):
        # States will be a bunch of unpadded frames in case of sparsity
        states = np.empty(amount, dtype=object)
        actions = np.zeros(amount, dtype=self.actions.dtype)
        rewards = np.zeros(amount, dtype=self.rewards.dtype)
        # terminal = np.zeros(amount, dtype=self.terminal.dtype)
        next_states = np.zeros_like(states)
        next_actions = np.zeros_like(actions)

        count = 0
        while count < amount:
            # Exclude the last state in the circular buffer because it doesn't
            # have a next state yet; it's the last state
            current_index = self.rng.randint(self.bottom, self.bottom + self.size - 1)
            next_index = current_index + 1

            # If the state is terminal, then next state will belong to a
            # different episode. Discard and try anew.
            if self.terminal.take(current_index, mode='wrap'):
                continue

            # All good, add it to response
            state = self.states.take(current_index, axis=0, mode='wrap')
            if self.sparse: # Unpad the state in case of sparsity
                state = state[:self.state_lengths.take(current_index, mode='wrap')]
            states[count] = state
            actions[count] = self.actions.take(current_index, mode='wrap')
            rewards[count] = self.rewards.take(current_index, mode='wrap')
            # terminal[count] = self.terminal.take(next_ind)
            next_states[count] = self.states.take(next_index, axis=0, mode='wrap')
            next_actions[count] = self.actions.take(next_index, axis=0,
                    mode='wrap')

            count += 1

        return states, actions, rewards, next_states, next_actions
            

class ALEReplayAgent(RAMALESarsaAgent):
    @classmethod
    def register_with_parser(cls, parser):
        super(ALEReplayAgent, cls).register_with_parser(parser)
        parser.add_argument('--replay_memory', type=int, 
                            default=10000, help='replay memory size')
        parser.add_argument('--replay_frequency', type=int,
                            default=100, help='replay frequency (T); if 0, replay '
                            + 'after every episode')
        parser.add_argument('--replay_times', type=int,
                            default=3, help='times to replay the database (N); ' +
                            'useless if --replay_size is specified')
        parser.add_argument('--replay_size', type=int, default=None,
                            help='amount of replayed transitions; if unspecified, ' + 
                            'replay_times*memory_size is used')


    # def __init__(self, replay_memory, replay_frequency, replay_times,
    #              replay_size, **kwargs):
    #     super(ALEReplayAgent,self).__init__(**kwargs)
    #     self.name = 'replaySARSA'
    #     self.replay_memory = replay_memory
    #     self.replay_frequency = replay_frequency
    #     self.replay_times = replay_times
    #     self.replay_size = replay_size

    #     self.total_steps = 0

    def __init__(self, args):
        super(ALEReplayAgent,self).__init__(args)
        self.name = 'replaySARSA'
        self.replay_memory = args.replay_memory
        self.replay_frequency = args.replay_frequency
        self.replay_times = args.replay_times
        self.replay_size = args.replay_size

        self.total_steps = 0


    def agent_init(self, task_spec):
        super(ALEReplayAgent,self).agent_init(task_spec)
        self.transitions = TransitionTable(self.replay_memory,
                                           self.state_projector.num_features(),
                                           self.rng, self.sparse)
        self.total_steps = 0

    def agent_start(self, observation):
        return super(ALEReplayAgent,self).agent_start(observation)
        # self.trace = None

    def step(self, reward, phi_ns=None):
        # The reward is from previous state-action
        # TODO figure out if traces are useful in this context
        self.update_trace(self.last_phi, self.last_action)
        # Add sample to the transition database
        terminal = phi_ns is None
        self.transitions.add(self.last_phi, self.last_action, reward, terminal)

        self.total_steps += 1
        # Work some magic every so often
        if self.replay_frequency != 0 and self.total_steps % self.replay_frequency == 0:
            self.replay()

        action_idx = None
        if not phi_ns is None:
            ns_values = self.get_all_values(phi_ns,self.sparse)
            action_idx = self.select_action(ns_values)
        return action_idx

    def agent_end(self, reward):
        super(ALEReplayAgent, self).agent_end(reward)
        if self.replay_frequency == 0:
            self.replay()

    def replay(self):
        """ SARSA-Replay-Samples """
        sample_size = self.replay_size if not self.replay_size is None else len(self.transitions)
        samples = self.transitions.sample(sample_size)
        states, actions, rewards, next_states, next_actions = samples
        repeats = 1 if not self.replay_size is None else self.replay_times 

        for _ in xrange(repeats):
            for i in xrange(sample_size):
                state, action, reward, next_state, next_action = states[i], \
                    actions[i], rewards[i], next_states[i], next_actions[i]
                n_rew = self.normalize_reward(reward)
                # assert np.unique(state), 'state contains duplicate values'
                delta = n_rew - self.get_value(state, action, self.sparse)
                assert not np.any(np.isnan(delta), np.isinf(delta)), \
                        'delta is nan or infinite: %s' % str(delta)
                ns_values = self.get_all_values(next_state, self.sparse)
                # Here's the difference with Q-learning: next_action is used
                delta += self.gamma*ns_values[next_action]
                # Normalize alpha with # of active features
                alpha = self.alpha / float(np.sum(state!=0.))
                # TODO I might be missing out on something, compare formula
                # Maybe trace made up for the fact that a factor is missing
                self.theta += alpha * delta * self.trace
            
    def create_projector(self):
        return RAMALEFeatures()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run Sarsa Replay Agent')
    ALEReplayAgent.register_with_parser(parser)
    args = parser.parse_args()
    AgentLoader.loadAgent(ALEReplayAgent(args))
