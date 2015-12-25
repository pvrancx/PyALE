import numpy as np
import copy

from util.ALEFeatures import BasicALEFeatures,RAMALEFeatures
from agents.ALESarsaAgent import ALESarsaAgent, RAMALESarsaAgent

class TransitionTable(object):
    def __init__(self, max_steps, num_features, rng):
        self.max_steps = max_steps      # Memory size
        self.num_features = num_features    # State dimension tuple
        self.rng = rng                   

        # These are circular buffers; they wrap around when looking for next
        self.states = np.zeros(max_steps, num_features, dtype=np.uint8)
        self.actions = np.zeros(max_steps, dtype=np.uint8) # actually indices
        self.rewards = np.zeros(max_steps, dtype=np.float64)
        self.terminal = np.zeros(max_steps, dtype=np.bool)

        self.bottom = 0
        self.top = 0    # Points to free or at least usable index
        self.size = 0   # Will start wrapping once max_steps == size

    def add(self, state, action, reward, terminal):
        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = termina;

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        return self.size

    def sample(self, amount):
        # idx = np.array(self.rng.random_sample(amount) * self.size,
        #                dtype=np.uint32)
        states = np.zeros((amount, self.num_features), dtype=self.states.dtype)
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
            states[count] = self.states.take(current_index, axis=0,
                    mode='wrap')
            actions[count] = self.actions.take(current_index, mode='wrap')
            rewards[count] = self.rewards.take(current_index, mode='wrap')
            # terminal[count] = self.terminal.take(next_ind)
            next_states[count] = self.states.take(next_index, axis=0, mode='wrap')
            next_actions[counts] = self.actions.take(next_index, axis=0,
                    mode='wrap')

            count += 1

        return states, actions, rewards, next_states, next_actions
            

class ALEReplayAgent(RAMALESarsaAgent):
    def __init__(self, random_seed, replay_memory, replay_frequency, replay_times,
                 replay_size, **kwargs):
        super(ALEReplayAgent,self).__init__(**kwargs)
        self.name = 'replaySARSA'
        self.replay_memory = replay_memory
        self.replay_frequency = replay_frequency
        self.replay_times = replay_times
        self.replay_size = replay_size

        self.rng = np.random.RandomState(random_seed)  
        self.total_steps = 0

        # Don't use this for now
        # self.trace = None

    def agent_init(self, task_spec):
        super(ALEReplayAgent,self).agent_init(task_spec)
        self.transitions = TransitionTable(self.replay_memory,
                                           self.state_projector.num_features())
        self.total_steps = 0

    def agent_start(self, observation):
        super(ALEReplayAgent,self).agent_start(observation)
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
        samples = self.transitions.sample(self.replay_size if not
                                         self.replay_size is None 
                                         else len(self.transitions))
        repeats = 1 if not self.replay_size is None else self.replay_times 

        for _ in xrange(repeats):
            for i in xrange(len(samples)):
                state, action, reward, next_state, next_action = samples[i]
                n_rew = self.normalize_reward(reward)
                delta = n_rew - self.get_value(state, action, self.sparse)
                ns_values = self.get_all_values(,self.sparse)
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
    # TODO delegate argument parsing to other entity

    ### SARSA parameters ###
    parser.add_argument('--id', metavar='I', type=int, help='agent id',
                        default=0)
    parser.add_argument('--gamma', metavar='G', type=float, default=0.999,
                    help='discount factor')
    parser.add_argument('--alpha', metavar='A', type=float, default=0.5,
                    help='learning rate')
    parser.add_argument('--lambda_', metavar='L', type=float, default=0.9,
                    help='trace decay')
    parser.add_argument('--eps', metavar='E', type=float, default=0.05,
                    help='exploration rate')
    parser.add_argument('--savepath', metavar='P', type=str, default='.',
                    help='save path')  
    parser.add_argument('--no-traces', dest='no_traces', type=bool,
                        default=False, help='')
    parser.add_argument('--actions', metavar='C',type=int, default=None, 
                        nargs='*',help='list of allowed actions')

    ### Replay parameters ###
    parser.add_argument('--random_seed', metavar='Z', type=int,
                        default=None, help='Seed for random number generation')
    parser.add_argument('--replay_memory', metavar='M', type=int, 
                        default=10000, help='replay memory size')
    parser.add_argument('--replay_frequency', metavar='R', type=int,
                        default=100, help='replay frequency (T); if 0, replay '
                        + 'after every episode')
    parser.add_argument('--replay_times', metavar='N', type=int,
                        default=3, help='times to replay the database (N); ' +
                        + 'useless if --replay_size is specified')
    parser.add_argument('--replay_size', metavar='S', type=int, default=None,
                        help='amount of replayed transitions; if unspecified, ' + 
                        'replay_times*memory_size is used')

    args = parser.parse_args()

    act = None
    if not (args.actions is None):
        act = np.array(args.actions)


    AgentLoader.loadAgent(ALEReplayAgent(agent_id=args.id,
                                    alpha =args.alpha,
                                    lambda_=args.lambda_,
                                    eps =args.eps,
                                    gamma=args.gamma, 
                                    save_path=args.savepath,
                                    actions = act,
                                    random_seed = args.random_seed,
                                    replay_memory = args.replay_memory,
                                    replay_frequency = args.replay_frequency,
                                    replay_times = args.replay_times,
                                    replay_size = args.replay_size))
