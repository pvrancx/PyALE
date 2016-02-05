# -*- coding: utf-8 -*-
import argparse
import copy

from rlglue.agent import AgentLoader as AgentLoader

from util.ALEFeatures import BasicALEFeatures
from agents.ALESarsaAgent import ALESarsaAgent, BasicALESarsaAgent

import numpy as np

class ALEShapingAgent(BasicALESarsaAgent):

    @classmethod
    def register_with_parser(cls, parser):
        super(ALEShapingAgent, cls).register_with_parser(parser)
        parser.add_argument('--allow_negative_rewards', action='store_true', 
                             dest='allow_negative_rewards')
        parser.add_argument('--bonus_per_alien', type=float, default=15,
                            help='shaping bonus per alien killed')
        # parser.add_argument('--clear_alien_bonus', type=float, default=200)
        parser.add_argument('--laser_penalty', type=float, default=20,
                            help='shaping penalty when beneath laser')
        parser.add_argument('--shield_bonus', type=float, default=10,
                            help='shaping bonus when beneath shield')
        parser.set_defaults(allow_negative_rewards=True)

    
    def __init__(self, args):
        super(ALEShapingAgent, self).__init__(args)
        self.bonus_per_alien = 15
        self.last_observation = None
        self.last_num_enemies = 0


    def agent_start(self, observation):
        action = super(ALEShapingAgent, self).agent_start(observation)
        self.last_potential = self.potential(observation)
        return action

    def space_ship_position(self, frame):
        return (np.min(np.where(frame[184:194,:]==196)[1]),
                np.max(np.where(frame[184:194,:]==196)[1]))

    def num_enemies(self, frame):
        return np.sum(frame == 20)//37 #on average each enemy consists of 37 pixels

    def below_laser(self, frame):
        c1,c2 = self.space_ship_position(frame)
        return np.any(
            frame[:,c1:c2+1] == 4 #+1, upper boundary is exclusive
        )

    def below_shield(self, frame):
        c1,c2 = self.space_ship_position(frame)
        return np.all( #test if all columns contain shield
                    np.any(frame[:,c1:c2+1] == 52,
                    axis=0) ) #add axis argument to check per column

    def potential(self, frame):
        # Reward for killing an alien is between say 5-30 for starters
        # Boost this up a bit by giving an extra 15 per alien killed
        num_enemies = self.num_enemies(frame)
        self.most_enemies = max(self.most_enemies, num_enemies)
        alien_bonus = self.bonus_per_alien * (self.most_enemies - num_enemies)
        # Standing under a laser might be bad
        laser_penalty = - self.laser_penalty if self.below_laser(frame) else 0
        # Standing under a shield might be good
        shield_bonus = self.shield_bonus if self.below_shield(frame) else 0

        F = alien_bonus + shield_bonus + laser_penalty

        return F

    def agent_step(self,reward, observation):
        super(ALESarsaAgent,self).agent_step(reward, observation)
        phi_ns = self.get_phi(observation)
        # No negative shaping rewards
        current_potential = self.potential(observation)
        F = self.gamma * (current_potential - self.last_potential)

        if not self.allow_negative_rewards:
            F = max(0, F)

        a_ns = self.step(reward,phi_ns)
        #log state data
        self.last_phi = copy.deepcopy(phi_ns)
        self.last_action = copy.deepcopy(a_ns)
        self.last_potential = current_potential
        
        return self.create_action(self.actions[a_ns])#create RLGLUE action
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run Sarsa Agent')
    ALEShapingAgent.register_with_parser(parser)
    args = parser.parse_args()
    AgentLoader.loadAgent(ALEShapingAgent(args))
