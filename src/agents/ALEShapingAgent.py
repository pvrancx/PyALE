# -*- coding: utf-8 -*-
import argparse
import copy
import matplotlib.pyplot as plt

from rlglue.agent import AgentLoader as AgentLoader
from util.image_processing import as_RGB

from util.ALEFeatures import BasicALEFeatures
from agents.ALESarsaAgent import ALESarsaAgent, BasicALESarsaAgent

import numpy as np

class ALEShapingAgent(BasicALESarsaAgent):
    @classmethod
    def register_with_parser(cls, parser):
        super(ALEShapingAgent, cls).register_with_parser(parser)
        parser.add_argument('--allow_negative_rewards', action='store_true', 
                             dest='allow_negative_rewards')
        parser.add_argument('--bonus_per_alien', type=float, default=10,
                            help='shaping bonus per alien killed')
        # parser.add_argument('--clear_alien_bonus', type=float, default=200)
        parser.add_argument('--laser_penalty', type=float, default=10,
                            help='shaping penalty when beneath laser')
        # Ship is 7-8 in width, every shielded pixel gets a bonus
        parser.add_argument('--shield_bonus', type=float, default=3,
                            help='shaping bonus when beneath shield')
        parser.add_argument('--lowest_enemy_penalty', type=float, default=2)
        parser.set_defaults(allow_negative_rewards=True)

    
    def __init__(self, args):
        super(ALEShapingAgent, self).__init__(args)
        self.allow_negative_rewards = args.allow_negative_rewards
        self.bonus_per_alien = args.bonus_per_alien
        self.laser_penalty = args.laser_penalty
        self.shield_bonus = args.shield_bonus
        self.lowest_enemy_penalty = args.lowest_enemy_penalty
        self.ship_width = None


    def agent_init(self,taskspec):
        super(ALEShapingAgent, self).agent_init(taskspec)
        self.shaping_data_filename = '/'.join((self.save_path, 'shaping.csv'))
        with open(self.shaping_data_filename, 'wb') as f:
            f.write(','.join(('alien_bonus', 'laser_penalty', 'shield_bonus',
                              'lowest_enemy_penalty')) + '\n')


    def agent_start(self, observation):
        action = super(ALEShapingAgent, self).agent_start(observation)
        self.outstanding_shaping_data = {
            'alien_bonus': [],
            'laser_penalty': [],
            'shield_bonus': [],
            'lowest_enemy_penalty': [],
        }
        self.last_num_enemies = 0
        self.alien_potential = 0
        self.last_potential = self.potential(self.as_frame(observation))
        return action

    def as_frame(self, observation):
        return self.get_frame_data(observation).reshape((210, 160))

    def space_ship_position(self, frame):
        # It's always in this specific strip
        space_ship_strip = frame[184:194,:]
        # space ship is either 194 or 196
        return (np.min(np.where((space_ship_strip==196) +
                                (space_ship_strip==194))[1]),
                np.max(np.where((space_ship_strip==196) +
                                (space_ship_strip==194))[1]))

    def is_spaceship_present(self, frame):
        space_ship_strip = frame[184:194,:]
        return np.any(space_ship_strip == 196)

    def num_enemies(self, frame):
        # Looks like aliens are either 18 or 20
        return np.sum((frame == 20) + (frame == 18))//37 #on average each enemy consists of 37 pixels
    def lowest_enemy(self, frame):
        return np.max(np.where(frame==20)[0]) #max:lower screen has higher row index

    def below_laser(self, frame):
        c1, c2 = self.space_ship_position(frame)
        # if 6 in np.unique(frame):
        # self.screenshot(frame)
        return np.any(
            frame[:,c1:c2+1] == 6 #+1, upper boundary is exclusive
        )

    def pixels_below_shield(self, frame):
        c1,c2 = self.space_ship_position(frame)
        return np.count_nonzero(
                    # It's either 52 or 53
                    np.any(frame[:,c1:c2+1] / 2 == 26,
                    axis=0) ) #add axis argument to check per column
        

    def potential(self, frame):
        # Reward for killing an alien is between say 5-30 for starters
        # Boost this up a bit by giving an extra 15 per alien killed
        if not self.is_spaceship_present(frame):
            alien_bonus = 0
            shield_bonus = 0
            laser_penalty = 0
        else:
            num_enemies = self.num_enemies(frame)
            # self.most_enemies = max(self.most_enemies, num_enemies)
            self.alien_potential = self.alien_potential + self.bonus_per_alien * max(0, num_enemies - self.last_num_enemies)
            self.last_num_enemies = num_enemies
            alien_bonus = self.alien_potential
            # Standing under a laser might be bad
            laser_penalty = - self.laser_penalty if self.below_laser(frame) else 0
            # Standing under a shield might be good. The more covered the better
            shielded_pixels = self.pixels_below_shield(frame)
            shield_bonus = self.shield_bonus * shielded_pixels

        # This one goes regardless of spaceship position
        # The lower the lowest enemy, the closer to doom
        lowest_enemy = self.lowest_enemy(frame)
        lowest_enemy_penalty = self.lowest_enemy_penalty * (120 - lowest_enemy)

        F = alien_bonus + shield_bonus + laser_penalty + lowest_enemy_penalty
        
        self.outstanding_shaping_data['alien_bonus'].append(alien_bonus)
        self.outstanding_shaping_data['shield_bonus'].append(shield_bonus)
        self.outstanding_shaping_data['laser_penalty'].append(laser_penalty)
        self.outstanding_shaping_data['lowest_enemy_penalty'].append(lowest_enemy_penalty)

        return F

    def agent_step(self,reward, observation):
        super(ALESarsaAgent,self).agent_step(reward, observation)
        phi_ns = self.get_phi(observation)
        # No negative shaping rewards
        current_potential = self.potential(self.as_frame(observation))
        F = self.gamma * (current_potential - self.last_potential)

        if not self.allow_negative_rewards:
            F = max(0, F)

        a_ns = self.step(reward,phi_ns)
        #log state data
        self.last_phi = copy.deepcopy(phi_ns)
        self.last_action = copy.deepcopy(a_ns)
        self.last_potential = current_potential
        
        return self.create_action(self.actions[a_ns])#create RLGLUE action

    def agent_end(self, reward):
        with open(self.shaping_data_filename, 'a') as f:
            for i in range(len(self.outstanding_shaping_data['alien_bonus'])):
                f.write(','.join(map(str, [
                    self.outstanding_shaping_data['alien_bonus'][i],
                    self.outstanding_shaping_data['laser_penalty'][i],
                    self.outstanding_shaping_data['shield_bonus'][i],
                    self.outstanding_shaping_data['lowest_enemy_penalty'][i]
                    ])) + '\n')
        super(ALEShapingAgent, self).agent_end(reward)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run Sarsa Agent')
    ALEShapingAgent.register_with_parser(parser)
    args = parser.parse_args()
    AgentLoader.loadAgent(ALEShapingAgent(args))
