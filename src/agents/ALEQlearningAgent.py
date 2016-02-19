# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:03:10 2015

@author: pvrancx
"""

from rlglue.agent import AgentLoader as AgentLoader

import argparse

from util.ALEFeatures import BasicALEFeatures,RAMALEFeatures
from agents.ALESarsaAgent import ALESarsaAgent
from agents.BasicALEAgent import BasicALEAgent
from agents.RAMALEAgent import RAMALEAgent

import numpy as np


class ALEQlearningAgent(ALESarsaAgent):
    
    def __init__(self,**kwargs):
        #use full images and color mode
        super(ALEQlearningAgent,self).__init__(**kwargs)
        self.name='QLearning'
  
    def step(self,reward,phi_ns = None):
        n_rew = self.normalize_reward(reward)
        self.update_trace(self.phi,self.a)
        delta = n_rew - self.get_value(self.phi,self.a,self.sparse)
        a_ns = None
        greedy = True
        if not (phi_ns is None):
            ns_values = self.get_all_values(phi_ns,self.sparse)
            a_ns = self.select_action(ns_values)
            ns_value = np.max(ns_values)
            delta += self.gamma*ns_value
            #check if next action is greedy
            greedy = (ns_value == ns_values[a_ns])
        #normalize alpha with nr of active features
        alpha = self.alpha / float(np.sum(self.phi!=0.))
        self.theta+= alpha*delta*self.trace
        if not greedy:
            self.trace *= 0. #reset trace
        return a_ns  #a_ns is action index (not action value)


class BasicALEQLearningAgent(BasicALEAgent, ALEQLearningAgent):
    pass
    
class RAMALEQLearningAgent(RAMALEAgent, ALEQLearningAgent):
    pass
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run Sarsa Agent')
    parser.add_argument('--features', metavar='F', type=str, default='RAM',
                    help='features to use: RAM or BASIC')
    ALEQLearningAgent.register_with_parser(parser)
    args = parser.parse_args()
    if args.features == 'RAM':
        AgentLoader.loadAgent(RAMALEQLearningAgent(args))
    elif args.features == 'BASIC':
        AgentLoader.loadAgent(BasicALEQLearningAgent(args))
    else:
        raise Exception('unknown feature type')
