# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:03:10 2015

@author: pvrancx
"""


from rlglue.agent import AgentLoader as AgentLoader



import argparse

from util.ALEFeatures import BasicALEFeatures,RAMALEFeatures
from agents.ALESarsaAgent import ALESarsaAgent


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

        
class BasicALEQlearningAgent(ALEQlearningAgent):
    def __init__(self,bg_file='../data/space_invaders/background.pkl',**kwargs):
        super(BasicALEQlearningAgent,self).__init__(**kwargs)
        self.background = bg_file
        
    def create_projector(self):
        return BasicALEFeatures(num_tiles=np.array([14,16]),
            background_file =  self.background )
 
    def get_data(self,obs):
        return self.get_frame_data(obs)

    
class RAMALEQlearningAgent(ALEQlearningAgent):
    def create_projector(self):
        return RAMALEFeatures()
        
    def get_data(self,obs):
        return self.get_ram_data(obs)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run Sarsa Agent')
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
    parser.add_argument('--features', metavar='F', type=str, default='BASIC',
                    help='features to use: RAM or BASIC')
    parser.add_argument('--actions', metavar='C',type=int, default=None, 
                        nargs='*',help='list of allowed actions')

    args = parser.parse_args()

    if args.features == 'RAM':
        AgentLoader.loadAgent(RAMALEQlearningAgent(agent_id=args.id,
                                     alpha =args.alpha,
                                     lambda_=args.lambda_,
                                     eps =args.eps,
                                     gamma=args.gamma, 
                                     save_path=args.savepath,
                                     actions = args.actions))
    elif args.features == 'BASIC':
        AgentLoader.loadAgent(BasicALEQlearningAgent(agent_id=args.id,
                                     alpha =args.alpha,
                                     lambda_=args.lambda_,
                                     eps =args.eps,
                                     gamma=args.gamma, 
                                     save_path=args.savepath,
                                     actions = args.actions))
    else:
        print 'unknown feature type'
    
        
