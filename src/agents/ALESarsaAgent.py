# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:03:10 2015

@author: pvrancx, Ruben Vereecken
"""
import argparse
import copy

from rlglue.agent import AgentLoader as AgentLoader

from util.ALEFeatures import BasicALEFeatures,RAMALEFeatures
from agents.ALEAgent import ALEAgent
from agents.BasicALEAgent import BasicALEAgent
from agents.RAMALEAgent import RAMALEAgent

import numpy as np


class ALESarsaAgent(ALEAgent):
    
    @classmethod
    def register_with_parser(cls, parser):
        super(ALESarsaAgent, cls).register_with_parser(parser)
        parser.add_argument('--gamma', type=float, default=0.999,
                        help='discount factor')
        parser.add_argument('--alpha', type=float, default=0.5,
                        help='learning rate')
	#lambda 0. means no traces
        parser.add_argument('--lambda', type=float, default=0.5, dest='lambda_',
                        help='trace decay')
        parser.add_argument('--eps', type=float, default=0.05,
                        help='exploration rate')

    def __init__(self, args):
        super(ALESarsaAgent, self).__init__(args)
        self.name = 'SARSA'
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.alpha0 = args.alpha
        self.lambda_ = args.lambda_
        self.eps = args.eps

    def agent_start(self,observation):
        super(ALESarsaAgent,self).agent_start(observation)
        #reset trace
        self.trace = np.zeros_like(self.theta)
        #action selection
        phi_ns = self.get_phi(observation)
        vals = self.get_all_values(phi_ns,self.sparse)
        action_idx = self.select_action(vals)
        #store state and action
        self.last_phi = copy.deepcopy(phi_ns)
        self.last_action = copy.deepcopy(action_idx)
        return self.create_action(self.actions[action_idx])
        
    
    def agent_init(self,taskSpec):
        super(ALESarsaAgent,self).agent_init(taskSpec)
        self.state_projector =  self.create_projector()
        print 'Using {} features'.format(self.state_projector.num_features())
        self.theta = np.zeros((self.state_projector.num_features(),
                                self.num_actions()))
        self.sparse = True
      
    def create_projector(self):
        raise NotImplementedError()
        
    def get_data(self,obs):
        raise NotImplementedError()

    #convert observation to feature vector
    def get_phi(self,obs):
        im = self.get_data(obs)
        if self.sparse:
            #returns only active tiles indices
            return self.state_projector.phi_idx(im)
        else:
            #returns full binary features vector
            return self.state_projector.phi(im)

        
    def get_value(self,phi,a,sparse=False):
        if sparse:
            return np.sum(self.theta[phi,a])
        else:
            return np.dot(phi,self.theta[:,a])
            
    def get_all_values(self,phi,sparse=False):
        if sparse:
            return np.sum(self.theta[phi,:],axis=0)
        else:
            return np.dot(phi,self.theta)
            
    def select_action(self,values): 
        #egreedy
        acts = np.arange(values.size)
        if np.random.rand()< self.eps:
            return np.random.choice(acts)
        else:
            #randomly select action with maximum value
            max_acts = acts[(values==np.max(values))]
            #print np.max(values)
            return np.random.choice(max_acts)
            
    def update_trace(self,phi,a):
        self.trace *= self.gamma*self.lambda_
        if self.sparse:
            #phi consists of nonzero feature indices
            self.trace[self.last_phi,a] += 1.
        else:
            #phi is full vector of feature values
            self.trace[:,a] += self.last_phi
        
       # self.trace = np.clip(self.trace,0.,5.)
        
        
    def step(self,reward,phi_ns = None):
        # phi_ns can be None in case of agent_end, since you only get a
        # reward for your last action
        n_rew = self.normalize_reward(reward)
        self.update_trace(self.last_phi,self.last_action)
        delta = n_rew - self.get_value(self.last_phi,self.last_action,self.sparse)
        a_ns = None
        if not (phi_ns is None):
            ns_values = self.get_all_values(phi_ns,self.sparse)
            a_ns = self.select_action(ns_values)
            delta += self.gamma*ns_values[a_ns]
        #normalize alpha with nr of active features
        alpha = self.alpha / float(np.sum(self.last_phi!=0.))
        self.theta+= alpha*delta*self.trace
        return a_ns  #a_ns is action index (not action value)

    def agent_step(self,reward, observation):
        super(ALESarsaAgent,self).agent_step(reward, observation)
        phi_ns = self.get_phi(observation)
        a_ns = self.step(reward,phi_ns)
        #log state data
        self.last_phi = copy.deepcopy(phi_ns)
        self.last_action = copy.deepcopy(a_ns)
        
        return self.create_action(self.actions[a_ns])#create RLGLUE action
             
    def agent_end(self,reward):
        super(ALESarsaAgent,self).agent_end(reward)
        self.step(reward)
        

class BasicALESarsaAgent(BasicALEAgent, ALESarsaAgent):
    pass
    
class RAMALESarsaAgent(RAMALEAgent, ALESarsaAgent):
    pass
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run Sarsa Agent')
    parser.add_argument('--features', metavar='F', type=str, default='RAM',
                    help='features to use: RAM or BASIC')
    ALESarsaAgent.register_with_parser(parser)
    args = parser.parse_args()
    if args.features == 'RAM':
        AgentLoader.loadAgent(RAMALESarsaAgent(args))
    elif args.features == 'BASIC':
        AgentLoader.loadAgent(BasicALESarsaAgent(args))
    else:
        raise Exception('unknown feature type')
