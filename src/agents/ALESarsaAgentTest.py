# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:03:10 2015

@author: pvrancx
"""


from rlglue.agent import AgentLoader as AgentLoader
from util.projectors import TileCoding


import argparse

from util.BasicALEFeatures import BasicALEFeatures
from agents.ALEAgent import ALEAgent


import numpy as np

class ALESarsaAgent(ALEAgent):
    
    def __init__(self,alpha=0.1,lambda_=0.9,gamma=.999,eps=0.0,
                 agent_id=0,save_path='.'):
        #use full images and color mode
        super(ALESarsaAgent,self).__init__(np.array([0,1,2]),agent_id,save_path)
        
        self.eps = eps
        self.name='SARSA'
        self.alpha0 = alpha
        self.alpha = alpha
        self.lambda_ = lambda_
        self.gamma = gamma

    def agent_start(self,observation):
        super(ALESarsaAgent,self).agent_start(observation)
        #reset trace
        self.trace = np.zeros_like(self.theta)
        #action selection
        phi = self.get_phi(observation)
        vals = self.get_all_values(phi,self.sparse)
        a = self.select_action(vals)
        #log state and action
        self.phi = phi
        self.a = a
        return self.create_action(self.actions[a])
        
    
    def agent_init(self,taskSpec):
        super(ALESarsaAgent,self).agent_init(taskSpec)
        self.state_projector = TileCoding(limits=self.limits)
        #BasicALEFeatures(num_tiles=np.array([14,16]),
           # background_file = '../data/space_invaders/background.pkl')
        self.theta = np.zeros((self.state_projector.num_features(),
                                self.num_actions()))
        self.sparse = True

    def get_phi(self,obs):
        im = np.array(obs.doubleArray)#self.get_frame_data(obs)#.reshape((210,160))
        if self.sparse:
            #returns only active tiles indices
            return self.state_projector.phi_idx(im)
        else:
            #returns full binary features vector
            return self.state_projector.phi(im)
    
    def update_alpha(self):
        pass
        
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
            max_acts = acts[values == np.max(values)]
            return np.random.choice(max_acts)
            
    def update_trace(self,phi,a):
        self.trace *= self.gamma*self.lambda_
        if self.sparse:
            self.trace[phi,a] += 1
        else:      
            self.trace[:,a] += phi
        #self.trace = np.clip(self.trace,0.,5.)
        
    def step(self,reward,phi_ns = None):
        n_rew = self.normalize_reward(reward)
        self.update_trace(self.phi,self.a)
        delta = n_rew - self.get_value(self.phi,self.a,self.sparse)
        a_ns = None
        if not (phi_ns is None):
            ns_values = self.get_all_values(phi_ns,self.sparse)
            a_ns = self.select_action(ns_values)
            delta += self.gamma*ns_values[a_ns]
        #normalize alpha with nr of active features
        alpha = self.alpha / float(np.sum(self.phi!=0.))
        self.theta+= alpha*delta*self.trace
        return a_ns  #a_ns is action index (not action value)

    def agent_step(self,reward, observation):
        super(ALESarsaAgent,self).agent_step(reward, observation)
        phi_ns = self.get_phi(observation)
        a_ns = self.step(reward,phi_ns)
        #log state data
        self.phi = phi_ns
        self.a = a_ns 
        
        return self.create_action(self.actions[a_ns])#create RLGLUE action
        
        
             
    def agent_end(self,reward):
        super(ALESarsaAgent,self).agent_end(reward)
        self.step(reward)

        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run Sarsa Agent')
    parser.add_argument('--id', metavar='I', type=int, help='agent id')
    parser.add_argument('--gamma', metavar='G', type=float, default=1.,
                    help='discount factor')
    parser.add_argument('--alpha', metavar='A', type=float, default=0.5,
                    help='learning rate')
    parser.add_argument('--lambda_', metavar='L', type=float, default=0.9,
                    help='trace decay')
    parser.add_argument('--eps', metavar='E', type=float, default=0.0,
                    help='exploration rate')
    parser.add_argument('--savepath', metavar='P', type=str, default='.',
                    help='save path')  
    parser.add_argument('--projector', metavar='R', type=str, default='TileCoding',
                    help='Projector')
    parser.add_argument('--config', metavar='C', type=str, default='',
                    help='config file for projector')

    args = parser.parse_args()
    
 
    AgentLoader.loadAgent(ALESarsaAgent(agent_id=args.id,
                                     alpha =args.alpha,
                                     lambda_=args.lambda_,
                                     eps =args.eps,
                                     gamma=args.gamma, 
                                     save_path=args.savepath))
    
        
