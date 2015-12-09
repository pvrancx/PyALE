# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:22:21 2015

@author: pvrancx
"""

from rlglue.agent import AgentLoader as AgentLoader
import cPickle as pickle


import argparse
import copy

from agents.ALEAgent import ALEAgent


import numpy as np

'''
Simple agent that applies fixed policy and logs the transitions
'''
class ALELoggerAgent(ALEAgent):
    def __init__(self,actions=None,agent_id=0,save_path='.',mem_size=10000):
        super(ALELoggerAgent,self).__init__(actions,agent_id,save_path)
        self.name = 'Logger'
        self.acts = np.zeros(mem_size,dtype='int')
        self.terms = np.zeros(mem_size,dtype='bool')
        self.rewards = np.zeros(mem_size)
        self.obs = np.zeros((mem_size,33728),dtype='uint8')
        self._mem_index = 0
        self._mem_size = mem_size
        
    def log_transition(self,o,a,r,t=False):
        if self._mem_index >= self._mem_size:
            return 
        self.obs[self._mem_index,:] = copy.copy(o)
        self.acts[self._mem_index] =a
        self.terms[self._mem_index] = t
        self.rewards[self._mem_index] = r
        self._mem_index += 1
        
    def policy(self,o):
        a = np.random.choice(self.actions)
        return a 
        
    def step(self,o,r=0,t=False):
        if t:
            a = -1
        else:
            a = self.policy(o)
        self.log_transition(o,a,r,t)
        return a 
        
    def agent_start(self, observation):
        super(ALELoggerAgent,self).agent_start(observation)
        #select ranfom action
        act=self.step(np.array(observation.intArray))
        return self.create_action(act)
         
    def agent_step(self,reward, observation):
         super(ALELoggerAgent,self).agent_step(reward, observation)
         act=self.step(np.array(observation.intArray),reward)
         return self.create_action(act)
         
    def agent_end(self,reward):
         super(ALELoggerAgent,self).agent_end(reward)
         act=self.step(0,reward,True)
         return self.create_action(act)
         
    def agent_cleanup(self):
        print 'saving log to file...'
        name = self.save_path+'/'+str(self.name)+'_trans.'+str(self.agent_id)
        with open(name,'wb') as f:
            pickle.dump((self.obs,self.acts,self.rewards,self.terms),f,-1)
                
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run Sarsa Agent')
    parser.add_argument('--id', metavar='I', type=int, help='agent id',
                        default=0)
    parser.add_argument('--memory', metavar='M', type=int, default=1000,
                    help='memory size')

    parser.add_argument('--savepath', metavar='P', type=str, default='.',
                    help='save path')
    parser.add_argument('--actions', metavar='C',type=int, default=None, 
                        nargs='*',help='list of allowed actions')


    args = parser.parse_args()
    
    act = None
    if not (args.actions is None):
        act = np.array(args.actions)

    print act
AgentLoader.loadAgent(ALELoggerAgent(agent_id=args.id,
                                     mem_size = args.memory,
                                     save_path=args.savepath,
                                     actions = act))

    