# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 12:33:59 2015

@author: pvrancx
"""

import numpy as np
from agents.AbstractAgent import AbstractAgent

class ALEAgent(AbstractAgent):
    '''
    Base class for ALE RL GLUE agents
    '''
    actions = None #action set to use
    base_reward = None #reward reference, needed for normalization
    
    @classmethod
    def register_with_parser(cls, parser):
        super(ALEAgent, cls).register_with_parser(parser)
        parser.add_argument('--actions', type=int, default=None, 
                            nargs='*',help='list of allowed actions')
        parser.add_argument('--random_seed', type=int,
                            default=None, help='Seed for random number generation')


    def __init__(self, args):
        super(ALEAgent,self).__init__(args)
        if args.actions is None:
            self.actions = np.arange(18) #18 buttons
        else:
            args.actions = np.array(args.actions)
            assert np.all(np.logical_and(args.actions>=0,args.actions<18)), \
                'invalid action'
            self.actions = args.actions
        print "using actions", self.actions
        self.rng = np.random.RandomState(args.random_seed)  
            
    
    '''
    Reward normalization. Stores first nonzero reward and divides all future
    rewards by its abolute value.  
    ''' 
    def normalize_reward(self,rew):
        if rew == 0.:
            return rew
        if self.base_reward is None:
            self.base_reward = np.abs(rew)
        return rew / self.base_reward
    
    '''
    Extract RAM from ALE observation (as flat uint8 array)
    Inputs:
        ALE observation as returned by RLGLUE
    Outputs:
        new numpy vector (128,)  containing ALE RAM data (uint8)

    '''
    def get_ram_data(self,obs):
        #extract ram info: first 128 bytes
        res=np.array(obs.intArray[:128],dtype=np.uint8)
        return res
    
    '''
    Extract current frame from ALE observation (as flat uint8 array)
    Inputs:
        ALE observation as returned by RLGLUE
    Outputs:
        new numpy vector (33600,) containing ALE frame data (uint8)

    '''
    def get_frame_data(self,obs):
        #extract screen info
        im=np.array(obs.intArray[128:33728],dtype=np.uint8)
        return im
        
    ''' RL GLUE Interface '''
    
    def agent_init(self,taskspec):
        super(ALEAgent,self).agent_init(taskspec)
        
        #change default ALE task, no ply2 act
        self._n_int_actions = len(self.actions)
        self._n_int_act_dims = 1
        print 'ALE AGENT:'
        print 'Integer Action dimensions:'
        print self.int_action_dims()
        print '#number of Integer Actions'
        print self.num_actions()

    
        
    def agent_start(self, observation):
        super(ALEAgent,self).agent_start(observation)
        #select ranfom action
        act=np.random.choice(self.actions)
        #return action as RLGLUE struct
        return self.create_action(act)
         
    def agent_step(self,reward, observation):
         super(ALEAgent,self).agent_step(reward, observation)
         act=np.random.choice(self.actions)
         return self.create_action(act)
    
            

    
    #def agent_message(self,inMessage):
    #    return "I don't know how to respond to your message"
