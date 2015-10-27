
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 12:50:45 2015

@author: pvrancx
"""

import numpy as np
import copy
import cPickle as pickle


from rlglue.agent.Agent import Agent
from rlglue.utils import TaskSpecVRLGLUE3
from rlglue.types import Action

#from util.log import Logger


#convert agents to new python classes
class AbstractAgent(Agent,object):
    
    '''
    Base RlGlue Agent
    '''
    
    #problem description
    _n_double_dims = 1
    _n_int_dims = 0
    _n_int_act_dims = 1
    _n_double_act_dims = 0
    _n_int_actions = 0
    limits= None
    act_limits = None
    act_range = None
    
    
    #logging
    last_action = None
    last_observation = None
    last_state = None
    log_freq = 100
    
    num_eps=0
    ep_rew =0.
    ep_steps=0
    exploration = True
    name = 'AbstractAgent'    
    
    def __init__(self,agent_id=0,save_path='.'):
        self.agent_id = agent_id
        self.save_path=save_path
        self.log = {}
    
    def log_value(self,key,value):
        if self.log.has_key(key):
            self.log[key].append(value)
        else:
            self.log[key] = [value]
    

    '''
        Turns RLGlue observation into state representation (single double array)
    '''
    def get_state(self,obs):
        self.last_observation=copy.deepcopy(obs)
        state = []
        if self.double_dims()>0:
            state.append(obs.doubleArray)
        if self.int_dims()>0:
            state.append(obs.intArray)
        self.last_state=np.array(state).flatten()
        return self.last_state
        
    def create_action(self,act):
        self.last_act=act
        if np.isscalar(act):
            act = np.array([act])
        assert (act.size == self.action_dims()),'illegal action dimension'
        return_action=Action()
        if self.int_action_dims() > 0:
            return_action.intArray=[act[:self.int_action_dims()].astype(int)] 
        if self.double_action_dims() > 0:
            return_action.doubleArray=[
                act[self.double_action_dims():].astype(float)]
        return return_action
      
    def state_dims(self):
        return self.double_dims() +self.int_dims()
        
    def is_discrete_problem(self):
        return (self.double_dims()==0)
        
    def has_discrete_actions(self):
        return (self.double_action_dims==0)
        
    def double_dims(self):
        return self._n_double_dims
        
    def int_dims(self):
        return self._n_int_dims
        
    def double_action_dims(self):
        return self._n_double_act_dims
        
    def int_action_dims(self):
        return self._n_int_act_dims
        
    def num_actions(self):
        return self._n_int_actions
        
    def action_dims(self):
        return self.double_action_dims() + self.int_action_dims()
        
    def agent_init(self,taskspec):
        self.parse_taskspec(taskspec)
        self.log.clear()
        self.agent_id += 1
        
    def agent_start(self,obs):
        self.ep_rew = 0.
        self.ep_steps = 0
        
    def agent_step(self,reward,obs):
        self.ep_rew += reward
        self.ep_steps+= 1 
    
    def agent_end(self,reward):
        self.ep_rew += reward
        self.ep_steps+= 1 
        self.log_value('episode',self.num_eps)
        self.log_value('steps',self.ep_steps)
        self.log_value('reward',self.ep_rew)
        self.num_eps+=1


    
    '''
    Support for evaluation and learning episodes.
    Agents should implement approriate response to exploration flag
    '''
    def agent_message(self,msg):
        if msg == 'learning mode':
            self.exploration = True
            return 'ok'
        elif msg == 'evaluation mode':
            self.exploration = False
            return 'ok'
        else:
            return "I don't know how to respond to your message"
            
    def agent_cleanup(self):
        print 'saving log to file...'
        name = self.save_path+'/'+str(self.name)+'_log.'+str(self.agent_id)
        with open(name,'wb') as f:
            pickle.dump(self.log,f)
        
    def parse_taskspec(self,spec):
        TaskSpec=TaskSpecVRLGLUE3.TaskSpecParser(spec)
        if TaskSpec.valid:
            self._n_double_dims = len(TaskSpec.getDoubleObservations())
            self._n_int_dims = len(TaskSpec.getIntObservations())
            self._n_double_act_dims = len(TaskSpec.getDoubleActions())
            self._n_int_act_dims = len(TaskSpec.getIntActions())

            self.limits= np.array(TaskSpec.getDoubleObservations())
            self.act_limits = None
            if self._n_double_act_dims != 0:
               self.act_limits =  np.array(TaskSpec.getDoubleActions())
            if self._n_int_act_dims != 0:
                if self.act_limits is None:
                    self.act_limits =  np.array(TaskSpec.getIntActions())
                else:
                    self.act_limits = np.append(self.act_limits,
                                            np.array(TaskSpec.getIntActions()),
                                            axis=0)
            print self.act_limits
            self.act_range = self.act_limits[:,1] - self.act_limits[:,0] 
            self._n_int_actions = np.prod(
                self.act_range[self._n_double_act_dims:])
            print spec
            print 'Double state variables:'
            print len(TaskSpec.getDoubleObservations())
            print 'Integer state variables:'
            print len(TaskSpec.getIntObservations())            
            print 'Double Actions dimensions:'
            print self.double_action_dims()
            print 'Integer Action dimensions:'
            print self.int_action_dims()
            print '#number of Integer Actions'
            print self.num_actions()
        else:
            print "Task Spec could not be parsed: "+spec
            





