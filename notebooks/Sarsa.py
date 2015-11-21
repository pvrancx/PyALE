
# coding: utf-8

# In[1]:

import sys
sys.path.append('/home/ruben/PyALE/src')


# In[17]:

# %load ../src/util/RLGlueRunner.py
from multiprocessing import Process
import os
import rlglue.RLGlue as RLGlue
from rlglue.agent import AgentLoader as AgentLoader



def run_rlglue():
    os.system('rl_glue')
    
def run_ale(game):
    ale_str = 'ale -display_screen false -game_controller rlglue -frame_skip 5 -disable_colour_averaging -use_environment_distribution '+ale_path +'/roms/'+game 
    os.system(ale_str)   

def run_experiment(maxsteps=100,numeps=1):
    taskSpec = RLGlue.RL_init()
    for ep in range(numeps):
        terminal=RLGlue.RL_episode(maxsteps)
        totalSteps=RLGlue.RL_num_steps()
        totalReward=RLGlue.RL_return()
        print "Episode "+str(ep)+"\t "+str(totalSteps)+ " steps \t" + str(totalReward) + " total reward\t " + str(terminal) + " natural end"
    RLGlue.RL_cleanup()
    
def run_agent(agent=None):
    AgentLoader.loadAgent(agent)
    
class RLGlueRunner(object):
    procs = []
    
    def __init__(self,host='127.0.0.1',port='4096',
                 game = 'space_invaders.bin',agent=None,num_eps=1,max_steps=100):
        self.host = host
        self.port = port
        self.game = game
        self.agent = agent
        self.num_eps = num_eps
        self.max_steps = max_steps
        
    def create_procs(self):
        self.procs = []
        self.procs.append(Process(target=run_rlglue))
        self.procs.append(Process(target=run_ale,args=(self.game)))
        self.procs.append(Process(target=run_experiment,args=(self.max_steps,self.num_eps)))
        self.procs.append(Process(target=run_agent,args=(self.agent,)))
        return self.procs
        
    def run(self):
        os.environ['RLGLUE_HOST'] = self.host
        os.environ['RLGLUE_PORT'] = self.port
        for p in self.procs:
            p.start()
            
    def is_finished(self):
        return reduce(lambda x,y: x and y,map(lambda x: not x.is_alive(),self.procs))
                        
    def terminate(self):
        for p in self.procs:
            p.terminate()
    


# In[18]:

from agents.ALESarsaAgent import RAMALESarsaAgent

agent= RAMALESarsaAgent(alpha=0.5,lambda_=0.9,gamma=0.999,agent_id=2)
runner= RLGlueRunner(agent=agent,num_eps=10,max_steps=2000)
runner.run()


# In[16]:

runner.is_finished()


# In[9]:

runner.terminate()


# In[5]:

import cPickle as pickle
with open('SARSA_log.0','rb') as f:
    log = pickle.load(f)


# In[6]:

print log


# In[11]:

ls


# In[6]:

import cPickle as pickle
with open('../src/SARSA_log.None','rb') as f:
    d = pickle.load(f)


# In[16]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

def avg_rew(data,w):
    return np.convolve(data,np.ones(w),mode='valid')/float(w)

plt.plot(avg_rew(d['reward'],100))


# In[5]:

1024*1024+1024


# In[ ]:



