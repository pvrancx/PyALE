# Copyright (C) 2008, Brian Tanner
# 
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys, getopt
import numpy as np
from scipy.misc import imsave

from rlglue.agent import AgentLoader as AgentLoader

from ALEAgent import ALEAgent
from util.image_processing import as_RGB

import cPickle as pickle



# Simple method of background detection for ALE games
# keep counts of colors occuring in each pixel location
# most common color is background for that pixel

class ALEBackgroundAgent(ALEAgent):
    
    
    acts = np.array([0,1,3,4,11,12])
    
    log_name = 'background'

    def __init__(self,im_size=np.array([210,160]),log_name='background'):
        super(ALEBackgroundAgent,self).__init__()
        self.im_size = im_size
        self.num_pixels = np.prod(im_size)
        self.counts = np.zeros((self.num_pixels,256))
        self.log_name = log_name
 


    def agent_start(self,obs):
        action = super(ALEBackgroundAgent,self).agent_start(obs)

        im=self.get_frame_data(obs)

        self.counts[np.arange(self.num_pixels),im]+=1
     
        return action
        


    
    def agent_step(self,reward, observation):
         action =super(ALEBackgroundAgent,self).agent_step(reward, observation)
         im=self.get_frame_data(observation)
         self.counts[np.arange(self.num_pixels),im]+=1

         return action
    
    def agent_end(self,reward):
        pass
    
    def agent_cleanup(self):
        print 'saving memory'
        background = np.argmax(self.counts,axis=1)
        
        with file(self.log_name+'.pkl','wb') as f:
            pickle.dump(background,f,-1)
            
        imsave(self.log_name+'.png',
               as_RGB(background.reshape(self.im_size)))
    

if __name__=="__main__":
    path ='./background.pkl'
    try:
        opts, args = getopt.getopt(sys.argv[1:],"c:p:")
    except getopt.GetoptError:
        print 'ALEBackgroundAgent.py -p path'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-c':
            capacity = int(arg)
        elif opt in ("-p"):
            path = arg
    print "ALEBackgroundAgent agent with  path "+path
    AgentLoader.loadAgent(ALEBackgroundAgent())
