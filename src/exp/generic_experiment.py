# 
# Copyright (C) 2008, Brian Tanner
# 
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import argparse
import numpy as np
import cPickle as pickle
import rlglue.RLGlue as RLGlue

#defaults
parser = argparse.ArgumentParser(description='run rl glue experiment')
parser.add_argument('--maxsteps', metavar='S', type=int, default=0,
                   help='maximum steps per trial (0 for unlimited)')
parser.add_argument('--numtrials', metavar='T', type=int, default=1,
                    help='number of independent trials')
parser.add_argument('--numeps', metavar='T', type=int, default=100,
                    help='number of episodes per trials')
parser.add_argument('--path', type=str,default='.',help='save path')
parser.add_argument('--logname', type=str,default='exp',help='save file name')
parser.add_argument('--expid', type=int,default=1,help='experiment id')
args = parser.parse_args()

whichEpisode=0

def runEpisode(stepLimit):
	global whichEpisode
	terminal=RLGlue.RL_episode(stepLimit)
	totalSteps=RLGlue.RL_num_steps()
	totalReward=RLGlue.RL_return()
	
	print "Experiment "+str(args.expid)+"\t Episode "+str(whichEpisode)+"\t "+str(totalSteps)+ " steps \t" + str(totalReward) + " total reward\t " + str(terminal) + " natural end"
	
	whichEpisode=whichEpisode+1

#Main Program starts here

# Remember that stepLimit of 0 means there is no limit at all!*/
for t in range(args.numtrials):
    print 'trial: '+str(t)
    whichEpisode=0
    taskSpec = RLGlue.RL_init()
    steps=np.zeros(args.numeps)
    rews=np.zeros(args.numeps)
    for ep in range(args.numeps):
        runEpisode(args.maxsteps)
        steps[ep] = RLGlue.RL_num_steps()
        rews[ep] = RLGlue.RL_return()
    print 'trial finished, final reward: '+str(rews[-1])
    #with open(args.path+'/'+args.logname+'_'+str(args.expid)+'_'+str(t)+'.pkl','w') as f:
    #    pickle.dump((steps,rews),f,-1)
    RLGlue.RL_cleanup()







