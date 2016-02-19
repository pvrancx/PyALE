import numpy as np

from util.ALEFeatures import BasicALEFeatures,RAMALEFeatures

class BasicALEAgent(object):
    def __init__(self, args, bg_file='../data/space_invaders/background.pkl'):
        super(BasicALEAgent,self).__init__(args)
        print 'blub'
        self.background = bg_file
        
    def create_projector(self):
        return BasicALEFeatures(num_tiles=np.array([14,16]),
            background_file = self.background, secam=True )
 
    def get_data(self,obs):
        return self.get_frame_data(obs)

    def file_name(self):
        return str(self.name)+'_BASIC_'+str(self.agent_id)
