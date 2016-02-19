from util.ALEFeatures import RAMALEFeatures

import numpy as np

class RAMALEAgent(object):
    def create_projector(self):
        return RAMALEFeatures()
        
    def get_data(self,obs):
        return self.get_ram_data(obs)

    def file_name(self):
        return str(self.name)+'_RAM_'+str(self.agent_id)
