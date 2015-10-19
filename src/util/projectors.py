# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 15:19:49 2015

@author: pvrancx
"""

import numpy as np
from util.CTiles import tiles


class Projector(object):
    normalized = False
    
    def __init__(self,normalize=False,
                 limits=np.array([[0,1],[0,1]]),bias = False):
        self.limits = limits
        self.ranges = limits[:,1]-limits[:,0]
        self.normalized = normalize
        self.bias = bias
    
    def project(self,state):
        phi = self.phi(state)
        if self.normalized:
            phi = phi / np.sum(phi)
        if self.bias:
            phi = np.r_[phi,[1.]]
        return phi
            
    def phi(self,state):
        pass
    
    def normalize_state(self,state):
        return (state - self.limits[:,0]) / self.ranges
        
    ''' Indicates if projector supports returning nonzero indices/values only
        Returns: boolean
    '''
    def supports_sparse(self):
        return False
    
class RBF(Projector):
    
    def __init__(self,num_centers=np.array([5]),
                 stdev=0.1,limits=np.array([[0,1],[0,1]]),
                 randomize=False,normalize = True,bias=True):
                     
        super(RBF,self).__init__(self,normalize,limits,bias) 
        if not (type(num_centers) is np.ndarray):
            num_centers = np.array(num_centers)
        if num_centers.size == 1:
            num_centers = np.ones(limits.shape[0])*num_centers[0]
        self.stdev = stdev
        dim = []
        if randomize:
            #randomly spaced centers
            for d in range(limits.shape[0]):
                dim.append(np.sort(np.random.rand(num_centers[d])))
        else: 
            #equally spaced centers
            for d in range(limits.shape[0]):
                dim.append(np.linspace(0,1,num_centers[d]))
        if len(dim) == 1:
            self.centers=dim[0].flatten()
        else:
            grid = np.meshgrid(*dim)
            self.centers=grid[0].flatten()
            for d in range(1,len(grid)):
                self.centers = np.c_[self.centers,grid[d].flatten()]

        
    def num_features(self):
        return (self.centers.shape[0]+int(self.bias))
        
    def phi(self,state):
        if len(self.centers.shape)==1:
            dists = self.centers-self.normalize_state(state)
        else:
            dists = np.linalg.norm(self.centers-
                self.normalize_state(state),axis=1)
        res = np.exp(-0.5 * dists**2 / self.stdev**2)
        
        return res
        
    

    
    #def phiIdx(self,state):
     #   unsupported
    
 
        
        
        
class StateToStateAction(Projector):
    num_actions = 1
    state_projector = None
    
    def __init__(self,projector,num_actions=1):
        self.state_projector = projector
        self.num_actions = num_actions

        
    def num_features(self):
        return self.num_actions*self.state_projector.num_features()
        
    def to_state_action(self,phi,a,idx=False):
        if idx:
            phi_a = phi + (a*phi.size)
        else:
            phi_a = np.zeros(phi.size*self.num_actions)
            phi_a[(a*phi.size):((a+1)*phi.size)]=phi
        return phi_a
        
    def phi_idx(self,state):
        return self.state_projector.phi_idx(state)
        
    def phi(self,state):
        return self.state_projector.phi(state)
        
    def project(self,state):
        return self.state_projector.project(state)
        
    def supports_sparse(self):
        return self.state_projector.supports_sparse()
    
        
    def phi_sa(self,state,action):
        phi = self.state_projector.phi(state)
        return self.to_state_action(phi,action)
        
    def project_sa(self,state,a):
        phi = self.phi(state,a)
        if self.normalized:
            return phi / np.sum(phi)
        else:
            return phi
            

class TileCoding(Projector):
    def __init__(self, num_tilings=[10],
                 num_tiles=np.array([[10,10]]),
                limits=np.array([[0,1],[0,1]]), 
                 memory_size=32768, id =0):
        if np.isscalar(num_tiles):
            num_tiles = num_tiles*np.ones((len(num_tilings),limits.shape[0]))
        if not (type(num_tiles) is np.ndarray):
            num_tiles = np.array(num_tiles)
        assert limits.shape[0] == num_tiles.shape[1], 'Dimension mismatch'
        assert len(num_tilings) == num_tiles.shape[0], 'Undefined tilings'
        Projector.__init__(self,False,limits)        
        self.memory_size = memory_size
        self.num_tilings = num_tilings
        self.total_tilings = np.sum(num_tilings)
        self.num_tiles = 1./num_tiles
        self.id =id
        
    def num_features(self):
        return self.memory_size
        
    def supports_sparse(self):
        return True
        
        
    def phi_idx(self,state):
        n_state = self.normalize_state(state)
        ind = []
        for idx,nt in enumerate(self.num_tilings):
            ind.extend(tiles.tiles(nt, self.memory_size, 
                    (n_state/self.num_tiles[idx,:]).tolist(),[self.id+idx]))
        return np.array(ind)
        
    def phi(self,state):
        phi_vector = np.zeros(self.memory_size)
        inds = self.phi_idx(state)
        phi_vector[inds] = 1.
        return phi_vector
    
    
        
