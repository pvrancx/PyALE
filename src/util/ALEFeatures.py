# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:49:04 2015

@author: pvrancx
"""

import numpy as np
import cPickle as pickle
from util.palettes import secam_palette

class ALEFeatures(object):
    _num_feat = 0
    
    #number of features
    def num_features(self):
        return self._num_feat
        
    #normalize state values if need
    def normalize_state(self,state):
        return state
        
    #sparse representation possible (e.g. tilecoding)
    def supports_sparse_phi(self):
        return True
    
    #return full feature vector
    def phi(self,s):
        res = np.zeros(self.num_features())
        res[self.phi_idx(s)] = 1.
        return res
        
    #return non-zero feature vector
    def phi_idx(self,s):
        return NotImplementedError()
        
    '''
    pairwise features for binary vectors. Calculates and of all
    combinations of bits in binary vector.
    Inputs: 
        r_idx: indices of nonzero bits (i.e. sparse representation)
                assumed to be sorted in ascending order
        num_bits: length of binary vector
    Returns:
        pairwise combination of features, i.e. indices of all nonzero bits
        when we take the AND of all combinations of bits in the vector
        (only consides unique pairs i.e. it only includes  
        AND(bit0,bit1)) and not AND(bit1,bit0) as well)
    '''
    
    def pairwise(self,r_idx,num_bits):
            #assumes r_idx is sorted in ascending order        
            #get pairwise nonzero indices 
            #nr of  combos for n 1bits: (n**2+n)/2) 
            # assuming we include and of bit with itself
            p_idx = np.zeros((r_idx.size**2+r_idx.size)//2,dtype='uint')
            # pairwise vector is indexed as follows:
            #first n bits are AND of bit 0 with all bits
            #next n-1 bits are AND of bit 1 with bits 1:n
            #next n-2 bits are AND of bit 2 with bits 2:n ...
            #so the offset for
            #AND of bit k with bits k:n is sum i:0 to k-1 (n-i)
            # = k*n - k(k+1)/2
            offs = r_idx*num_bits - 0.5*r_idx*(r_idx+1)
            idx = 0
            for i in range(r_idx.size):
                p_idx[idx:(idx+r_idx.size-i)] = offs[i] +r_idx[i:]-r_idx[i]
                idx = idx+r_idx.size-i
            return p_idx

    '''!use vectorization, iteration over pairs is very slow (order of magn):
        for i,ind1 in enumerate(r_idx):
            for ind2 in r_idx[i:]:
                offset = ind1*num_bits - 0.5*ind1*(ind1+1)
                p_idx[j] = (offset+ind2-ind1)
                j+=1   
        return p_idx
    '''
#RAM features, see Najaf,2010      
class RAMALEFeatures(ALEFeatures):
    def __init__(self):
        self.num_bits = 1024 #bits in ramvector
        self._num_feat = self.num_bits + (self.num_bits**2 +self.num_bits)//2
        
    def phi_idx(self,s):
        #represent 128 byte ram vector as 1024 bit vector
        s_bits = np.unpackbits(s)
        # r_idx = np.nonzero(s_bits)[0]
        r_idx = np.flatnonzero(s_bits)
        p_idx = self.pairwise(r_idx,s_bits.size)
        return np.r_[r_idx,(s_bits.size+p_idx)].astype('uint')

        
#basic frame features, see Bellemare et al, 2013
class BasicALEFeatures(ALEFeatures):
    def __init__(self,im_size=np.array([210,160]),
                 num_tiles=np.array([21,16]), secam=False,
                 background_file='background.pkl'):
        self.im_size= im_size
        self.num_pixels = np.prod(im_size)
        self.num_tiles = num_tiles
        self.secam = secam
        self.num_colors = 9 if secam else 128
        self._num_feat = np.prod(num_tiles)*self.num_colors
        with open(background_file,'rb') as f:
            self.background = pickle.load(f)
        #precompute pixel/color tile indices:
        #x,y pixel coordinates
        r,c = np.unravel_index(np.arange(self.num_pixels),im_size)
        #tile coordinates
        self.tile_height = im_size[0]/ num_tiles[0]
        self.tile_width = im_size[1]/ num_tiles[1]
        tile_rows = np.floor(r /self.tile_height).astype(int)
        tile_columns = np.floor(c /self.tile_width).astype(int)

        #each pixel's tile index
        self.pixel_indices = np.ravel_multi_index((tile_rows,tile_columns),
                                                  num_tiles)
        #lookup table to conver to 8 secam colors
        self.secam_colors = np.zeros_like(secam_palette)
        for i,c in enumerate(np.unique(secam_palette)):
            self.secam_colors[secam_palette==c] = i #replace by color index
        if secam:
            self.background = self.secam_colors[self.background]

        
    def phi_idx(self,im):
        
        #calculate pixel indices
        #idea: each tile has seperate 128-bit binary vector 
        # indicating occurence of each color in that tile
        #so for each pixel we calculate which bit is set based on
        #its tile (offset for tile 0 is 0, tile 1 is 128, tile 2  is 256, ...)
        # and its color if pixel is color 3, 3rd bit in tile vector is flipped
        if self.secam:
            im = self.secam_colors[im]
        ind = self.pixel_indices *self.num_colors + (im // 2)
        #remove background pixels
        ind = ind[im!=self.background] 
       # res[ind] = 1.
        return ind
        
    '''
    Subtract the background from given image.
    (default = black)
    
    inputs: -im: uint8 image vector using ATARI 128 bit color scheme
            (shape must be compatible with backgroud image)
            -fill_color: color to subsitute for background pixels[0-255]
                        (default: 0 - black)
    returns copy of image with backgound pixels set to fill color
            
    '''
    def subtract_background(self,im,fill_color = 15):
        result = np.ones_like(im,dtype='uint8')*int(fill_color)
        result[im!=self.background] = im[im!=self.background]
        return result
        
    '''
    Returns image representation of detected color features
    Each tile in the image is colored according to the (non-background ) 
    colors detected for that image
    '''
        
    def color_detection(self,im,fill_color=15):
        phi = self.phi(im)
        offset = 0
        tile_pixels = int(self.tile_height*self.tile_width)
        result = np.ones(self.im_size,dtype='uint8')*int(fill_color)
        
        for i in range(np.prod(self.num_tiles)):
            offset = i*self.num_colors
            tile_colors = np.sum(phi[offset:offset+self.num_colors])

            if tile_colors > 0:
                color_pix = int(np.floor(tile_pixels/tile_colors))
                tile = np.ones(tile_pixels,dtype='uint8')*fill_color
                ind = 0
                for c in range(self.num_colors):
                    if phi[offset+c] != 0:
                        tile[ind:ind+color_pix]= c*2
                        ind += color_pix
                        tile_r,tile_c = np.unravel_index(i,self.num_tiles)
                        r_offs = tile_r*self.tile_height
                        c_offs = tile_c*self.tile_width
                        result[r_offs:r_offs+self.tile_height,
                               c_offs:c_offs+self.tile_width] = tile.reshape(
                                   (self.tile_height,self.tile_width))
        return result
            
# BASS features, see Najaf,2010     
class BASSALEFeatures(BasicALEFeatures):
    def __init__(self,im_size=np.array([210,160]),
                 num_tiles=np.array([21,16]),
                 background_file='background.pkl'):
        #use secam colors to avoid explosion of pairwise features
        super(BASSALEFeatures,self).__init__(im_size=im_size,
                                            num_tiles= num_tiles,
                                            background_file=background_file, 
                                            secam = True)
    
    def num_features(self):
        #add # pairwise features
        return self._num_feat+(self._num_feat**2+self._num_feat)//2
        
    def phi_idx(self,im):
        #these are basic features
        idx= super(BASSALEFeatures,self).ph_idx(im)
        #now add paiwise feature combinations
        p_idx = self.pairwise(idx,self._num_feat)
        return np.r_[idx,(self._num_feat+p_idx)]
        
