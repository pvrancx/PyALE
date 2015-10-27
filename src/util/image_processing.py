# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:52:50 2015

@author: pvrancx
"""

"""
Created 2014

@author: pvrancx
"""

import copy
import numpy as np
from PIL import Image

from util.palettes import ntsc_palette, grayscale_palette, pal_palette,\
                             secam_palette
                             
'''
Various image processing routines to preprocess ALE frames
'''

        

'''
Crop image to given shape. Cuts equal amount of borders until size is reached
Inputs: 
    image: 2D float array
    shape: tuple, new image shape. Must obey shape[0] <= image.shape[0] and
            shape[1] <= image.shape[0]
Returns:
    Copy of image, cut down to shape
'''
def crop_image(im,shape):
    #calculate difference between current shape and desired shape
    dx, dy = im.shape[0:2] - shape
    assert (dx>=0) and (dy>=0), 'cropping cannot increase im size'
    #divide cropped area equally over opposite borders
    indx,indy= np.floor([dx,dy]/2).astype('int')
    #check if difference is odd, need to substract extra row/column
    remx,remy = np.remainder([dx,dy],2).astype('int')
    # Cropping
    return np.array(im[(indx+remx): - indx, (indy+remy): -indy]) #new copy
    
'''
Resize image to given shape. Uses subsampling (with optional antialiasing)
Inputs: 
    image: 2D float array
    shape: tuple, new image shape. 
Returns:
    Copy of image, resized  to shape
'''
def resize_image(im,shape, antialias = False):
    if im.shape == shape:
        return copy.copy(im)
        
    if antialias: #use PIL
        #resize image, use PIL misc.imresize gives poor results for downsizing
        new_im = Image.fromarray(im).resize(shape,Image.ANTIALIAS)
    else: #simple subsampling
        dx,dy = np.divide(im.shape[:1],shape)
        x_idx = np.arange(0,im.shape[0],dx)
        y_idx = np.arange(0,im.shape[1],dy)
        xs,ys = np.meshgrid(x_idx,y_idx)
        if len(im.shape) == 3:
            new_im = im[xs,ys,:]
        else:
            new_im = im[xs,ys]
    #convert back to numpy array
    new_im =np.array(new_im)
    #ANTIALIAS can create invalid pixel values
    new_im = np.clip(0,255,new_im)
    return new_im
    
'''
Returns ALE frame as resized grayscale image
Inputs:
    RL GLUE ALE observation
Outputs:
    ndarray containing  current frame as grayscale image of size given
    in constructor
'''
def get_frame_data_gray(frame,im_shape):
    #extract screen info
    im=frame.reshape((210,160))
    #convert atari color codes to grayscale
    im = grayscale_palette[im]
    assert not np.any(im<0.),"invalid pixel"
    #resize image, use PIL misc.imresize gives poor results for downsizing
    #im = Image.fromarray(im).resize(self.imShape,Image.ANTIALIAS)
    #convert back to numpy array and scale to [0,1] range
    #im = np.array(im)/255.
    #antialiasing can produce negative values, clip
    #im[im<0.]=0.
    return resize_image(im,im_shape).astype(np.uint8)
        
        


'''
Convert ALE frame to RGB image
'''
def as_RGB(im,color_mode='ntsc'):
    if color_mode == 'ntsc':#rgb
        palette = ntsc_palette
    elif color_mode == 'pal':#rgb
        palette = pal_palette
    elif color_mode == 'secam':#rgb
        palette = secam_palette
    else:
        raise ValueError('invalid color mode')
    im = palette[im]
    #convert to rgb
    rgb = np.zeros((im.shape[0],im.shape[1],3),dtype=np.uint8)
    #red channel
    rgb[...,0] = np.bitwise_and(np.right_shift(im,16),0xff)
    #green channel
    rgb[...,1] = np.bitwise_and(np.right_shift(im,8),0xff)
    #blue channel
    rgb[...,2] = np.bitwise_and(im,0xff)
    return rgb
        
'''
Returns ALE frame as resized RGB image
'''
def get_frame_data_RGB(frame,im_shape):
    #extract screen info
    im=frame.reshape((210,160))
    #convert atari color codes to hex color values
    rgb = as_RGB(im)
    #resize image to desired frame proportions 
    return resize_image(rgb,im_shape).astype(np.uint8)
    
