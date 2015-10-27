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
Crop image by cutting specified amounts of top, bottom, left, right

Inputs: 
    image: 2D float array
    top = 0: amount to cut from top
    right = 0:amount to cut from right
    bottom =0: amount to cut from bottom
    left = 0:amount to cut from left
Returns:
    Copy of image, cut down by given amounts
'''

def crop_image(im,top=0,right=0,bottom=0,left=0):
    return copy.copy(im[top:im.shape[0]-bottom,left:im.shape[1]-right])
    
'''
Crop image to given shape. Cuts equal amount of borders until size is reached
Inputs: 
    image: 2D float array
    shape: tuple, new image shape. Must obey shape[0] <= image.shape[0] and
            shape[1] <= image.shape[0]
Returns:
    Copy of image, cut down to shape
'''

def crop_image_to_shape(im,shape):
    #calculate difference between current shape and desired shape
    dx, dy = im.shape[0:2] - np.array(shape)
    assert (dx>=0) and (dy>=0), 'cropping cannot increase im size'
    deltas = np.array([dx,dy])
    #divide cropped area equally over opposite borders
    indx,indy= np.floor(deltas/2).astype('int')
    #check if difference is odd, need to substract extra row/column
    remx,remy = np.remainder(deltas,2).astype('int')
    # Cropping
    return crop_image(im,(indx+remx),indy,indx,(indy+remy)) #new copy
    
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
        #convert back to numpy array
        new_im =np.array(new_im)
        #ANTIALIAS can create invalid pixel values
        new_im = np.clip(new_im,0,255)
    else: #simple subsampling
        xs,ys = np.meshgrid (
            np.linspace(0,im.shape[0]-1,shape[0]).astype('uint8'),
            np.linspace(0,im.shape[1]-1,shape[1]).astype('uint8'),
            indexing='ij' )

        if len(im.shape) == 3:
            new_im = copy.copy(im[xs,ys,:])
        else:
            new_im = copy.copy(im[xs,ys])


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
    
