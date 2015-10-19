"""
Created 2014

@author: pvrancx
"""

import copy
import argparse
import numpy as np
from PIL import Image
from rlglue.agent import AgentLoader as AgentLoader

import cPickle as pickle



from agents.ALEAgent import ALEAgent
from util.palettes import ntsc_palette, grayscale_palette, pal_palette,\
                             secam_palette

                         

class ALEVisionAgent(ALEAgent):
    '''
    ALE Agent implementing basic visual processing to extract data from ALE 
    frames
    '''
        
    image_memory = None
    log_name = 'images'
    im_step =0

    def __init__(self,actions=None,mem_capacity=0,resize=(105,80),
                 color_mode='ale', name='frames'):
        super(ALEVisionAgent,self).__init__(actions)
        self.log_name = name
        self.mem_capacity = mem_capacity
        self.color_mode = color_mode
        self.im_shape = (210,160) if resize is None else resize
        if self.color_mode == 'ntsc':#rgb
            self.palette = ntsc_palette
            self.image_memory = np.zeros((self.mem_capacity,3*
                 np.prod(self.im_shape)),dtype='uint8')
        elif self.color_mode == 'pal':#rgb
            self.palette = pal_palette
            self.image_memory = np.zeros((self.mem_capacity,3*
                 np.prod(self.im_shape)),dtype='uint8')
        elif self.color_mode == 'pal':#rgb
            self.palette = pal_palette
            self.image_memory = np.zeros((self.mem_capacity,3*
                 np.prod(self.im_shape)),dtype='uint8')
        elif self.color_mode == 'pal':#rgb
            self.palette = secam_palette
            self.image_memory = np.zeros((self.mem_capacity,3*
                 np.prod(self.im_shape)),dtype='uint8')
        elif self.color_mode == 'gray':
            #grayscale
            self.palette = grayscale_palette
            self.image_memory = np.zeros((self.mem_capacity,
                 np.prod(self.im_shape)),dtype='uint8')
        elif self.color_mode == 'ale':
            #grayscale
            self.palette = grayscale_palette
            self.image_memory = np.zeros((self.mem_capacity,
                 np.prod(self.im_shape)),dtype='uint8')
        else:
            raise ValueError('invalid color mode')
        
    
    def agent_init(self,taskspec):
        super(ALEVisionAgent,self).agent_init(taskspec)



    def agent_start(self,observation):
        act = super(ALEVisionAgent,self).agent_start(observation)
        self.store_image(observation)
        return act
        

    '''
    Crop image to given shape. Cuts equal amount of borders until size is reached
    Inputs: 
        image: 2D float array
        shape: tuple, new image shape. Must obey shape[0] <= image.shape[0] and
                shape[1] <= image.shape[0]
    Returns:
        Copy of image, cut down to shape
    '''
    def crop_image(self,im,shape):
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
    def resize_image(self,im,shape, antialias = False):
        if im.shape == shape:
            return im
            
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
    def get_frame_data_gray(self,obs):
        #extract screen info
        im=self.get_frame_data(obs).reshape((210,160))
        #convert atari color codes to grayscale
        im = grayscale_palette[im]
        assert not np.any(im<0.),"invalid pixel"
        #resize image, use PIL misc.imresize gives poor results for downsizing
        #im = Image.fromarray(im).resize(self.imShape,Image.ANTIALIAS)
        #convert back to numpy array and scale to [0,1] range
        #im = np.array(im)/255.
        #antialiasing can produce negative values, clip
        #im[im<0.]=0.
        return self.resize_image(im,self.im_shape).astype(np.uint8)
        
        
    '''
    Store frame in image memory
    '''        
    def store_image(self,obs):
        #store screen
        if self.mem_capacity > 0:
            if self.color_mode in ['ntsc','pal','secam']:
                self.image_memory[self.im_step%self.mem_capacity,:]= \
                    self.get_frame_data_RGB(obs).flatten()
            elif self.color_mode in ['gray','ale']:
                self.image_memory[self.im_step%self.mem_capacity,:]= \
                    self.get_frame_data_gray(obs).flatten()
            else:
                raise ValueError('invalid color mode')
            self.im_step += 1

    '''
    Convert ALE frame to RGB image
    '''
    def as_RGB(self,im):
        palette = self.palette if self.color_mode in \
                    ['ntsc','pal','secam'] else ntsc_palette
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
    def get_frame_data_RGB(self,obs):
        #extract screen info
        im=self.get_frame_data(obs).reshape((210,160))
        #convert atari color codes to hex color values
        rgb = self.as_RGB(im)
        #resize image to desired frame proportions 
        return self.resize_image(rgb,self.im_shape).astype(np.uint8)
    
    def agent_step(self,reward, observation):
         act=super(ALEVisionAgent,self).agent_step(reward, observation)
         self.store_image(observation)
         return act
    
    
    def save_images(self):
        with file(self.log_name,'wb') as f:
            pickle.dump(self.image_memory,f,-1)
        
    
    def agent_cleanup(self):
        if self.mem_capacity > 0:
            self.save_images()
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run ALEVisionAgent Agent')
    parser.add_argument('--capacity', metavar='C', type=int, default=0,
                    help='frame memory capacity')
    parser.add_argument('--height', metavar='H', type=int, default=None,
                    help='image height')
    parser.add_argument('--width', metavar='W', type=int, default=None,
                    help='image width')
    parser.add_argument('--color', metavar='L', type=str, default='ale',
                    help='frame color mode')
    parser.add_argument('--name', metavar='N', type=str, default='frames',
                    help='output file name')


    args = parser.parse_args()
    if (args.width is None) or (args.height is None):
        resize = None
    else:
        resize = (args.height,args.width)
    print "Vision agent with capacity "+str(args.capacity) 

    AgentLoader.loadAgent(ALEVisionAgent(mem_capacity= args.capacity,
                                         resize=resize,
                                         color_mode = args.color,
                                         name = args.name))


