
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def load_random_imnet(num_images=1, base_dir='data/image_net/'):
    im_list = glob.glob(base_dir + '*.JPEG')
    
    idxs = np.random.randint(len(im_list), size=num_images)
    
    ims = []
    
    for i in range(num_images):
        ims.append(plt.imread(im_list[idxs[i]]))            
    return ims

def load_imnet(base_dir='data/image_net/', max_ims=None):
    ims = []
    
    im_list = glob.glob(base_dir + '*.JPEG')
    np.random.shuffle(im_list)
    
    if max_ims is not None and max_ims < len(im_list):
        im_list = im_list[:max_ims]
    
    for imfname in im_list:
        ims.append(plt.imread(imfname))
        
    return ims
    
    
def load_random_imnet_patches(num_images=1, patch_size=(20,20)):
    
    ims = load_imnet(max_ims=num_images)
    
    im_patches = np.zeros((patch_size[0], patch_size[1], 3, num_images))
    
    for idx in range(num_images):
        if idx % 1000 == 0:
            print idx,
        cc=1
        im = ims[np.random.randint(len(ims))]
        
        while patch_size[0] >= im.shape[0]-1 or patch_size[1] >= im.shape[1]-1:
            # load a different image
            im = ims[np.random.randint(len(ims))]
            cc += 1
            if cc > 20:
                raise "patch size to large."
        
        randr_st = np.random.randint(im.shape[0] - patch_size[0]-1)
        randc_st = np.random.randint(im.shape[1] - patch_size[1]-1)
        
        if len(im.shape) < 3:
            # Then this is grayscale
            im_patches[:,:,0,idx] = im[randr_st:(randr_st + patch_size[0]), 
                                        randc_st:(randc_st + patch_size[1])]
            im_patches[:,:,1,idx] = im[randr_st:(randr_st + patch_size[0]), 
                                        randc_st:(randc_st + patch_size[1])]
            im_patches[:,:,2,idx] = im[randr_st:(randr_st + patch_size[0]), 
                                        randc_st:(randc_st + patch_size[1])]
        
        else:
            im_patches[:,:,:, idx] = im[randr_st:(randr_st + patch_size[0]), 
                                        randc_st:(randc_st + patch_size[1]), :]
        #except IndexError, e:
        #    print idx, im.shape, randr_st, randc_st
        #    raise e
            
    return im_patches+1

