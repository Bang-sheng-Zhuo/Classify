import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import cv2
import pdb
from skimage import io
from skimage import color
from PIL import Image



class retinalDataLoader(object):
    def __init__(self):
        self.X = None
        self.Y = None
        self.YY = None
        self.mask = None
        
    def load_CHASEDB1(self, file_path="D:/vessel dataset/CHASEDB/CHASEDB1"):
        file_list = glob.glob(file_path+"/*.jpg")
        file_list.sort()
        img = []
        for file in file_list:
            tmp = io.imread(file)
            img.append(tmp)
        self.X = np.asarray(img)
        seg_list = glob.glob(file_path+"/*.png")
        seg_list.sort()
        seg = []
        for file in seg_list:
            tmp = io.imread(file)
            seg.append(tmp)
        self.Y = np.asarray(seg[0::2])//255
        self.YY = np.asarray(seg[1::2])//255
        
    def load_DRIVE(self, 
                   file_path="D:/vessel dataset/DRIVE/DRIVE/training/images", 
                   seg_path="D:/vessel dataset/DRIVE/DRIVE/training/1st_manual", 
                   mask_path="D:/vessel dataset/DRIVE/DRIVE/training/mask"
                   ):
        # load original images
        file_list = glob.glob(file_path+"/*.tif")
        file_list.sort()
        img = []
        for file in file_list:
            tmp = io.imread(file)
            img.append(tmp)
        self.X = np.asarray(img)
        # load segmented-images
        seg_list = glob.glob(seg_path+"/*.gif")
        seg_list.sort()
        seg_img = []
        for seg in seg_list:
            tmp = io.imread(seg)
            seg_img.append(tmp)
        self.Y = np.asarray(seg_img)
        # load mask
        mask_list = glob.glob(mask_path+"/*.gif")
        mask_list.sort()
        mask_img = []
        for mask in mask_list:
            tmp = io.imread(mask)
            mask_img.append(tmp)
        self.mask = np.asarray(mask_img)
        
    def load_HRF(self, 
                 file_path="D:/vessel dataset/HRF/all/images",
                 seg_path="D:/vessel dataset/HRF/all/manual1", 
                 mask_path="D:/vessel dataset/HRF/all/mask"
                 ):
        # load original images
        file_list = glob.glob(file_path+"/*.jpg")
        file_list.sort()
        img = []
        for file in file_list:
            tmp = io.imread(file)
            img.append(tmp)
        self.X = np.asarray(img)
        # load segmented-images
        seg_list = glob.glob(seg_path+"/*.tif")
        seg_list.sort()
        seg_img = []
        for seg in seg_list:
            tmp = io.imread(seg)
            seg_img.append(tmp)
        self.Y = np.asarray(seg_img)
        # load mask
        mask_list = glob.glob(mask_path+"/*.tif")
        mask_list.sort()
        mask_img = []
        for mask in mask_list:
            tmp = io.imread(mask)
            tmp = color.rgb2gray(tmp)
            mask_img.append(tmp)
        self.mask = np.asarray(mask_img)
    
    def load_STARE(self,
                   file_path="D:/vessel dataset/STARE/stare-images", 
                   seg_path="D:/vessel dataset/STARE/labels-ah", 
                   mask_path="D:/vessel dataset/STARE/msf-images"
                   ):
        # load original images
        file_list = glob.glob(file_path+"/*.ppm")
        file_list.sort()
        img = []
        for file in file_list:
            tmp = io.imread(file)
            img.append(tmp)
        self.X = np.asarray(img)
        # load segmented-images
        seg_list = glob.glob(seg_path+"/*.ppm")
        seg_list.sort()
        seg_img = []
        for seg in seg_list:
            tmp = io.imread(seg)
            seg_img.append(tmp)
        self.Y = np.asarray(seg_img)
        # load mask
        mask_list = glob.glob(mask_path+"/*.ppm")
        mask_list.sort()
        mask_img = []
        for mask in mask_list:
            tmp = io.imread(mask)
            mask_img.append(tmp)
        self.mask = np.asarray(mask_img)
        
    def random_crop(self, height=224, width=224, nums_per_img=100):
        batch_size, img_h, img_w, _ = self.X.shape
        

if __name__ == '__main__':
    data = retinalDataLoader()
    # data.load_CHASEDB1()
    # data.load_DRIVE()
    # data.load_HRF()
    data.load_STARE()
    pdb.set_trace()
    print("end")
    
    
    