import cv2
import numpy as np
import os         
from random import shuffle 
from tqdm import tqdm      
import app.Utils as utils
import matplotlib.pyplot as plt

class DataProcessor():
    train_npy = []
    def create_label(self,index):
        labels = [0] * 10
        labels[index] = 1
        return labels
	
    def create_training_data(self):

        if os.path.isfile(utils.TRAIN_DATA_NPY):
            return utils.TRAIN_DATA_NPY
        else:     
            for category in utils.CATEGORIES: 
                path = os.path.join(utils.TRAIN_DATADIR,category)  
                class_num = utils.CATEGORIES.index(category)  
                for img in tqdm(os.listdir(path)):  
                    try:
                        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
                        new_array = cv2.resize(img_array, (utils.IMG_SIZE, utils.IMG_SIZE)) 
                        self.train_npy.append([np.array(new_array), self.create_label(class_num)])
                    except Exception as e: 
                        print("Exception in dataprocessing : " + str(e))
                        pass
            shuffle(self.train_npy)
            np.save(utils.TRAIN_DATA_NPY, self.train_npy)
            return utils.TRAIN_DATA_NPY

    
   