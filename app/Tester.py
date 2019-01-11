import cv2
import numpy as np
import os         
from random import shuffle 
from tqdm import tqdm      
import app.Utils as utils
import matplotlib.pyplot as plt

class Tester():
    font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self , model):
        self.model = model
    
    def test(self):
        i = 0
        for img in tqdm(os.listdir(utils.TEST_UNKNOWN)):  
            try:
                img_array = cv2.imread(os.path.join(utils.TEST_UNKNOWN,img) ,cv2.IMREAD_COLOR)  
                output = self.predict_output(img_array)
                # print(output)
                cv2.putText(img_array,output,(10,100), self.font,1,(255,0,0),2,cv2.LINE_AA)
                cv2.imwrite( "./output_test/output_image_{}".format(i)+".jpg", img_array)
                i += 1
            except Exception as e: 
                print("Exception Occured : " + str(e))
                pass
        print("prediction is DONE and output is saved in output folder.................")


    def predict_output(self, image):
        image = cv2.resize(image, (utils.IMG_SIZE, utils.IMG_SIZE)) 
        image = np.array(image)
        image = image.reshape(utils.IMG_SIZE, utils.IMG_SIZE, 3)
        model_out = self.model.predict([image])
        prediction = utils.image_class[np.argmax(model_out)]
        return prediction