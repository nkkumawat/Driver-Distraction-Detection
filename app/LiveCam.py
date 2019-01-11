import cv2 
import numpy as np
import app.CNN as cnn
import app.Utils as utils
import Queue as queue
import playsound
from threading import Thread

def beep_function():
    playsound.playsound('./audio/beep.mp3', True)


class LiveCam():
    def start_capture(self , model):
        cap = cv2.VideoCapture(0)
        prev = 0
        count = 0
        self.predicted_class = 0
        output = ""
        font = cv2.FONT_HERSHEY_SIMPLEX
        x = 10 
        y = 20 
        q = queue.Queue(maxsize = 10)
        fcount = 0
        while(True):
            ret, frame = cap.read()
            conti = True
            if conti == True:
                # print(str(count)  + " " + str(prev))
                gray = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
                if count == 2:
                    # print(str(count)  + " " + str(prev))
                    gray_scale,output, model_out = self.predict_output(gray , model)
                    count = 0
                    prev = 0
                    self.predicted_class = np.argmax(model_out)
                    if q.full() == True:
                        if fcount >= 5:
                            print("---------beep---------")
                            thread = Thread(target = beep_function)
                            thread.start()
                        fcount -= q.get()
                    if self.predicted_class == 0:
                        q.put(0)
                    else:
                        q.put(1)
                        fcount += 1
                    cv2.putText(gray,output,(10,100), font, 1,(0,255,0),2,cv2.LINE_AA)
                    cv2.imshow('frame',gray)
                count += 1 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else :
                break
        cap.release()
        cv2.destroyAllWindows()
        return "End"

    def predict_output(self , image , model):
        image = cv2.resize(image, (utils.IMG_SIZE, utils.IMG_SIZE)) 
        image = np.array(image)
        data = image.reshape(utils.IMG_SIZE, utils.IMG_SIZE, 3)
        model_out = model.predict([data])
        return data , "" + str(utils.image_class[np.argmax(model_out)]),model_out

      
