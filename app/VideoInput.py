import cv2 
import numpy as np
import app.DataPreProcessor as dpp
import app.CNN as cnn
import app.ConfusionMatrix as cm
import playsound
import app.Utils as utils
import Queue as queue
from threading import Thread

def beep_function():
    playsound.playsound('./audio/beep.mp3', True)

class VideoInput():	
	def start_capture(self , model, fileUrl):
		cap = cv2.VideoCapture(fileUrl)
		prev = 0
		count = 0
		self.predicted_class = 0
		output = ""
		font = cv2.FONT_HERSHEY_SIMPLEX
		x = 10 
		y = 20
		i = 0
		q = queue.Queue(maxsize = 10)
		fcount = 0
		ret, frame = cap.read()
		while(cap.isOpened()):
			ret, frame = cap.read()
			if ret == True:
				count += 1
				gray = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
				# print(str(count)  + " " + str(prev))	
				if count - prev == 2:
					i += 1
					gray_scale , output, model_out = self.predict_output(gray , model)
					count = 0
					prev = 0
					self.predicted_class = np.argmax(model_out)
					if q.full() == True:
						if fcount >= 5:
							print("---------beep---------")
							thread = Thread(target = beep_function)
							thread.start()
							# playsound.playsound('beep.mp3', True)
						fcount -= q.get()
					if self.predicted_class == 0:
						q.put(0)
					else:
						q.put(1)
						fcount += 1
					print(output)
					cv2.putText(gray,output,(10,100), font,1,(0,255,0),2,cv2.LINE_AA)
					cv2.imwrite( "./output_video/"+str(i)+".jpg", gray);
					cv2.imshow('frame',gray)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			else:
				break
		cap.release()
		cv2.destroyAllWindows()
		return "End"
	def predict_output(self , image , model):
		image = cv2.resize(image, (utils.IMG_SIZE, utils.IMG_SIZE)) 
		image = np.array(image)
		data = image.reshape(utils.IMG_SIZE, utils.IMG_SIZE, 3)
		model_out = model.predict([data])
		print(utils.image_class[np.argmax(model_out)])
		return data , "" + str(utils.image_class[np.argmax(model_out)]), model_out



