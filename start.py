import sys 
import app.DataPreProcessor as dpp
import app.VideoInput as vIp
import app.CNN as cnn
import numpy as np
import app.ConfusionMatrix as cm
import app.Tester as Tester
import app.LiveCam as lcm
import playsound

class Start():
      # argumentList = sys.argv 
    def loadDataSets(self):
        # playsound.playsound('beep.mp3', True)
        print("Starting .............................................................")
        dataPreProcessor = dpp.DataPreProcessor()
        self.training_data = dataPreProcessor.create_training_data()
        print("Training Data processed ! ............................................")
        return self.training_data

    def loadModel(self):
        cnn_algo = cnn.ConvolutionalNeuralNetwork()
        cnn_model = cnn_algo.create_model(self.training_data)
        print("Model Loaded ! .......................................................")
        return cnn_model

    def confusionMatrix(self, cnn_model):
        print("Confusion Matrix .....................................................")
        con_mat, class_w_acc =  cm.confusion_matrix_fn(cnn_model)
        return con_mat

    def videoInputTest(self,cnn_model,fileUrl):
        print("Video Input .........................................................")
        videoInput = vIp.VideoInput()
        videoInput.start_capture(cnn_model,fileUrl)
        return "Done"

    def testOnUnknown(self, cnn_model):
        print("Testing Unknown .....................................................")
        tester = Tester.Tester(cnn_model)
        tester.test()
        return "Done"

    def liveInputTest(self ,cnn_model):
        print("Live Input ..........................................................")
        livecam = lcm.LiveCam()
        livecam.start_capture(cnn_model)
        return "Done"
