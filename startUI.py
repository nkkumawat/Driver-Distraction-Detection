import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
import start as start
 
class App(QWidget):
 
    def __init__(self):
        super(App , self).__init__()
        self.title = 'Driver DISTRACTION'
        self.left = 25
        self.top = 25
        self.width = 1000
        self.height = 700

        self.strt = start.Start()
        self.training_datasets = self.strt.loadDataSets()
        self.cnn_model = self.strt.loadModel()
        self.con_matrix = self.strt.confusionMatrix(self.cnn_model)

        self.initUI()
        
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        self.button1 = QPushButton('Video Input', self)
        self.button1.setToolTip('Video Input')
        self.button1.move(250,100)
        self.button1.resize(150,40)
        self.button1.clicked.connect(self.video_input_fn)

        self.button2 = QPushButton('Live Feed', self)
        self.button2.setToolTip('Live Camera')
        self.button2.move(405,100)
        self.button2.resize(150, 40)
        self.button2.clicked.connect(self.live_input_fn)

        self.button3 = QPushButton('Test on Unknown', self)
        self.button3.setToolTip('Test on Unknown')
        self.button3.move(560,100)
        self.button3.resize(150, 40)
        self.button3.clicked.connect(self.test_unknown)

        self.label1 = QLabel("",self)
        self.label1.move(700, 250)
        self.label1.resize(200,200)

        font = QtGui.QFont("Times", 20, QtGui.QFont.Bold) 
        self.label2 = QLabel("EARLY WARNING SYSTEM FOR DISTRACTED DRIVER",self)
        self.label2.move(150, 40)
        self.label2.setFont(font)

        self.label = QLabel(self)
        self.pixmap = QPixmap('graphs/clsacc.png')
        self.label.setPixmap(self.pixmap)
        self.label.move(100 ,200)
                            
        self.label3 = QLabel(self)
        self.pixmap1 = QPixmap('graphs/conmtrx.png')
        self.label3.setPixmap(self.pixmap1)
        self.label3.move(500 ,200)

        self.label4 = QLabel("Classwise Accuracy ",self)
        self.label4.move(100 ,170)
        self.label5 = QLabel("Confusion Matrix ",self)
        self.label5.move(500 ,170)
 
        self.show()
 

    def FileChooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Mp4 (*.mp4)", options=opt)
        return fileUrl
        
    @pyqtSlot()  
    def video_input_fn(self):
        fileUrl = self.FileChooser()
        print(fileUrl)
        self.strt.videoInputTest(self.cnn_model,fileUrl[0])

    @pyqtSlot()
    def live_input_fn(self):
        print("Live feed")
        self.strt.liveInputTest(self.cnn_model)

    @pyqtSlot()
    def test_unknown(self):
        self.strt.testOnUnknown(self.cnn_model)
        

    # @pyqtSlot()
    # def confusion_matrix(self):
    #     self.label1.setText("Loading .................")
    #     # self.con_matrix = self.strt.confusionMatrix(self.cnn_model)
    #     # hell    o = bar.App(self)
    #     # hello.show()
    #     print("nk")
    #     return None

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
