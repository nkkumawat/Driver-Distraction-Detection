IMG_SIZE = 128
LR = 1e-3 
MODEL_NAME = "cnn-drive"
CATEGORIES = ['c0' , 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8' , 'c9']
TRAIN_DATADIR = './datasets/train/'
TEST_DATADIR = './datasets/test/'
TRAIN_DATA_NPY = './npy_arrays/train_data.npy'
TRAIN_DATA_COLOR_NPY = './npy_arrays/train_color_data.npy'
TEST_DATA_NPY = './npy_arrays/test_data.npy'
TEST_UNKNOWN = './datasets/testunknown'
image_class = ['Safe Driving',
  'Texting Right' ,
  'Talking Right' , 
  'Texting Left ' ,
  'Talking Left ' ,
  'Adjust Radio/Music Player' , 
  'Drinking',
  'Reaching Behind', 
  'Hair and Makeup' , 
  'Talking to Passenger']