from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

try:
    resample_method = Image.Resampling.LANCZOS
except AttributeError:
    resample_method = Image.LANCZOS
def gen_labels():
    train = './Data/Train'
    train_generator = ImageDataGenerator(rescale = 1/255)

    train_generator = train_generator.flow_from_directory(train,
                                                        target_size = (300,300),
                                                        batch_size = 32,
                                                        class_mode = 'sparse')
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    return labels

def preprocess(image):
    image=np.array(image.resize((300,300),resample=resample_method))
    image=np.array(image,dtype='uint8')
    image=np.array(image)/255.0

    return image
