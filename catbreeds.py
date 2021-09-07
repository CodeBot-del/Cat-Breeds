from __future__ import absolute_import, division, print_function
#from catsAndDogs import BATCH_SIZE, IMG_SHAPE
import os                     
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Load the cats breed dataset
_URL = "https://github.com/CodeBot-del/Cat-Breeds/blob/main/catmodel.zip"
_dir = tf.keras.utils.get_file('catmodel.zip',  origin = _URL, extract=True)

dir_base = os.path.dirname(_dir)
#print(dir_base)

base_dir = os.path.join(os.path.dirname(_dir), 'catmodel')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

#get the path for the train set of cats
train_abby_dir = os.path.join(train_dir, 'Abyssinian')
train_amer_dir = os.path.join(train_dir, 'American_bobtail')
train_bali_dir = os.path.join(train_dir, 'Balinese')
train_ben_dir = os.path.join(train_dir, 'Bengal')
train_cal_dir = os.path.join(train_dir, 'Calico')

#get path for the validation set of cats
validation_abby_dir = os.path.join(validation_dir, 'Abyssinian')
validation_amer_dir = os.path.join(validation_dir, 'American_bobtail')
validation_bali_dir = os.path.join(validation_dir, 'Balinese')
validation_ben_dir = os.path.join(validation_dir, 'Bengal')
validation_cal_dir = os.path.join(validation_dir, 'Calico')

#Look how many images of cats we have in our datasets
num_abby_tr = len(os.listdir(train_abby_dir))
num_amer_tr = len(os.listdir(train_amer_dir))
num_bali_tr = len(os.listdir(train_bali_dir))
num_ben_tr = len(os.listdir(train_ben_dir))
num_cal_tr = len(os.listdir(train_cal_dir))

num_abby_val = len(os.listdir(validation_abby_dir))
num_amer_val = len(os.listdir(validation_amer_dir))
num_bali_val = len(os.listdir(validation_bali_dir))
num_ben_val = len(os.listdir(validation_ben_dir))
num_cal_val = len(os.listdir(validation_cal_dir))

total_train = num_abby_tr + num_amer_tr + num_bali_tr + num_ben_tr + num_cal_tr
total_val = num_abby_val + num_amer_val + num_bali_val + num_ben_val + num_cal_val

print('total training Abyssinian images: ', num_abby_tr)
print('total training American bobtail images: ', num_amer_tr)
print('total training Balinese images: ', num_bali_tr)
print('total training Bengal images: ', num_ben_tr)
print('total training Calico images: ', num_cal_tr)

print('total validation Abyssinian images: ', num_abby_val)
print('total validation American bobtail images: ', num_amer_val)
print('total validation Balinese images: ', num_bali_val)
print('total validation Bengal images: ', num_ben_val)
print('total validation Calico images: ', num_cal_val)
print('--------')
print('total training images: ', total_train)
print('total validation images: ', total_val)

#setting model parameters
BATCH_SIZE = 5
IMG_SHAPE = 150 #cat images to be 150*150 pixels

#Data preparation
train_image_generator = ImageDataGenerator(rescale=1./255)     #generator for our training images
validation_image_generator = ImageDataGenerator(rescale=1./255) #generator for our validation images


#flow_from_directory method will load images from the disk, apply rescaling, and resize them
train_data_gen = train_image_generator.flow_from_directory(batch_size= BATCH_SIZE, directory= train_dir, shuffle=True, 
                                                            target_size=(IMG_SHAPE,IMG_SHAPE), class_mode='sparse')

validation_data_gen = validation_image_generator.flow_from_directory(batch_size = BATCH_SIZE, directory= validation_dir, shuffle=False, #shuffle false for validation as we want to verify
                                                            target_size=(IMG_SHAPE,IMG_SHAPE),class_mode='sparse')

#Visualizing the model 
sample_training_images, _ = next(train_data_gen)
#this function will plot images in grid of 1 row and 5 columns
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten() 
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

#plot_images(sample_training_images[:5]) #plot from 0 - 4

#Create the Model
#the model consists of four convolutional blocks with a max pool in each of them. then we have a total of 512 neurons/units
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SHAPE,IMG_SHAPE,3)), #convolutional block
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),                           #convolutional block
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),                          #convolutional block
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),                          #convolutional block
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')                                  #output layer
])

#compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model summary
print(model.summary())
EPOCHS  = 100 
history = model.fit_generator(train_data_gen,steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
epochs = EPOCHS, validation_data = validation_data_gen, validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))))

##APPLY AUGMENTATION to minimize overfitting


