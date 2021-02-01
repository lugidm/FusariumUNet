import tensorflow as tf

from tensorflow import keras
#from keras_unet.models import vanilla_unet

from keras_unet.losses import jaccard_distance
from keras_unet.metrics import jaccard_coef
import keras_unet as kunet
from keras_unet.models import satellite_unet

#from model import *
from data import (prepare_images, get_all_resized_image_names)
from tensorflow.keras.optimizers import Adam
#import keras
import os
from constants import *
#import PIL
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
#from data import *
#from model import *
from tensorflow.keras.preprocessing.image import load_img


CONTINUE_LEARNING = True # True means, the old model gets loaded from KERAS_WD (defined in bzw. constants.py) before learning => False means a new model gets created
BATCH_SIZE = 4 # Batch-size is the number of 512*512 images, that are fed per epoch as training data, its better when kept small (UNet says so in their paper)
NO_OF_EPOCHS = 15 # After 15 epochs the learning finishes

# IF YOU CHANGE THE RESOLUTION, MAKE SURE; THAT YOU DELETED THE /label and /image folder - it has to be updated
NEW_IMAGE_SIZE = (512, 512, 3) # should be consistent with the IMAGE size used when predicting with unet- 512*512 is fittable in normal RAM
NEW_IMAGE_SIZE_2d = (512, 512) 
FILTER_SIZE = 64 #params for unet model
NUM_LAYERS = 4 # same as above

DATA_PATH = os.path.join("..", KERAS_WD) #here the model gets stored (and its log-outputs) -> folder should be already there before launching this script
#######DATA-structure##########
#../DATA_PATH:
#		/image:
#			/train:
#				- M images (.jpg)
#			/val:
#				- N images (original RGB-images.jpg)
#		/label
#			/train:
#				- M label images (.bmp)
#			/val:
#				- N label images (mask-images .bmp)
#		/weights: here the model gets stored and back-up'd after each epoch, furthermore, the logs are stored here in log.out
#			/model.h5
#			/log.out

IMG_PATH = os.path.join(DATA_PATH, "image") #the directory-structure is crucial. Because I use a standard-image generator. (otherwise it is a buggy road)
MASK_PATH = os.path.join(DATA_PATH, "label")
TRAIN_MASK_PATH = os.path.join(MASK_PATH, "train", "train")
TRAIN_IMAGE_PATH = os.path.join(IMG_PATH, "train", "train")
VAL_MASK_PATH = os.path.join(MASK_PATH, "val", "val")
VAL_IMAGE_PATH = os.path.join(IMG_PATH, "val", "val")

if os.path.exists(TRAIN_MASK_PATH) and os.path.exists(VAL_MASK_PATH):
	NO_OF_TRAINING_IMAGES = len(os.listdir(TRAIN_MASK_PATH))
	NO_OF_VAL_IMAGES = len(os.listdir(VAL_MASK_PATH))

WEIGHTS_PATH = os.path.join(DATA_PATH, "weights")


##### split the image names_list
def split_train_set(all_images, test_ratio):
	shuffled_indices = np.random.permutation(len(all_images))
	test_set_size = int(len(all_images) * test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return np.array(all_images)[train_indices.astype(int)].tolist(), np.array(all_images)[
		test_indices.astype(int)].tolist()


#### save the training and validation set in KERAS_WD/label and KERAS_WD/image
def save_train_and_test_set(train_set, test_set):
	if not os.path.exists(IMG_PATH):
		os.mkdir(IMG_PATH)
	else:
		shutil.rmtree(IMG_PATH)
		os.mkdir(IMG_PATH)
	if not os.path.exists(MASK_PATH):
		os.mkdir(MASK_PATH)
	else:
		shutil.rmtree(MASK_PATH)
		os.mkdir(MASK_PATH)
	os.mkdir(os.path.join(IMG_PATH, "train"))
	os.mkdir(os.path.join(MASK_PATH, "train"))
	os.mkdir(os.path.join(MASK_PATH, "val"))
	os.mkdir(os.path.join(IMG_PATH, "val"))
	os.mkdir(TRAIN_IMAGE_PATH)
	os.mkdir(TRAIN_MASK_PATH)
	os.mkdir(VAL_MASK_PATH)
	os.mkdir(VAL_IMAGE_PATH)
	file_counter = 0
	for tr in train_set:
		if "label" in tr:  # only save when label is present
			label_image = cv2.imread(tr["label"], cv2.IMREAD_GRAYSCALE)
			label_image[label_image > 0] = 1
			cv2.imwrite(os.path.join(TRAIN_MASK_PATH, str(file_counter) + FILE_FORMAT), label_image)
			shutil.copy(tr['original'],
						os.path.join(TRAIN_IMAGE_PATH, str(file_counter) + os.path.splitext(tr['original'])[1]))
		file_counter += 1
	file_counter = 0
	for ts in test_set:
		if "label" in ts:
			label_image = cv2.imread(ts["label"])
			label_image[label_image > 0] = 1
			cv2.imwrite(os.path.join(VAL_MASK_PATH, str(file_counter) + FILE_FORMAT), label_image)
			shutil.copy(ts["original"], os.path.join(VAL_IMAGE_PATH, str(file_counter) +
													 os.path.splitext(ts['original'])[1]))
		file_counter += 1


def main():
	keras.backend.clear_session()
	# The original images should be downloaded to ../../ONLINE_RESULTS_DIRECTORY so just download the annotation_results folder from BOKU-drive and place it correspondingly into your FS
	if not os.path.exists(VAL_IMAGE_PATH) or not os.path.exists(TRAIN_MASK_PATH):
		# crop images according to our hierarchy: only ears in plots are considered
		prepare_images(new_image_size=NEW_IMAGE_SIZE_2d, path=os.path.join("..", "..", ONLINE_RESULTS_DIRECTORY),
					   mask_path=EAR_LABEL_IMAGE)
		# get all image-names
		all_images = get_all_resized_image_names(parent_directory=os.path.join(DATA_PATH, RESIZED),
												 file_ending_original=".jpg")
		#split the images into train / val-set
		train_set, test_set = split_train_set(all_images, 0.2)
		save_train_and_test_set(train_set, test_set)
		print("restart the program")
		# don't know why but isnt it funny?
		return
	# create 4 image generators, because it only runs like this. Could be improved but...
	train_image_Generator = kunet.utils.ImageDataGenerator(horizontal_flip=True, rescale=(1.0/255))
	train_mask_Generator = kunet.utils.ImageDataGenerator(horizontal_flip=True)
	val_image_Generator = kunet.utils.ImageDataGenerator(rescale=(1.0/255))
	val_mask_Generator = kunet.utils.ImageDataGenerator()

	train_image_generator = train_image_Generator.flow_from_directory(os.path.join(IMG_PATH, "train"), batch_size=BATCH_SIZE,
															  target_size=NEW_IMAGE_SIZE_2d, seed=412, class_mode='input')
	train_mask_generator = train_mask_Generator.flow_from_directory(os.path.join(MASK_PATH, "train"), batch_size=BATCH_SIZE,
															 target_size=NEW_IMAGE_SIZE_2d, color_mode="grayscale",
															 seed=412, class_mode='input')
	val_image_generator = val_image_Generator.flow_from_directory(os.path.join(IMG_PATH, "val"), batch_size=BATCH_SIZE,
														  target_size=NEW_IMAGE_SIZE_2d,
														  seed=42, class_mode='input')
	val_mask_generator = val_mask_Generator.flow_from_directory(os.path.join(MASK_PATH, "val"), batch_size=BATCH_SIZE,
														 target_size=NEW_IMAGE_SIZE_2d, color_mode="grayscale",
														 seed=42, class_mode='input')

	### look into the generator:
	""" x, y = train_image_generator.next()
			 plt.imshow(x[0])
			 plt.show()
			 return"""

	train_generator = zip(train_image_generator, train_mask_generator)
	val_generator = zip(val_image_generator, val_mask_generator)
	# creates a new model.h5-file (and overwrites the old!!!!)
	if not CONTINUE_LEARNING:
		print("attention I will start learning from 0!")
		model = satellite_unet(NEW_IMAGE_SIZE, num_classes=1)
		# model = unet()
		# model = vanilla_unet(NEW_IMAGE_SIZE, num_classes=1, filters=FILTER_SIZE, num_layers=NUM_LAYERS)
		opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		"""Arguments
			lr: float >= 0. Learning rate.
			rho: float >= 0.
			epsilon: float >= 0. Fuzz factor.
			decay: float >= 0. Learning rate decay over each update."""
		# my_optimizer = rmsprop(lr=1, rho=0.9, epsilon=1e-08, decay=0.05)
		model.compile(optimizer=opt, loss=jaccard_distance, metrics=jaccard_coef)
	else:
		print("I WILL USE THE ALREADY LEARNED WEIGHTS from output.h5")
		model = keras.models.load_model(os.path.join(DATA_PATH, "weights/output.h5"), custom_objects={'jaccard_distance': jaccard_distance, 'jaccard_coef':jaccard_coef})
		model.load_weights(os.path.join(DATA_PATH, "weights/output.h5"))
	#backup after each epoch
	checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(WEIGHTS_PATH, "output.h5"), monitor='loss',
												 verbose=1)
	#nice to have
	csv_logger = keras.callbacks.CSVLogger(os.path.join(WEIGHTS_PATH, "log.out"), append=True, separator=';')
	#never reached this condition but could become handy. But probably one has to change min_delta and mode?
	earlystopping = keras.callbacks.EarlyStopping(monitor='loss', verbose=1,
												  min_delta=0.01, patience=3, mode='min')
	# mode= min for mse, max for non-error-functions

	model.summary()
	#prints a summary

	model.fit(train_generator, epochs=NO_OF_EPOCHS, steps_per_epoch=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
			  validation_data=val_generator, validation_steps=(NO_OF_VAL_IMAGES // BATCH_SIZE),
			  callbacks=[checkpoint, csv_logger, earlystopping])
	# when learning is finished, model gets saved in model.h5
	model.save(os.path.join(DATA_PATH, "model.h5"))
	return
	# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
	# model.fit_generator(myGen, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])
	# testGene = testGenerator("data/membrane/test")
	# results = model.predict_generator(testGene, 30, verbose=1)
	# saveResult("data/membrane/test", results)


if __name__ == '__main__':
	main()
