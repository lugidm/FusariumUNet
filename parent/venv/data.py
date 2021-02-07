##################################### The script for all data-manipulations
import os
import cv2
from constants import *
import shutil
import numpy as np
import math as math
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("..", KERAS_WD)

# IF YOU CHANGE THE RESOLUTION, MAKE SURE; THAT YOU DELETED THE /label and /image folder - it will update them automatically this way. 
"""This function divides the big images into smaller patches
@:param new_image_size: the new image size (512,512)
@:param path: the path of the original (big) images
@:param mask_path: the name of the according label image
@:param relevance_mask: the name of the mask which defines the area, where the smaller mask is relevant. Everythin in this
    mask should be annotated (e.g. parcel -> ears or row->ears)
@:param relevance_mask_percentage: the amount the relevance_mask has to occupy the input image for beeing relevant"""


def prepare_images(new_image_size=(512, 512), path=os.path.join("..", "..", ONLINE_RESULTS_DIRECTORY),
                   mask_path=EAR_LABEL_IMAGE,
                   relevance_mask=PARZELLE_LABEL_IMAGE, relevance_mask_percentage=0.75,
                   output_path=os.path.join(DATA_PATH, RESIZED),
                   original_image_name=DOWNLOADED_IMAGE):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        shutil.rmtree(output_path)
        os.mkdir(output_path)
        print("removed old resized-directory")
    relevant_list = dict()
    for folder in os.listdir(path):
        relevant_list[folder] = list()
        for image in os.listdir(os.path.join(path, folder)):
            if image in relevance_mask:
                label = cv2.imread(os.path.join(path, folder, image), cv2.IMREAD_GRAYSCALE)
                width, height = np.shape(label)
                non_zeros_x, non_zeros_y = np.nonzero(label)
                x_start = np.min(non_zeros_x)
                y_start = np.min(non_zeros_y)
                x_max = np.max(non_zeros_x)
                y_max = np.max(non_zeros_y)
                if x_max > width - new_image_size[0]:
                    x_max = width - new_image_size[0]
                if y_max > height - new_image_size[1]:
                    y_max = height - new_image_size[1]
                for x in range(x_start, x_max, new_image_size[0]):
                    for y in range(y_start, y_max, new_image_size[1]):
                        crop_label = label[x:x + new_image_size[0], y:y + new_image_size[1]]
                        n_zero = np.count_nonzero(crop_label)
                        if n_zero / (new_image_size[0] * new_image_size[1]) >= relevance_mask_percentage:
                            relevant_list[folder].append((x, y))

    print("Found some relevant image patches:")
    print(relevant_list)
    for image_folder in os.listdir(path):
        for image in os.listdir(os.path.join(path, image_folder)):
            if image == mask_path:  # label _image
                if not os.path.exists(os.path.join(output_path, image_folder)):
                    os.mkdir(os.path.join(output_path, image_folder))
                if not image_folder in relevant_list:
                    break
                img = cv2.imread(os.path.join(path, image_folder, image), cv2.IMREAD_GRAYSCALE)
                for x, y in relevant_list[image_folder]:
                    crop_img = img[x:x + new_image_size[0], y:y + new_image_size[1]]
                    cv2.imwrite(os.path.join(output_path, image_folder, "l" + str(y) + "," +
                                             str(x) + os.path.splitext(image)[1]), crop_img)

    for image_folder in os.listdir(path):
        for image in os.listdir(os.path.join(path, image_folder)):
            if original_image_name in image:
                if not os.path.exists(os.path.join(output_path, image_folder)):
                    os.mkdir(os.path.join(output_path, image_folder))
                if not image_folder in relevant_list:
                    break
                img = cv2.imread(os.path.join(path, image_folder, image))
                for x, y in relevant_list[image_folder]:
                    crop_img = img[x:x + new_image_size[0], y: y + new_image_size[1]]
                    cv2.imwrite(os.path.join(DATA_PATH, RESIZED, image_folder, "o" + str(y) + "," +
                                             str(x) + os.path.splitext(image)[1]), crop_img)

"""This function returns the name of all patched images (see prepare_images(...))
@:param parent_directory: should be consistent to the output_path of prepare_images(...)
@:param file_ending_original: .jpg
"""
def get_all_resized_image_names(parent_directory, file_ending_original=".jpg"):
    image_list = list()
    counter = 0
    for image_folder in os.listdir(parent_directory):
        for image in os.listdir(os.path.join(parent_directory, image_folder)):
            if "l" in image:
                image_list.append(dict())
                coordinates = image[1:-4]
                image_list[counter]['label'] = os.path.abspath(os.path.join(parent_directory, image_folder, image))
                image_list[counter]['original'] = os.path.abspath(os.path.join(parent_directory, image_folder, "o" +
                                                                               coordinates + file_ending_original))
                counter += 1
    return image_list


def get_offset(x, y, input_size, desired_size):

    if y < input_size[1] // desired_size[1] and x < input_size[0] // desired_size[0]:
        offset_x = x * desired_size[0]
        offset_y = y * desired_size[1]
    elif y >= input_size[1] // desired_size[1] and x < input_size[0] // desired_size[0]:
        offset_x = x * desired_size[0]
        offset_y = input_size[1] - desired_size[1]
    elif y < input_size[1] // desired_size[1] and x >= input_size[0] // desired_size[0]:
        offset_x = input_size[0] - desired_size[0]
        offset_y = y * desired_size[1]
    else:
        offset_x = input_size[0] - desired_size[0]
        offset_y = input_size[1] - desired_size[1]
    return offset_x, offset_y


"""This function returns a generator for small patches of a big images: useful for the predictions
@:param input_img: a (X,Y,3)- image-np.array (you get it with openCV)
@:param input_dim: a tuple(X,Y) of the image size. This function is not so smart
@:param desired_batch_size: e.g. (512,512)
"""
def split_image(input_img, input_dim, desired_batch_size):
    for x in range(0, math.ceil(input_dim[0] / desired_batch_size[0])):
        for y in range(0, math.ceil(input_dim[1] / desired_batch_size[1])):
            offset_x, offset_y = get_offset(x, y, input_dim, desired_batch_size)
            yield input_img[0, offset_x:offset_x + desired_batch_size[0], offset_y: offset_y + desired_batch_size[1], :]

"""This function glues the image-patches together to a big image (because the image gets fed to the model in a redundant way)
@:param batches: a list of (512,512,1)- image-np.arrays from the model
@:param batch_size: a tuple(512,512) of the batch size. This function is not so smart
@:param desired_img_shape: e.g. (1234,2456,3)
"""
def glue_img(batches, batch_size, desired_img_shape):
    glued_img = np.zeros(shape=desired_img_shape)
    counter = 0
    for x in range(0, math.ceil(desired_img_shape[0] / batch_size[0])):
        for y in range(0, math.ceil(desired_img_shape[1] / batch_size[1])):
            i = batches[counter][:, :, 0]
            counter += 1
            x_offset, y_offset = get_offset(x, y, desired_img_shape, batch_size)
            glued_img[x_offset:x_offset + batch_size[0], y_offset:y_offset + batch_size[0]] = i
    return glued_img

"""
img_path = "/home/lugi/Bacc/Heuristik/ABE00097.jpg"
img = image.load_img(img_path)
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
splt= split_image(img, img.shape[1:], (512,512))
splt_list = list()
[splt_list.append(s) for s in splt]

plt.imshow(glue_img(splt_list, (512, 512), np.shape(img)))
plt.show()
"""

#### just a gescheitertes Projekt

class CustomPredictionGenerator(keras.utils.Sequence):
    def __init__(self, image_path, batch_size=4, target_dim=(512, 512), n_channels=3, n_classes=1, rescale=1,
                 callback_function=None):
        self.target_dim = target_dim
        self.batch_size = batch_size
        self.image_path = image_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.whole_image = image.load_img(image_path)
        self.img = np.true_divide(np.expand_dims(self.whole_image, axis=0), 1 / rescale)
        self.num_sub_images = (math.ceil(np.shape(self.img)[1] / self.target_dim[0]) *
                               math.ceil(np.shape(self.img)[2] / self.target_dim[1]))
        self.gen = split_image(self.img, np.shape(self.img)[1:], self.target_dim)
        self.so_far = 0
        self.callback_function = callback_function
        self.indices = []

    def __data_generation(self, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.target_dim[0], self.target_dim[1], self.n_channels))
        # Generate data
        if self.so_far > math.ceil(self.num_sub_images / self.batch_size) * self.batch_size:
            return None# now all batches have to be full

        for i in range(self.batch_size):
            if index == -1: # dont count this one
                print("dont count")
                X[i, ] = np.zeros((1, self.target_dim[0], self.target_dim[1], self.n_channels))
            else:
                self.so_far += 1
            try:
                X[i, ] = next(self.gen)
            except StopIteration:
                X[i,] = np.zeros((1, self.target_dim[0], self.target_dim[1], self.n_channels))
            if self.callback_function is not None:
                self.callback_function(self.so_far-1)

        return X

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(self.num_sub_images / self.batch_size)

    def __getitem__(self, index):
        if index in self.indices:
            return self.__data_generation(-1)
        self.indices.append(index)
        return self.__data_generation(index)

    def __get__(self):
        return self.__data_generation(0)

    def __next__(self):
        return self.__data_generation(0)



