import cv2
import os
import warnings
import shutil
import sys
import argparse
import tkinter as tk
import sklearn
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree
import sklearn.tree._utils
import sklearn.utils._cython_blas
#from matplotlib import pyplot as plt
import numpy as np
#from k_means import k_means_via_sklearn
from main_window import *
from constants import *



def refine_binary(binary_img, kernel_size=3):
    kernel = np.ones((kernel_size + 2, kernel_size), np.uint8)
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


########################################################################
##########  BASIC COLOR SEGMENTATION USING THE FORMULA R/G < p1 ########
##########  B/G < p2 and 2*G - R - B > p3                       ########
########################################################################

def acp_canopeo_binary(img, p1=P1, p2=P2, p3=P3):  ## BGR Channels
    b, g, r = cv2.split(img)
    threshold_indices = g < 10
    g[threshold_indices] = 0

    indices = np.where(np.logical_and(np.logical_and(  # r/g < p1, b/g < p2),
        np.divide(r, g, np.ones_like(r, dtype=float) * 5, where=g != 0) < p1,
        np.divide(b, g, np.ones_like(b, dtype=float) * 5, where=g != 0) < p2),
        2 * g - r - b > p3))
    binary = np.zeros(img.shape, img.dtype)
    binary[indices] = 255
    # refine the image

    refined_binary = refine_binary(binary, 5)
    cv2.imwrite(os.path.join(IMAGE_FOLDER_PATH, 'segmented_unrefined_' + IMAGE_NAME), cv2.bitwise_and(img, binary))

    colored_binary = cv2.bitwise_and(img, refined_binary)
    return refined_binary, colored_binary


def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 300)
    return edges


def adjust_brightness(img):
    brightness = np.sum(img) / (255 * img.shape[0] * img.shape[1])
    minimum_brightness = 1
    ratio = brightness / minimum_brightness
    if ratio >= 1:
        return img
    return cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)


def k_means(img, n_clusters=4):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(data=pixel_values, K=n_clusters, bestLabels=None, criteria=criteria,
                                      attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    # segmented_image[labels == 0] = [0, 0, 0]
    # segmented_image[labels == 1] = [0, 0, 0]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image, centers


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path) # change to current directory
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
    #get_means_of_rectangle()
    #main_window()
    """parser = argparse.ArgumentParser(description='Arguments for image segmentation')
    parser.add_argument('--seg', metavar='Images', type=int, nargs='+',
                        help='The images to be segmented')
    parser.add_argument('--dir', help='segment a whole direcotry')
    parser.add_argument('--set', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='Set the cluster-centers for the segmentation')

    args = parser.parse_args()
    print(args.accumulate(args.integers))"""
    #### first try to normalize the image

    """img = cv2.imread(os.path.join(IMAGE_FOLDER_PATH, IMAGE_NAME), cv2.IMREAD_COLOR)
    img = adjust_brightness(img)
    segmented, colored, centers = k_means_via_sklearn(img)

    
    #segmented, centers = k_means(img)
    centers = centers[np.argsort(centers[:, 1])] ## sort by green value (BGR / RGB invariant)
    print(centers)
    plt.imshow(segmented)
    plt.show()
    cv2.imwrite(os.path.join(IMAGE_FOLDER_PATH, 'segmented_' + IMAGE_NAME), segmented)
    cv2.imwrite(os.path.join(IMAGE_FOLDER_PATH, 'colored_' + IMAGE_NAME), colored)"""


    #cv2.imwrite(os.path.join(IMAGE_FOLDER_PATH, 'colored_' + IMAGE_NAME), img[segmented > [0,0,0]])

    #np.savetxt(IMAGE_NAME+"centers.csv", centers, delimiter=",")

    """  binary_img, colored_binary = acp_canopeo_binary(img, p1=0.95, p2=0.95, p3=25)

    edges = detect_edges(colored_binary)
    plt.imshow(edges)
    plt.show()
    cv2.imwrite(os.path.join(IMAGE_FOLDER_PATH, 'edges_' + IMAGE_NAME), edges)

    plt.imshow(colored_binary, cmap='RdYlGn')
    plt.show()
    cv2.imwrite(os.path.join(IMAGE_FOLDER_PATH, 'segmented_'+IMAGE_NAME), colored_binary)
    """

    # plt.imshow(binary_img, cmap='binary', interpolation='bicubic')
    # plt.show()

    # numpy_horizontal = np.vstack((img_red,img_green, img_blue))
    # cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
    # cv2.imshow('RGB', result_img)
    # cv2.resizeWindow('RGB', 600, 900)

    # cv2.waitKey(0)

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
