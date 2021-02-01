from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import warnings
import math
from point2d import Point2D
from PIL import Image, ImageTk
import os
import errno
from message_box import *
import shutil
import cv2
import numpy as np
import sys as sys
import re as re
import matplotlib.pyplot as plt

from constants import *
from annotation_tool import MeanWindow
from apply_k_means_window import ApplyKMeans


NOTHING_DONE = "N"
PARZELLE = "P"
EINZELREIHE = "R"
EAR = "E"


def delete_annotations():
    answer = messagebox.askyesnocancel("WARNING", "are you sure? deleting all previous annotation cannot be undone")
    if answer:
        shutil.rmtree(K_MEANS_RESULTS_PATH)


def delete_centers():
    answer = messagebox.askyesnocancel("WARNING", "are you sure? deleting the centers cannot be undone")
    if answer:
        shutil.rmtree(K_MEANS_CENTERS)


def learn_new():
    amount = learn()
    messagebox.showinfo("Found", "Found " + str(amount) + "annotated images. The resulting mean-colors of the labels " +
                        "are saved in: " + str(os.path.join(K_MEANS_CENTERS, K_MEANS_CENTERS_CSV)))


def calculate_dominant_color(pixels, n_colors):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    return palette[np.argmax(counts)]



class LearnCentersWindow:
    def __init__(self, master):
        self.root = master
        self.root.geometry('900x600')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.what_to_do_fr = tk.LabelFrame(self.root, text="What do you want to do?")
        self.what_to_do_fr.pack(fill='y', expand=True)

        self.annotate_new_centers_bt = tk.Button(self.what_to_do_fr, text="Annotate new cluster centers",
                                                 command=self.annotate_new_centers)
        self.annotate_new_centers_bt.pack()
        self.learn_new_centers_bt = tk.Button(self.what_to_do_fr, text="Learn new cluster centers of annotated data",
                                              command=learn_new)
        self.learn_new_centers_bt.pack()
        self.apply_bt = tk.Button(self.what_to_do_fr, text="apply K-Means algorithm on a picture", command=self.apply)
        self.apply_bt.pack()
        self.forget_centers_bt = tk.Button(self.what_to_do_fr, text="Forget the previously calculated centers",
                                          command=delete_centers, fg='red')
        self.forget_centers_bt.pack(side='bottom')
        self.delete_previos_annotations_bt = tk.Button(self.what_to_do_fr, text="Delete ALL previous annotations",
                                                       command=delete_annotations, fg='red')
        self.delete_previos_annotations_bt.pack(side='bottom')

    def annotate_new_centers(self):
        tl = tk.Toplevel(self.root)
        MeanWindow(master=tl, label_list=[SOIL_LABEL, HEALTHY_LABEL, DISEASED_LABEL, REST_LABEL],
                   label_names=["soil", "healthy", "diseased", "rest"],
                   label_image_names=[SOIL_LABEL_IMAGE, HEALTHY_LABEL_IMAGE, DISEASED_LABEL_IMAGE, REST_LABEL_IMAGE],
                   hierarchical=False,
                   save_online=False, offline_directory=K_MEANS_RESULTS_PATH)

    def apply(self):
        tl = tk.Toplevel(self.root)
        ApplyKMeans(tl)


def learn():
    num_colors_per_cluster = 5
    i = 0
    if not os.path.exists(K_MEANS_CENTERS):
        os.makedirs(K_MEANS_CENTERS)

    if not os.path.exists(K_MEANS_RESULTS_PATH):
        messagebox.showerror("ERROR",
                             "There are no annotated pictures yet or" + K_MEANS_RESULTS_PATH + "has been deleted")
        return
    for subdir, dirs, files in os.walk(K_MEANS_RESULTS_PATH):
        label_healthy = None
        label_rest = None
        label_diseased = None
        label_soil = None
        original = None
        skip_this = False
        if len(files) == 0:
            continue
        if os.path.exists(os.path.join(subdir, K_MEANS_COLORS)):  # this image-colors have been calculated already
            i += 1
            cluster_centers = np.genfromtxt(os.path.join(subdir, K_MEANS_COLORS), delimiter=',').astype(int)
        else:
            for filename in files:
                filepath = subdir + os.sep + filename
                if filepath.endswith(HEALTHY_LABEL_IMAGE):
                    label_healthy = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                elif filepath.endswith(DISEASED_LABEL_IMAGE):
                    label_diseased = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                elif filepath.endswith(SOIL_LABEL_IMAGE):
                    label_soil = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                elif filepath.endswith(REST_LABEL_IMAGE):
                    label_rest = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                elif DOWNLOADED_IMAGE in os.path.basename(filepath):
                    original = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if ((label_healthy is not None or label_diseased is not None or label_soil is not None or
                 label_rest is not None) and original is not None):  # there is something in the subdir
                i += 1
                if label_healthy is not None:
                    healthy = cv2.bitwise_and(original, original, mask=label_healthy)
                    healthy = healthy[healthy != 0]
                    healthy = np.float32(healthy.reshape((healthy.shape[0]//3, 3)))
                    healthy_color = calculate_dominant_color(healthy, num_colors_per_cluster)
                else:
                    healthy_color = np.zeros(3)
                if label_diseased is not None:
                    diseased = cv2.bitwise_and(original, original, mask=label_diseased)
                    diseased = diseased[np.where((diseased != [0, 0, 0]).all(axis=2))]
                    diseased = diseased[diseased != 0]
                    diseased = np.float32(diseased.reshape((diseased.shape[0] // 3, 3)))
                    diseased_color = calculate_dominant_color(diseased, num_colors_per_cluster)
                else:
                    diseased_color = np.zeros(3)
                if label_soil is not None:
                    soil = cv2.bitwise_and(original, original, mask=label_soil)
                    soil = soil[np.where((soil != [0, 0, 0]).all(axis=2))]
                    soil = soil[soil != 0]
                    soil = np.float32(soil.reshape((soil.shape[0] // 3, 3)))
                    soil_color = calculate_dominant_color(soil, num_colors_per_cluster)
                else:
                    soil_color = np.zeros(3)
                if label_rest is not None:
                    rest = cv2.bitwise_and(original, original, mask=label_rest)
                    rest = rest[np.where((rest != [0, 0, 0]).all(axis=2))]
                    rest = rest[rest != 0]
                    rest = np.float32(rest.reshape((rest.shape[0] // 3, 3)))
                    rest_color = calculate_dominant_color(rest, num_colors_per_cluster)
                else:
                    rest_color = np.zeros(3)
                cluster_centers = np.array([diseased_color, healthy_color, soil_color, rest_color], np.uint8)
                print(cluster_centers)
                np.savetxt(os.path.join(subdir, K_MEANS_COLORS), cluster_centers,
                           delimiter=',',
                           fmt='%d')
            else:
                skip_this = True
        if os.path.exists(os.path.join(K_MEANS_CENTERS, K_MEANS_CENTERS_CSV)) and not skip_this:
            old_cluster_centers = np.genfromtxt(os.path.join(K_MEANS_CENTERS, K_MEANS_CENTERS_CSV),
                                                delimiter=',').astype(int)
            new_cluster_centers = np.zeros((4, 3), dtype=np.uint8)
            for i in range(4):
                if np.all(old_cluster_centers[i] == 0):
                    new_cluster_centers[i] = cluster_centers[i]
                elif np.all(cluster_centers[i] == 0):
                    new_cluster_centers[i] = old_cluster_centers[i]
                else:
                    new_cluster_centers[i] = (old_cluster_centers[i] + cluster_centers[i]) / 2
            np.savetxt(os.path.join(K_MEANS_CENTERS, K_MEANS_CENTERS_CSV), new_cluster_centers,
                       delimiter=',',
                       fmt='%d')
        elif not skip_this:
            np.savetxt(os.path.join(K_MEANS_CENTERS, K_MEANS_CENTERS_CSV), cluster_centers,
                       delimiter=',',
                       fmt='%d')
    return i
