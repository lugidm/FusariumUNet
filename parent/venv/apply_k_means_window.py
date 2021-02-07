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

from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

from constants import *
from pyramid_canvas import CanvasImage

initial_search_directory = "."
kernel_size = KERNEL_SIZE


def path_leaf(path):
    """    head, tail = os.path.split(path)
        return tail or os.path.basename(head)"""
    return re.sub('[^A-Za-z0-9]+', '_', path)


class ApplyKMeans:
    global kernel_size

    def __init__(self, master):
        self.root = master
        self.root.geometry('900x600')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.k_means_centers_path = os.path.join(K_MEANS_CENTERS, K_MEANS_CENTERS_CSV)
        self.filename = ""
        self.original_img = None
        self.segmented_img = None
        self.labels = None
        self.nr_centers_val = K_MEANS_NR_CENTERS
        self.nr_iterations_val = K_MEANS_NR_ITERATIONS
        self.resulting_img_path = None
        self.initial_directory = "."
        self.pre_centers = None
        self.center_names = ['diseased', 'healthy', 'soil', 'rest']
        self.displayed_centers = []
        self.button_fr = tk.Frame(master)
        self.button_fr.grid_rowconfigure(1, weight=1)
        self.button_fr.grid_rowconfigure(0, weight=1)
        self.button_fr.grid_columnconfigure(0, weight=1)
        self.button_fr.grid_columnconfigure(1, weight=1)
        self.button_fr.grid_columnconfigure(2, weight=1)

        self.original = tk.BooleanVar()

        self.ch0 = tk.BooleanVar()
        self.ch1 = tk.BooleanVar()
        self.ch2 = tk.BooleanVar()
        self.ch3 = tk.BooleanVar()
        self.ch4 = tk.BooleanVar()
        self.ch5 = tk.BooleanVar()
        self.ch6 = tk.BooleanVar()
        self.ch7 = tk.BooleanVar()
        self.ch8 = tk.BooleanVar()
        self.ch9 = tk.BooleanVar()
        self.wanted_clusters = [self.ch0, self.ch1, self.ch2, self.ch3, self.ch4, self.ch5, self.ch6, self.ch7,
                                self.ch8, self.ch9]
        self.pr0 = tk.StringVar()
        self.pr1 = tk.StringVar()
        self.pr2 = tk.StringVar()
        self.pr3 = tk.StringVar()
        self.pr4 = tk.StringVar()
        self.pr5 = tk.StringVar()
        self.pr6 = tk.StringVar()
        self.pr7 = tk.StringVar()
        self.pr8 = tk.StringVar()
        self.pr9 = tk.StringVar()
        self.percentages = [self.pr0, self.pr1, self.pr2, self.pr3, self.pr4, self.pr5, self.pr6, self.pr7,
                            self.pr8, self.pr9]

        self.search_bt = tk.Button(self.button_fr, text="Search Image", command=self.search)
        self.search_bt.grid(row=0, column=1)
        self.calculate_k_means_bt = tk.Button(self.button_fr, text="(Re)Run K-Means", command=self.calculate,
                                              fg='green')
        self.calculate_k_means_bt.grid(row=0, column=0)
        self.quit_bt = tk.Button(self.button_fr, text="QUIT", fg="red", command=self.root.destroy)
        self.quit_bt.grid(row=0, column=2)
        self.img_name_entry = tk.Entry(self.button_fr)
        self.img_name_entry.grid(row=1, columnspan=3, pady=2, sticky='EW')
        self.img_name_entry.insert(0, self.filename)
        self.button_fr.pack()

        self.frame_canvas = tk.Frame(self.root, bg='black', relief='sunken', bd=1)
        self.frame_canvas.rowconfigure(1, weight=1)
        self.frame_canvas.columnconfigure(1, weight=1)
        self.canvas = None
        self.frame_canvas.pack(fill=tk.BOTH, expand=1, side='left')
        self.all_options_fr = tk.Frame(self.root, relief='sunken', bd=1)
        self.all_options_fr.rowconfigure(0, weight=1)
        self.all_options_fr.pack(expand=1, side='right')
        self.original_or_cluster_fr = tk.Frame(self.all_options_fr, relief='sunken', bd=1)
        self.original_or_cluster_fr.grid(row=1, column=0)
        self.original_rd = tk.Radiobutton(self.original_or_cluster_fr, variable=self.original, value=True,
                                          text="original color", command=self.repaint_clusters)
        self.original_rd.grid(row=0, column=0)
        self.cluster_rd = tk.Radiobutton(self.original_or_cluster_fr, variable=self.original, value=False,
                                         text="color of cluster centers", command=self.repaint_clusters)
        self.cluster_rd.grid(row=0, column=1)


        self.all_scales_fr = tk.Frame(self.all_options_fr, relief='sunken', bd=1)
        self.all_scales_fr.rowconfigure(0, weight=1)
        self.all_scales_fr.grid(row=0, column=0, sticky='E')

        self.nr_centers_description_lb = tk.Label(self.all_scales_fr, text="#Cluster-Centers, \n4-5 is recommended",
                                                  relief='sunken')
        self.nr_centers_description_lb.grid(row=0, column=0)
        self.nr_centers_sc = tk.Scale(self.all_scales_fr, from_=2, to=10, resolution=1, name="nr_centers_scale")
        self.nr_centers_sc.set(K_MEANS_NR_CENTERS)
        self.nr_centers_sc.bind("<ButtonRelease-1>", self._update_value)
        self.nr_centers_sc.grid(row=1, column=0)

        self.nr_iterations_lb = tk.Label(self.all_scales_fr, text="#Iterations", relief='sunken')
        self.nr_iterations_lb.grid(row=0, column=1)
        self.nr_iterations_sc = tk.Scale(self.all_scales_fr, from_=1, to=20, resolution=1, name="nr_iterations_scale")
        self.nr_iterations_sc.set(K_MEANS_NR_ITERATIONS)
        self.nr_iterations_sc.bind("<ButtonRelease-1>", self._update_value)
        self.nr_iterations_sc.grid(row=1, column=1)

        self.check_fr = tk.Frame(self.all_options_fr, relief='sunken', bd=1)
        self.check_fr.rowconfigure(0, weight=1)
        self.check_fr.grid(row=2, column=0)
        self.check_lb = tk.Label(self.check_fr, relief='sunken', text='which clusters do you want to display?')
        self.check_lb.grid(row=0, column=0, columnspan=2)
        self.check_buttons = []
        self.percentages_labels = []
        for i in range(int(self.nr_centers_sc['to'])):
            self.percentages_labels.append(tk.Entry(self.check_fr, textvariable=self.percentages[i]))
            self.percentages[i].set("%")
            self.check_buttons.append(tk.Checkbutton(self.check_fr, text=str(i + 1), variable=self.wanted_clusters[i],
                                                     command=self.repaint_clusters))
            if i < 4:
                self.check_buttons[-1].configure(text=self.center_names[i])
        for i in range(self.nr_centers_val):
            self.check_buttons[i].grid(column=0, row=i + 1)
            self.percentages_labels[i].grid(column=1, row=i + 1)

        """self.save_img_bt = tk.Button(self.all_options_fr, text="save current picture", command=self.save)
        self.save_img_bt.grid(row=2)
        self.all_options_fr.pack()"""

    def search(self):
        global initial_search_directory
        filename = filedialog.askopenfilename(initialdir=initial_search_directory, title="Select Image",
                                                   filetypes=(("jpeg files", ("*.jpg", "*JPG", "*jpeg", "*JPEG")),
                                                              ("all files", "*.*")))

        if filename is not None and type(filename) == str and filename != "":
            self.initial_directory = os.path.split(filename)[0]
            self.img_name_entry.delete(0, tk.END)
            self.img_name_entry.insert(0, filename)
            self.filename = filename

    def _update_value(self, event):
        old_val = self.nr_centers_val
        self.nr_centers_val = self.nr_centers_sc.get()
        self.nr_iterations_val = self.nr_iterations_sc.get()
        if old_val != self.nr_centers_val:
            if old_val > self.nr_centers_val:
                for j in range(old_val - 1, self.nr_centers_val - 1, -1):
                    self.check_buttons[j].grid_remove()
                    self.wanted_clusters[j].set(False)
                    self.percentages_labels[j].grid_remove()
            for i in range(self.nr_centers_val):
                self.check_buttons[i].grid(column=0, row=i + 1)
                self.percentages_labels[i].grid(column=1, row=i+1)

    def repaint_clusters(self):
        total_amount = 0
        values = np.zeros(10)
        if self.canvas is not None:
            self.canvas.destroy()
            self.canvas = None
        if self.segmented_img is not None and self.labels is not None:
            if self.original.get():
                new_segmented_image = np.zeros(self.segmented_img.shape, dtype=self.original_img.dtype)
            else:
                new_segmented_image = np.copy(self.segmented_img)

            for i in range(0, len(self.wanted_clusters)):
                if self.wanted_clusters[i].get() == False:
                    if not self.original.get():
                        new_segmented_image[self.labels == i] = [255, 255, 255]
                    self.percentages[i].set("0%")
                else:
                    if self.original.get():
                        new_segmented_image[self.labels == i] = [1, 1, 1]
                    values[i] = np.count_nonzero(self.labels == i)
                    total_amount += values[i]
            for i in range(0, len(self.percentages)):
                if values[i] != 0:
                    self.percentages[i].set(str("%.2f" % ((values[i] / total_amount) * 100)) + "%")
            if self.original.get():
                new_segmented_image = new_segmented_image.reshape(self.original_img.shape) * self.original_img
            cv2.imwrite(self.resulting_img_path, new_segmented_image.reshape(self.original_img.shape))
            if self.canvas is not None:
                self.canvas.destroy()
                self.canvas = None
            self.canvas = CanvasImage(self.frame_canvas, self.resulting_img_path, self)
            self.canvas.grid(columnspan=2, rowspan=2)

    def calculate(self):
        global initial_search_directory
        if not os.path.exists(K_MEANS_RESULTS_PATH):
            os.mkdir(K_MEANS_RESULTS_PATH)
        if not type(self.filename) == str or not os.path.exists(self.filename):
            messagebox.showinfo("no file chosen!", "choose a file with search")
            return
        if self.canvas is not None:
            self.canvas.destroy()
            self.canvas = None

        if not os.path.exists(self.k_means_centers_path):
            messagebox.showinfo("There seems to be no cluster-centers in " + self.k_means_centers_path +
                                ". The centers will be set to default-values")
            self.pre_centers = None
        else:
            self.pre_centers = np.genfromtxt(self.k_means_centers_path, delimiter=',').astype(int)

        self.resulting_img_path, self.original_img, self.segmented_img, self.labels = apply_k_means(self.filename,
                                                                                                    self.nr_centers_val,
                                                                                                    self.nr_iterations_val,
                                                                                                    self.pre_centers)

        self.repaint_clusters()



def apply_k_means(filename, nr_centers=K_MEANS_NR_CENTERS, nr_iterations=K_MEANS_NR_ITERATIONS, pre_centers=None,
                  img=None):
    if img is None:  # the image needs to be loaded first
        img = cv2.imread(filename, cv2.IMREAD_COLOR)

    segmented, colored, centers = k_means_via_sklearn(img, nr_centers, nr_iterations, pre_centers)
    path_to_segmented = os.path.join(K_MEANS_RESULTS_PATH, path_leaf(filename) + ".bmp")
    cv2.imwrite(path_to_segmented, segmented.reshape(img.shape))
    return path_to_segmented, img, segmented, centers


def k_means_via_sklearn(img, nr_centers, nr_iterations, pre_centers):
    image = img
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.uint8(pixel_values)
    if pre_centers is None:
        messagebox.showinfo("Info", "There seems to be no previously calculated cluster-centers")
        pre_centers = np.array([[54, 63, 44],
                                [95, 112, 85],
                                [141, 154, 115],
                                [199, 202, 173]], np.uint8)
    pre_centers = np.copy(pre_centers)  # make sure the initially read file isnt changed
    if nr_centers < 4:
        pre_centers = np.delete(pre_centers, range(nr_centers - 1, 3), axis=0)
    elif nr_centers > 4:
        pre_centers = np.append(pre_centers, np.zeros(shape=(nr_centers - 4, 3)), axis=0)
    km = KMeans(n_clusters=nr_centers, init=pre_centers, n_init=1, max_iter=nr_iterations).fit(pixel_values)
    centers = km.cluster_centers_
    labels = km.labels_
    segmented_image = np.uint8(centers[labels.flatten()])
    colored = np.zeros(segmented_image.shape, dtype=img.dtype)
    return segmented_image, colored, labels
