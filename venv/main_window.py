from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import warnings
import os
from message_box import *
import shutil
import sys as sys

from annotation_tool import MeanWindow
from canopeo import CanopeoWindow
from k_means_learn_window import LearnCentersWindow
from model_window import ApplyModel
#from ccm_calculation_window import CCMCalculations
from constants import *


class MainWindow:
    def __init__(self, master):
        self.root = master
        self.root.geometry("600x600")
        self.root.title("FungoMatics")
        self.root.maxsize = (600, 600)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.annotation_window = None
        self.canopeo_window = None
        self.k_means_window = None
        self.model_window = None
        self.app = None
        cleanup()
        self.title_fr = tk.Frame(self.root)
        self.title_label = tk.Label(self.title_fr, text="Welcome to FungoMatics, choose an application")
        self.title_label.pack()
        self.title_fr.pack()

        self.button_fr = tk.Frame(master)
        self.annotation_bt = tk.Button(self.button_fr, text="Annotation Tool", command=self.start_annotation)
        self.annotation_bt.pack()
        self.canopeo_bt = tk.Button(self.button_fr, text="Canopeo-based segmentation", command=self.start_canopeo)
        self.canopeo_bt.pack()
        self.k_means_bt = tk.Button(self.button_fr, text="K-MEANS", command=self.start_k_means)
        self.k_means_bt.pack()
        self.apply_model_bt = tk.Button(self.button_fr, text="Apply U-NET model",
                                          command=self.apply_model)
        self.apply_model_bt.pack()
        self.button_fr.pack()

    def start_annotation(self):
        self.annotation_window = tk.Toplevel(self.root)
        self.app = MeanWindow(self.annotation_window)

    def start_canopeo(self):
        self.canopeo_window = tk.Toplevel(self.root)
        self.app = CanopeoWindow(self.canopeo_window)

    def start_k_means(self):
        self.k_means_window = tk.Toplevel(self.root)
        self.app = LearnCentersWindow(self.k_means_window)

    def apply_model(self):
        self.model_window = tk.Toplevel(self.root)
        self.app = ApplyModel(self.model_window)


def cleanup():
    if os.path.exists(CANOPEO_RESULTS_PATH):
        shutil.rmtree(CANOPEO_RESULTS_PATH)
    if os.path.exists(WORKING_DIRECTORY):
        shutil.rmtree(WORKING_DIRECTORY)
