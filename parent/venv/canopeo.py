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

from constants import *
from seafile_functions import (FileWalker, FileHandler)

initial_search_directory = "."
kernel_size = KERNEL_SIZE


def path_leaf(path):
    """    head, tail = os.path.split(path)
        return tail or os.path.basename(head)"""
    return re.sub('[^A-Za-z0-9]+', '_', path)


def cleanup_and_create_working_dir():
    if os.path.exists(os.path.join(os.getcwd(), WORKING_DIRECTORY)):
        shutil.rmtree(os.path.join(os.getcwd(), WORKING_DIRECTORY))
    os.mkdir(os.path.join(os.getcwd(), WORKING_DIRECTORY))


def get_file_ending_from_path(path):
    head, tail = os.path.split(path)
    return '.' + re.split("\.", tail)[1] or os.path.basename(head)


def resize_img(image, maxsize):
    r1 = image.size[0] / maxsize[0]  # width ratio
    r2 = image.size[1] / maxsize[1]  # height ratio
    ratio = max(r1, r2)
    newsize = (int(image.size[0] / ratio), int(image.size[1] / ratio))
    image = image.resize(newsize, Image.ANTIALIAS)
    return image


class CanopeoWindow:
    global kernel_size

    def __init__(self, master):
        self.root = master
        self.root.geometry('900x600')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.filename = ""
        self.displayed_filename = tk.StringVar()
        self.original_img = None
        self.p1_val = P1
        self.p2_val = P2
        self.p3_val = P3
        self.resulting_img_path = None
        self.initial_directory = "."
        self.file_walker = None
        self.button_fr = tk.Frame(master)
        self.button_fr.grid_rowconfigure(1, weight=1)
        self.button_fr.grid_rowconfigure(0, weight=1)
        self.button_fr.grid_columnconfigure(0, weight=1)
        self.button_fr.grid_columnconfigure(1, weight=1)
        self.button_fr.grid_columnconfigure(2, weight=1)
        self.refined_mask = None

        self.search_bt = tk.Button(self.button_fr, text="Search Image", command=self.search)
        self.search_bt.grid(row=0, column=0)
        self.quit_bt = tk.Button(self.button_fr, text="QUIT", fg="red", command=self.root.destroy)
        self.quit_bt.grid(row=0, column=2)
        self.img_name_lb = tk.Label(self.button_fr, textvariable=self.displayed_filename, background="white")
        self.img_name_lb.grid(row=1, columnspan=3, pady=2, sticky='EW')
        self.button_fr.pack()

        self.frame_canvas = tk.Frame(self.root, bg='black', relief='sunken', bd=1)
        self.frame_canvas.rowconfigure(1, weight=1)
        self.frame_canvas.columnconfigure(1, weight=1)
        self.canvas = None
        self.frame_canvas.pack(fill=tk.BOTH, expand=1, side='left')
        self.all_scales_fr = tk.Frame(self.root, relief='sunken', bd=1)
        self.all_scales_fr.rowconfigure(0, weight=1)
        self.p_values_fr = tk.Frame(self.all_scales_fr, relief='sunken', bd=1)
        self.p1_sc = tk.Scale(self.p_values_fr, from_=0, to=2, label="p1", resolution=0.01, name="p1_scale")
        self.p1_sc.set(P1)
        self.p1_sc.bind("<ButtonRelease-1>", self._update_value)
        self.p1_sc.grid(row=0, column=0)
        self.p2_sc = tk.Scale(self.p_values_fr, from_=0, to=2.0, label="p2", resolution=0.01)
        self.p2_sc.set(P2)
        self.p2_sc.bind("<ButtonRelease-1>", self._update_value)
        self.p2_sc.grid(row=0, column=1)
        self.p3_sc = tk.Scale(self.p_values_fr, from_=0, to=100, label="p3")
        self.p3_sc.set(P3)
        self.p3_sc.bind("<ButtonRelease-1>", self._update_value)
        self.p3_sc.grid(row=0, column=2)
        self.p_values_fr.grid()
        self.kernel_fr = tk.Frame(self.all_scales_fr, relief='sunken', bd=1)
        self.kernel_size_sc = tk.Scale(self.kernel_fr, from_=0, to=20, label='kernel size')
        self.kernel_size_sc.set(kernel_size)
        self.kernel_size_sc.bind("<ButtonRelease-1>", self._update_value)
        self.kernel_size_sc.pack()
        self.kernel_fr.grid(row=1)

        self.diseased_str = tk.StringVar()
        self.healthy_str = tk.StringVar()
        self.healthy_perc = 0
        self.diseased_perc = 0


        img = Image.open(os.path.join(os.getcwd(), 'icons', 'info.png'))
        img = resize_img(img, [30, 30])
        self.info_icon = ImageTk.PhotoImage(img)
        img = Image.open(os.path.join(os.getcwd(), 'icons', 'refresh.png'))
        img = resize_img(img, [30, 30])
        self.apply_icon = ImageTk.PhotoImage(img)
        self.statistics_frame = tk.LabelFrame(self.all_scales_fr, text='Statistics', bd=1)
        self.statistics_frame.grid(row=2, column=0, sticky='EW')
        self.statistics_frame.grid_rowconfigure(0, weight=1)
        self.statistics_frame.grid_rowconfigure(1, weight=1)
        self.statistics_frame.grid_columnconfigure(0, weight=1)
        self.statistics_frame.grid_columnconfigure(1, weight=1)
        self.diseased_lb = tk.Label(self.statistics_frame, textvariable=self.diseased_str, background='white')
        self.diseased_lb.grid(row=0, column=0, sticky='EW')
        self.healthy_lb = tk.Label(self.statistics_frame, textvariable=self.healthy_str, background='white')
        self.healthy_lb.grid(row=1, column=0, sticky='EW')
        self.s_info_bt = tk.Button(self.statistics_frame, image=self.info_icon, command=self.s_info)
        self.s_info_bt.grid(row=0, column=1, sticky='E')
        self.s_apply_bt = tk.Button(self.statistics_frame, image=self.apply_icon, command=self.s_apply)
        self.s_apply_bt.grid(row=1, column=1, sticky='E')
        img = Image.open(os.path.join(os.getcwd(), 'icons', 'finished.png'))
        img = resize_img(img, [30, 30])
        self.finish_icon = ImageTk.PhotoImage(img)
        img = Image.open(os.path.join(os.getcwd(), 'icons', 'focus.png'))
        img = resize_img(img, [30, 30])
        self.focus_icon = ImageTk.PhotoImage(img)
        self.s_focus_bt = tk.Button(self.statistics_frame, image=self.focus_icon, command=self.s_focus)
        self.s_focus_bt.grid(row=2, column=1, sticky='E')
        self.update_statistics()




        self.save_img_bt = tk.Button(self.all_scales_fr, text="save current picture", command=self.save)
        self.save_img_bt.grid(row=3)
        self.all_scales_fr.pack()

    def cleanup_canvas(self):
        if self.canvas is not None:
            for p in self.canvas.polygon_groundstructure:
                self.canvas.canvas.delete(p)
            for p in self.canvas.polygons:
                self.canvas.canvas.delete(p)


    def search(self):
        global initial_search_directory
        self.search_bt['state'] = ['disabled']
        cleanup_and_create_working_dir()
        online = messagebox.askyesnocancel("Server or Local", "Do you want to choose an online file?")
        if online is None:
            self.search_bt['state'] = ['normal']
            return
        elif online:
            self.file_walker = tk.Toplevel(self.root)
            filename, file_ending = FileWalker(self.file_walker).show()
            if filename == "ERROR":
                if self.canvas is not None:
                    self.canvas.destroy()
                self.filename = ""
                self.displayed_filename.set(self.filename)
                self.img_name_lb.update()
                self.search_bt['state'] = ['normal']
                return
            else:  # download was fine
                self.filename = filename
                self.displayed_filename.set(self.filename)
                self.img_name_lb.update()
                self.filename = os.path.join(os.getcwd(), WORKING_DIRECTORY, DOWNLOADED_IMAGE + file_ending)
                self.load()
                self.search_bt['state'] = ['normal']

        else:  # search for local file
            filename = filedialog.askopenfilename(initialdir=initial_search_directory, title="Select Image",
                                                  filetypes=(("jpeg files", ("*.jpg", "*JPG", "*jpeg", "*JPEG")),
                                                             ("all files", "*.*")))
            if not type(filename) == str or filename == "":
                if self.canvas is not None:
                    self.canvas.destroy()
                self.filename = ""
                self.search_bt['state'] = ['normal']
                return
            self.filename = filename
            self.displayed_filename.set(self.filename)
            self.img_name_lb.update()
            shutil.copy(self.filename, os.path.join(os.getcwd(), WORKING_DIRECTORY, DOWNLOADED_IMAGE +
                                                    get_file_ending_from_path(self.filename)))
            self.filename = os.path.join(os.getcwd(), WORKING_DIRECTORY,
                                         DOWNLOADED_IMAGE + get_file_ending_from_path(self.filename))
            self.load()
            self.search_bt['state'] = ['normal']

    def load(self):
        self.cleanup_canvas()
        self.original_img = cv2.imread(self.filename)
        self.resulting_img_path, self.refined_mask = apply_canopeo(self.original_img, self.p1_val, self.p2_val, self.p3_val)
        
        self.canvas = CanvasImage(self.frame_canvas, self.resulting_img_path, self)
        self.canvas.grid(columnspan=2, rowspan=2)

    def save(self):
        f = filedialog.asksaveasfile(mode='w', defaultextension=".jpg", initialdir=self.initial_directory)
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        self.initial_directory = os.path.split(f.name)[0]
        shutil.copy2(self.resulting_img_path, f.name)

    def update_statistics(self):
        self.healthy_str.set("Healthy: " + str(self.healthy_perc) + "%")
        self.diseased_str.set("Diseased: " + str(self.diseased_perc) + "%")
        self.healthy_lb.update()
        self.diseased_lb.update()

    def _update_value(self, event):
        global kernel_size
        if self.resulting_img_path is not None:
            self.p1_val = self.p1_sc.get()
            self.p2_val = self.p2_sc.get()
            self.p3_val = self.p3_sc.get()
            kernel_size = self.kernel_size_sc.get()
            self.resulting_img_path, self.refined_mask = apply_canopeo(self.original_img, self.p1_val, self.p2_val, self.p3_val)
            if self.canvas is not None:
                self.canvas.destroy()
            self.canvas = CanvasImage(self.frame_canvas, self.resulting_img_path, self)
            self.canvas.grid(columnspan=2, rowspan=2)

    def calculate_percentage(self, polygon_points = None):
        if polygon_points is not None:
            polygon_points = np.int32([polygon_points])
            crop_mask = np.zeros(self.refined_mask.shape)
            crop_mask = cv2.fillPoly(crop_mask, polygon_points, 42)
            calculation_mask = np.copy(self.refined_mask)
            calculation_mask[crop_mask == 42] = 2
            calculation_mask[self.refined_mask == 255] = 1
            calculation_mask[crop_mask != 42] = 0
        else:
            calculation_mask = self.refined_mask / 255
        healthy_num = np.nansum(calculation_mask == 1) or 0
        diseased_num = np.nansum(calculation_mask == 2) or 0
        if diseased_num+ healthy_num == 0:
            self.healthy_perc = 0
            self.diseased_perc = 0
        else:
            self.healthy_perc = (healthy_num/(diseased_num+healthy_num)) * 100
            self.diseased_perc = (diseased_num/(healthy_num+diseased_num)) * 100
        self.update_statistics()

    def s_apply(self):
        if self.original_img is None:
            return
        self.cleanup_canvas()
        self.calculate_percentage()

    def s_focus(self):
        if self.original_img is None:
            return
        global choose_polygon, global_state, creating_polygon
        if self.s_focus_bt['relief'] == 'sunken':
            answer = self.canvas.finish_polygon()
            if type(answer) == str() and answer == 'too_few_points':
                self.cleanup_canvas()
            else:
                self.calculate_percentage(answer)
            self.s_focus_bt['image'] = self.focus_icon
            self.s_focus_bt['relief'] = 'raised'
            self.s_focus_bt.update()
            choose_polygon = False
            creating_polygon = False
        else:
            self.cleanup_canvas()
            self.s_focus_bt['image'] = self.finish_icon
            self.s_focus_bt['relief'] = 'sunken'
            self.s_focus_bt.update()
            global_state = 0
            choose_polygon = True
            creating_polygon = False

    def s_info(self):
        messagebox.showinfo("Statistics-info:", "By clicking on the refresh-button the corresponding statistics are " +
                            "calculated for the WHOLE image. You can choose a specific area by clicking on the " +
                            "focus-button and then when your pointer moves over this area, the values are shown.",
                            parent=self.root)

def apply_canopeo(img, p1=P1, p2=P2, p3=P3):
    refined_binary, finished = acp_canopeo_binary(img, p1, p2, p3)
    path_to_segmented = os.path.join(WORKING_DIRECTORY, "canopeo.png")
    cv2.imwrite(path_to_segmented, finished)
    return path_to_segmented, refined_binary


def refine_mask(img):
    global kernel_size
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def acp_canopeo_binary(img, p1, p2, p3):  ## BGR Channels
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

    refined_binary = refine_mask(binary)
    # cv2.imwrite(os.path.join(IMAGE_FOLDER_PATH, 'segmented_unrefined_' + IMAGE_NAME), cv2.bitwise_and(img, binary))

    colored_binary = cv2.bitwise_and(img, refined_binary)
    return refined_binary, colored_binary


def refine_binary(binary_img):
    global kernel_size
    kernel = np.ones((kernel_size + 2, kernel_size), np.uint8)
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened



"""
def apply_canopeo(filename, p1=P1, p2=P2, p3=P3, img=None):
    if img is None:
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if not os.path.exists(os.path.join(WORKING_DIRECTORY, CANOPEO_RESULTS_PATH)):
        os.makedirs(os.path.join(WORKING_DIRECTORY, CANOPEO_RESULTS_PATH))
    refined, colored = acp_canopeo_binary(img, p1, p2, p3)
    path_to_segmented = os.path.join(WORKING_DIRECTORY, CANOPEO_RESULTS_PATH, path_leaf(filename) + ".png")
    cv2.imwrite(path_to_segmented, colored)
    return path_to_segmented, img"""






global_state = -1
choose_rectangle = False
choose_polygon = False
creating_polygon = False
undo_stack = []
total_number_areas = 0



class CanvasImage:
    """ Display and zoom image """

    def __init__(self, placeholder, path, parent_class):
        """ Initialize the ImageFrame """
        self.placeholder = placeholder
        self.parent_class = parent_class
        self.rectangles = []  # The rectangles on the canvas
        self.start_point_rect = Point2D()
        self.end_point_rect = Point2D()
        self.last_image_id = 0
        self.clicked_rectangle = 0
        self.clicked_r_x = 0
        self.clicked_r_y = 0

        self.clicked_polygon = None
        self.clicked_point_on_polygon = None
        self.polygon_points_history = dict()  # to move the polygons after finishing them

        self.polygons = []
        self.polygon_points = []  # list of the latest polygon-points
        self.polygon_groundstructure = []  # list of points(rectangles and lines) of the polygons

        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.3  # zoom magnitude
        self.__filter = Image.ANTIALIAS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.path = path  # path to the image, should be public for outer classes
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = HiddenScrollbar(self.__imframe, orient='horizontal')
        vbar = HiddenScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized

        self.canvas.bind('<ButtonPress-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B1-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<ButtonRelease-1>', self.__mouse_release)  # call the wrapper function
        self.canvas.bind('<ButtonPress-3>', lambda event: self.__move_from(event, True))  # remember canvas position
        self.canvas.bind('<B3-Motion>', lambda event: self.__move_to(event, True))  # move canvas to the new position
        self.canvas.bind('<ButtonRelease-3>',
                         lambda event: self.__mouse_release(event, True))  # call the wrapper function
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        # self.canvas.bind('<Shift_L>', self.__shift) #detect if shift key was pressed
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)  # open image, but down't load it
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
                self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)
        while i < self.imheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k) + 1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30 * ' ' + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                    (int(x1 / self.__scale), int(y1 / self.__scale),
                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            if self.last_image_id != 0:
                self.canvas.delete(self.last_image_id)

            self.last_image_id = imageid
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event, right_click=False):
        global choose_rectangle, choose_polygon
        if (not right_click) and choose_polygon:
            self.__draw_polygon(event, klick=True)
        elif (not right_click) and choose_rectangle:
            self.__begin_rectangle(event)
        else:
            """ Remember previous coordinates for scrolling with the mouse """
            self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event, right_click=False):
        global choose_rectangle, choose_polygon
        if (not right_click) and choose_rectangle:
            self.__expand_rectangle(event)
        elif (not right_click) and choose_polygon:
            self.__draw_polygon(event, klick=False)
        else:
            """ Drag (move) canvas to the new position """
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            self.__show_image()  # zoom tile and show it on the canvas

    def __mouse_release(self, event, right_click=False):
        """wrapper function for mouse-button release"""
        global choose_rectangle
        if right_click:
            return
        if choose_rectangle:
            self.__finish_rectangle(event)

    def get_image_relative_coords(self, coords: tuple):
        # get the position of the img box on the canvas
        box_img_int = tuple(map(float, self.canvas.coords(self.container)))[0:2]
        # set vector coords according to the img_box
        point = np.asarray(coords, dtype=float) - np.asarray(box_img_int)
        point = np.true_divide(point, self.imscale).astype(int)
        return point

    def get_canvas_relative_coords(self, coords: tuple):
        box_img_int = tuple(map(float, self.canvas.coords(self.container)))[0:2]
        point = (np.asarray(coords) * self.imscale + np.asarray(box_img_int)).astype(int)
        return point

    def reset_stacks(self):
        global undo_stack
        for p in self.polygons:
            self.canvas.delete(p)
        self.polygons.clear()
        for r in self.rectangles:
            self.canvas.delete(r)
        self.rectangles.clear()
        for pg in self.polygon_groundstructure:
            self.canvas.delete(pg)
        self.polygon_groundstructure.clear()
        self.polygon_points.clear()
        self.polygon_points_history.clear()
        undo_stack.clear()

    def which_rectangle_clicked(self, x, y):
        self.clicked_rectangle = 0
        for rect in self.rectangles:
            r_x1, r_y1, r_x2, r_y2 = self.canvas.coords(rect)
            if r_x1 <= x <= r_x2:
                if r_y1 <= y <= r_y2:
                    self.clicked_rectangle = rect
                    return

    def callback_move_rectangle(self, event):
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)
        if pow(pow(curX - self.clicked_r_x, 2) + pow(curY - self.clicked_r_y, 2), 1 / 2) <= 15:
            return
        if self.clicked_rectangle == 0:
            return
        r_x1, r_y1, r_x2, r_y2 = self.canvas.coords(self.clicked_rectangle)

        x1 = abs(r_x1 - curX)
        x2 = abs(r_x2 - curX)
        y1 = abs(r_y1 - curY)
        y2 = abs(r_y2 - curY)
        if x1 < x2:
            x1 = curX
            x2 = r_x2
        else:
            x1 = r_x1
            x2 = curX
        if y1 < y2:
            y1 = curY
            y2 = r_y2
        else:
            y1 = r_y1
            y2 = curY
        self.canvas.coords(self.clicked_rectangle, x1, y1, x2, y2)

    def callback_click_rectangle(self, event):
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)
        self.clicked_r_x = curX
        self.clicked_r_y = curY
        self.which_rectangle_clicked(curX, curY)
        self.canvas.unbind('<B1-Motion>')
        self.canvas.unbind('<ButtonPress-1>')
        self.canvas.unbind('<ButtonRelease1>')

    def callback_release_rectangle(self, event):
        self.canvas.bind('<ButtonPress-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B1-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<ButtonRelease-1>', self.__mouse_release)  # call the wrapper function

    def __begin_rectangle(self, event):
        """begin drawing a rectangle when mouse-button is pressed"""
        self.start_point_rect = Point2D(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        global choose_rectangle
        if choose_rectangle:
            self.rectangles.append(self.canvas.create_rectangle(self.start_point_rect.x,
                                                                    self.start_point_rect.y, self.end_point_rect.x,
                                                                    self.end_point_rect.y,
                                                                    outline='blue', width=1,
                                                                    activewidth=2, fill='magenta', stipple='gray50'))

    def __expand_rectangle(self, event):
        """expand the begun rectangle"""
        global choose_rectangle
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if event.x > 0.9 * w:
            self.canvas.xview_scroll(1, 'units')
        elif event.x < 0.1 * w:
            self.canvas.xview_scroll(-1, 'units')
        if event.y > 0.9 * h:
            self.canvas.yview_scroll(1, 'units')
        elif event.y < 0.1 * h:
            self.canvas.yview_scroll(-1, 'units')

        # expand rectangle as you drag the mouse
        if choose_rectangle:
            self.canvas.coords(self.rectangles[-1], self.start_point_rect.x, self.start_point_rect.y, curX,
                               curY)
        self.end_point_rect = Point2D(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def __finish_rectangle(self, event):
        self.parent_class.activate_save_bt()
        """finish rectangles"""
        global choose_rectangle, undo_stack
        if choose_rectangle:
            undo_stack.append('r')
            self.parent_class.rectangle()
            self.canvas.tag_bind(self.rectangles[-1], '<ButtonPress-1>', self.callback_click_rectangle)
            self.canvas.tag_bind(self.rectangles[-1], '<B1-Motion>', self.callback_move_rectangle)
            self.canvas.tag_bind(self.rectangles[-1], '<ButtonRelease-1>', self.callback_release_rectangle)

    def callback_click_polygon(self, event):
        self.canvas.unbind('<B1-Motion>')
        self.canvas.unbind('<ButtonPress-1>')
        self.canvas.unbind('<ButtonRelease1>')
        poly = event.widget.find_withtag('current')[0]
        if poly == 1:
            return self.callback_release_polygon(event)
        if not poly in self.polygon_points_history:
            print("{} not in {}".format(poly, self.polygon_points_history))
            return self.callback_release_polygon(event)
        points = self.polygon_points_history[poly]
        curX = event.widget.canvasx(event.x)
        curY = event.widget.canvasy(event.y)
        self.clicked_point_on_polygon = closest_node(self.get_image_relative_coords((curX, curY)), points)
        self.clicked_polygon = poly

    def callback_release_polygon(self, event):
        self.clicked_polygon = None
        self.clicked_point_on_polygon = None
        self.canvas.bind('<ButtonPress-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B1-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<ButtonRelease-1>', self.__mouse_release)  # call the wrapper function

    def callback_move_polygon(self, event):
        if self.clicked_point_on_polygon is None or self.clicked_polygon is None:
            return self.callback_release_polygon(event)
        curX = event.widget.canvasx(event.x)
        curY = event.widget.canvasy(event.y)
        true_x, true_y = self.get_image_relative_coords((curX, curY))
        points = self.polygon_points_history[self.clicked_polygon]
        points[self.clicked_point_on_polygon][0] = true_x
        points[self.clicked_point_on_polygon][1] = true_y
        coords = []
        for p in range(points.shape[0]):
            x, y = self.get_canvas_relative_coords(points[p])
            coords.extend((x, y))
        self.canvas.coords(self.clicked_polygon, coords)

    def __draw_polygon(self, event, klick):
        """add some points to a polygon -- stored as line until finish polygon is called"""
        global creating_polygon
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)
        if not klick and len(self.polygon_points) >= 2:
            c_r_x, c_r_y = self.get_canvas_relative_coords((self.polygon_points[-2], self.polygon_points[-1]))
            distanceX = curX - c_r_x
            distanceY = curY - c_r_y
            if pow(pow(distanceX, 2) + pow(distanceY, 2), 1 / 2) <= 15:
                return
        image_relative_x, image_relative_y = self.get_image_relative_coords((curX, curY))
        self.polygon_points.extend((image_relative_x, image_relative_y))
        self.polygon_groundstructure.append(self.canvas.create_rectangle(curX - 2, curY - 2, curX + 2, curY + 2,
                                                                         outline='magenta', width=1,
                                                                         activewidth=2))
        if not creating_polygon:  # start with a new polygon
            creating_polygon = True
            return
        else:  # draw a line between the last points
            c_r_x1, c_r_y1 = self.get_canvas_relative_coords((self.polygon_points[-4], self.polygon_points[-3]))
            c_r_x2, c_r_y2 = self.get_canvas_relative_coords((self.polygon_points[-2], self.polygon_points[-1]))
            self.polygon_groundstructure.append(self.canvas.create_line([c_r_x1, c_r_y1, c_r_x2, c_r_y2],
                                                                        fill='magenta', width=2))

    def finish_polygon(self):
        """remove the basic-structure and redraw a actual polygon"""
        global undo_stack, choose_polygon
        if len(self.polygon_points) < 6:
            messagebox.showinfo(title='Info', message='Too few points for a polygon', parent=self.root)
            return 'too_few_points'
        relative_poly_points = []
        for p in range(0, len(self.polygon_points), 2):
            relative_poly_points.extend(self.get_canvas_relative_coords((self.polygon_points[p],
                                                                         self.polygon_points[p + 1])))
        if choose_polygon:
            undo_stack.append('p')
            self.polygons.append(self.canvas.create_polygon(relative_poly_points,
                                                                outline='blue', activewidth=3, width=1,
                                                                fill='magenta', stipple='gray50'))
            self.canvas.tag_bind(self.polygons[-1], '<ButtonPress-1>', self.callback_click_polygon)
            self.canvas.tag_bind(self.polygons[-1], '<ButtonRelease-1>', self.callback_release_polygon)
            self.canvas.tag_bind(self.polygons[-1], '<B1-Motion>', self.callback_move_polygon)
            for p in self.polygon_groundstructure:
                self.canvas.delete(p)
            self.polygon_points_history[self.polygons[-1]] = np.reshape(np.asarray(self.polygon_points),
                                                                            (round(len(self.polygon_points) / 2),
                                                                             2))
            self.polygon_points.clear()
            self.polygon_groundstructure.clear()

        return self.polygon_points_history[self.polygons[-1]]

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down, smaller
            if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.__delta
            scale /= self.__delta
        if event.num == 4 or event.delta == 120:  # scroll up, bigger
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.__delta
            scale *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            if event.char in [' ', 'f']:
                return self.parent_class.finish_polygons_key()
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right: keys 'D', 'Right' or 'Numpad-6'
                self.__scroll_x('scroll', 1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left: keys 'A', 'Left' or 'Numpad-4'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up: keys 'W', 'Up' or 'Numpad-8'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down: keys 'S', 'Down' or 'Numpad-2'
                self.__scroll_y('scroll', 1, 'unit', event=event)

    def crop(self, bbox):
        """ Crop rectangle from the image and return it """
        if self.__huge:  # image is huge and not totally in RAM
            band = bbox[3] - bbox[1]  # width of the tile band
            self.__tile[1][3] = band  # set the tile height
            self.__tile[2] = self.__offset + self.imwidth * bbox[1] * 3  # set offset of the band
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]
            return self.__image.crop((bbox[0], 0, bbox[2], band))
        else:  # image is totally in RAM
            return self.__pyramid[0].crop(bbox)

    def destroy(self):
        """ ImageFrame destructor """
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)  # close all pyramid images
        del self.__pyramid[:]  # delete pyramid list
        del self.__pyramid  # delete pyramid variable
        self.canvas.destroy()
        self.__imframe.destroy()
        return

class HiddenScrollbar(ttk.Scrollbar):
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)