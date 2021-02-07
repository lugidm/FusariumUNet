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
# from rearranging_frame import RearrangingFrame
from seafile_functions import FileWalker
from seafile_functions import FileHandler

# states of the application:
global_state = -1
choose_rectangle = False
choose_polygon = False
creating_polygon = False
undo_stack = []
total_number_areas = 0


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    node = np.asarray(node)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def path_leaf(path):
    """    head, tail = os.path.split(path)
        return tail or os.path.basename(head)"""
    return re.sub('[^A-Za-z0-9]+', '_', path)


def get_file_ending_from_path(path):
    head, tail = os.path.split(path)
    return '.' + re.split("\.", tail)[1] or os.path.basename(head)


def cleanup_and_create_working_dir():
    if os.path.exists(os.path.join(os.getcwd(), WORKING_DIRECTORY)):
        shutil.rmtree(os.path.join(os.getcwd(), WORKING_DIRECTORY))
    os.mkdir(os.path.join(os.getcwd(), WORKING_DIRECTORY))


class HiddenScrollbar(ttk.Scrollbar):
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)


def resize_img(image, maxsize):
    r1 = image.size[0] / maxsize[0]  # width ratio
    r2 = image.size[1] / maxsize[1]  # height ratio
    ratio = max(r1, r2)
    newsize = (int(image.size[0] / ratio), int(image.size[1] / ratio))
    image = image.resize(newsize, Image.ANTIALIAS)
    return image


class MeanWindow:
    def __init__(self, master, label_list=[PARZELLE_LABEL, EINZELREIHE_LABEL, EAR_LABEL],
                 label_names=["parzelle", "einzelreihe", "ear"],
                 label_image_names=[EAR_LABEL_IMAGE, EINZELREIHE_LABEL_IMAGE, PARZELLE_LABEL_IMAGE],
                 hierarchical=True, save_online=True, offline_directory=ONLINE_RESULTS_DIRECTORY):
        global global_state
        global_state = -1
        self.file_walker = None  # variable for tkinter window
        self.filename = ""
        self.original_image = ""  # variable gets set once and never altered
        self.root = master
        self.label_image_names = label_image_names
        self.label_names = label_names
        self.label_list = label_list
        self.save_online = save_online
        self.hierarchical = hierarchical
        self.potential_upload_files = list()
        self.default_potential_upload_files()
        self.offline_directory = offline_directory
        self.flattened_picture_name = ""
        self.root.geometry("900x600")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.displayed_filename = tk.StringVar()
        self.button_frame = tk.Frame(master)
        self.search_bt = tk.Button(self.button_frame, text="Search Image", command=self.search)
        self.search_bt.grid(row=0, column=1, padx=2, pady=2)
        self.quit = tk.Button(self.button_frame, text="QUIT", fg="red", command=self.root.destroy)
        self.quit.grid(row=0, column=2, padx=2, pady=2)
        self.button_frame.pack(side='top')
        self.img_name_lb = tk.Label(self.button_frame, background='white', textvariable=self.displayed_filename)
        self.img_name_lb.grid(row=1, column=0, columnspan=3, sticky='EW', pady=2, padx=2)

        self.current_label_name = tk.StringVar()
        self.current_label_name.set("annotate " + self.label_names[0] + "-area")
        self.frame_canvas = tk.Frame(master, bg='black', relief='sunken', bd=1)
        self.frame_canvas.rowconfigure(0, weight=1)
        self.frame_canvas.columnconfigure(0, weight=1)

        self.canvas = None
        self.frame_canvas.pack(fill=tk.BOTH, expand=1, side='left')

        """frame for area choosing"""
        self.choose_frame = tk.Frame(master)
        self.choose_frame.grid_rowconfigure(1, weight=1)
        self.choose_frame.grid_rowconfigure(2, weight=1)

        # self.choose_scrollable_fr = RearrangingFrame(self, self.choose_frame)

        self.paint_root_fr = tk.Frame(self.choose_frame)

        self.back_save_cancel_fr = tk.Frame(self.choose_frame)
        self.back_save_cancel_fr.grid_rowconfigure(0, weight=1)
        self.back_save_cancel_fr.grid_columnconfigure(0, weight=1)
        self.back_save_cancel_fr.grid_columnconfigure(1, weight=1)
        self.back_save_cancel_fr.grid_columnconfigure(2, weight=1)

        img = Image.open(os.path.join(os.getcwd(), 'icons', 'info.png'))
        img = resize_img(img, [30, 30])
        self.info_icon = ImageTk.PhotoImage(img)
        self.info_bt = tk.Button(self.back_save_cancel_fr, image=self.info_icon, command=self.info, name="info")
        self.info_bt.grid(row=0, column=0, sticky='W')

        img = Image.open(os.path.join(os.getcwd(), 'icons', 'back.png'))
        img = resize_img(img, [30, 30])
        self.back_icon = ImageTk.PhotoImage(img)
        self.back_bt = tk.Button(self.back_save_cancel_fr, image=self.back_icon, command=self.back, name="back")
        self.back_bt.grid(row=0, column=1, sticky='W')

        img = Image.open(os.path.join(os.getcwd(), 'icons', 'cancel.png'))
        img = resize_img(img, [30, 30])
        self.home_icon = ImageTk.PhotoImage(img)
        self.home_bt = tk.Button(self.back_save_cancel_fr, image=self.home_icon, command=self.home, name="home")
        self.home_bt.grid(row=0, column=2, sticky='W')
        img = Image.open(os.path.join(os.getcwd(), 'icons', 'finished.png'))
        img = resize_img(img, [30, 30])
        self.finished_icon = ImageTk.PhotoImage(img)
        self.finished_bt = tk.Button(self.back_save_cancel_fr, image=self.finished_icon, state='disabled',
                                     command=self.finished, name='finished')
        self.finished_bt.grid(row=0, column=3, sticky='E')
        # self.choose_scrollable_fr.grid(sticky='NS')
        self.choose_frame.pack(fill='y', expand=True)

        """frame to choose between parzelle+einzelreihe or ear"""
        img = Image.open(os.path.join(os.getcwd(), 'icons', 'start.png'))
        img = resize_img(img, [30, 30])
        self.start_icon = ImageTk.PhotoImage(img)
        self.start_bt = tk.Button(self.choose_frame, image=self.start_icon, command=self.start_annotation,
                                  state='disabled')
        self.start_bt.grid(row=0, column=0, sticky='NS')

        """painting_area"""
        self.paint_fr = tk.Frame(self.paint_root_fr, bg="magenta")
        self.paint_fr.columnconfigure(0, weight=1)
        self.paint_fr.rowconfigure(0, weight=1)
        self.paint_fr.rowconfigure(1, weight=1)
        self.paint_fr.rowconfigure(2, weight=1)
        self.paint_fr.rowconfigure(3, weight=1)
        self.paint_fr.grid(sticky='NEW')
        self.paint_lb = tk.Label(self.paint_fr, textvariable=self.current_label_name, bg=self.paint_fr['background'])
        self.paint_lb.grid(sticky='EW', pady=10)

        self.paint_rectangle_fr = tk.Frame(self.paint_fr, bg=self.paint_fr['background'],
                                           relief='sunken', bd=1, pady=2)
        self.paint_rectangle_fr.grid_columnconfigure(0, weight=1)
        self.paint_rectangle_fr.grid_rowconfigure(0, weight=1)
        self.paint_rectangle_fr.grid(row=1, sticky='EW')
        self.paint_rectangle_bt = tk.Button(self.paint_rectangle_fr, text='add a new rectangle', state="normal")
        self.paint_rectangle_bt.configure(command=self.rectangle)
        self.paint_rectangle_bt.grid(sticky='EW')

        self.paint_polygon_fr = tk.Frame(self.paint_fr, bg=self.paint_fr['background'],
                                         relief='sunken', bd=1, pady=2)
        self.paint_polygon_fr.grid_columnconfigure(0, weight=1)
        self.paint_polygon_fr.grid_columnconfigure(1, weight=2)
        self.paint_polygon_fr.grid_rowconfigure(0, weight=1)
        self.paint_polygon_fr.grid_rowconfigure(1, weight=1)
        self.paint_polygon_fr.grid(row=2, sticky='SNEW')
        self.new_polygon_bt = tk.Button(self.paint_polygon_fr, text="add a new polygon",
                                        name='einzelreihe_polygon')
        self.new_polygon_bt.configure(command=self.polygon)
        self.new_polygon_bt.grid(columnspan=2, sticky='EW', padx=2)

        self.undo_polygon_bt = tk.Button(self.paint_polygon_fr, text="Undo", fg='blue', state='disabled',
                                         command=self.undo_polygon)
        self.undo_polygon_bt.grid(row=1)

        self.finish_polygon_bt = tk.Button(self.paint_polygon_fr, text="finish this polygon", fg='green',
                                           state='disabled', command=self.finish_polygon)
        self.finish_polygon_bt.grid(row=1, column=1)

        self.remove_latest_area_bt = tk.Button(self.paint_fr, text="remove latest anotated area", fg='red',
                                               command=self.remove_latest)
        self.remove_latest_area_bt.grid(row=3, pady=3)

        self.disable_polygon_and_rectangle_buttons()

    def info(self):
        messagebox.showinfo("Usage", "While creating a polygon you can drag the mouse with the left mouse-button "
                            + "pressed to create a line. To move the image around whilst doing so, use the " +
                            "right mouse-button.\n" + "You can change the shape of the polygon after finishing it" +
                            " by clicking on the polygon with the left mouse button. The same holds for rectangles\n" +
                            "You can start and finish polygon by hitting the space-button")

    def activate_save_bt(self):
        self.finished_bt['state'] = 'normal'

    def disable_save_bt(self):
        self.finished_bt['state'] = 'disabled'

    def activate_polygon_and_rectangle_buttons(self):
        self.start_bt['state'] = 'normal'
        self.new_polygon_bt['state'] = 'normal'
        self.paint_rectangle_bt['state'] = 'normal'

    def disable_polygon_and_rectangle_buttons(self):
        self.start_bt['state'] = 'disabled'
        self.new_polygon_bt['state'] = 'disabled'
        self.paint_rectangle_bt['state'] = 'disabled'

    def default_potential_upload_files(self):
        self.potential_upload_files.clear()
        for l_i in self.label_image_names:
            self.potential_upload_files.append(os.path.join(os.getcwd(), WORKING_DIRECTORY, l_i))

    def search(self):
        global global_state, total_number_areas
        self.default_potential_upload_files()
        self.search_bt['state'] = ['disabled']
        online = messagebox.askyesnocancel("Server or Local", "Do you want to choose an online file?")
        if online is None:
            self.search_bt['state'] = ['normal']
            return
        elif online:
            cleanup_and_create_working_dir()
            self.__silent_abort()
            self.file_walker = tk.Toplevel(self.root)
            filename, file_ending = FileWalker(self.file_walker).show()
            if filename == "ERROR":
                if self.canvas is None:
                    self.filename = ""
                    self.displayed_filename.set(self.filename)
                    self.img_name_lb.update()
                self.search_bt['state'] = ['normal']
                return
            else:  # download was fine
                self.filename = filename
                self.displayed_filename.set(self.filename)
                self.img_name_lb.update()
                self.flattened_picture_name = path_leaf(self.filename)
                self.filename = os.path.join(os.getcwd(), WORKING_DIRECTORY, DOWNLOADED_IMAGE + file_ending)
                self.original_image = self.filename
                self.potential_upload_files.append(self.filename)
                self.load()
                self.search_bt['state'] = ['normal']
                global_state = -1
                total_number_areas = 0
        else:  # search for local file
            filename = filedialog.askopenfilename(initialdir=".", title="Select Image",
                                                  filetypes=(("jpeg files", ("*.jpg", "*JPG", "*jpeg", "*JPEG")),
                                                             ("all files", "*.*")))
            if not type(filename) == str or filename == "":
                if self.canvas is None:
                    self.filename = ""
                self.search_bt['state'] = ['normal']
                return
            cleanup_and_create_working_dir()
            self.filename = filename
            self.displayed_filename.set(self.filename)
            self.img_name_lb.update()
            self.flattened_picture_name = path_leaf(self.filename)
            shutil.copy(self.filename, os.path.join(os.getcwd(), WORKING_DIRECTORY, DOWNLOADED_IMAGE +
                                                    get_file_ending_from_path(self.filename)))
            self.filename = os.path.join(os.getcwd(), WORKING_DIRECTORY,
                                         DOWNLOADED_IMAGE + get_file_ending_from_path(self.filename))
            self.original_image = self.filename
            self.potential_upload_files.append(self.filename)
            self.load()
            global_state = -1
            total_number_areas = 0
            self.search_bt['state'] = ['normal']

    """load the image from self.filename"""

    def load(self):
        if self.canvas is not None:
            self.__silent_abort()
            self.canvas.destroy()
        if os.path.exists(self.filename):
            self.canvas = CanvasImage(self.frame_canvas, self.filename, self)
            self.canvas.grid(columnspan=2, rowspan=2)
            self.activate_polygon_and_rectangle_buttons()
        else:
            messagebox.showerror("File does not exist",
                                 "the given file does not exist (or could not be downloaded correctly)")

    """grid the buttons for the first annotation"""

    def start_annotation(self):
        global global_state
        global_state = 0
        if self.canvas is not None:
            self.canvas.reset_stacks()
        self.update_label()
        self.start_bt.grid_remove()
        self.back_save_cancel_fr.grid(sticky='NSWE', row=0, column=0)
        self.paint_root_fr.grid(sticky='NS', row=1, column=0)

    """if parzelle was annotated continue"""

    def continue_annotation(self):
        global global_state
        if global_state == -1:
            print("error!!!!")
            exit("SOME ERROR WITH THE STATES")
        else:
            global_state += 1
            self.update_label()
            if self.canvas is not None:
                self.canvas.reset_stacks()

    def update_label(self):
        global global_state
        self.current_label_name.set("annotate " + self.label_names[global_state] + "-area")
        self.paint_lb.update()

    def home(self):
        global global_state, total_number_areas
        self.search_bt['state'] = 'normal'
        if self.canvas is not None and total_number_areas > 0:
            answer = messagebox.askyesnocancel("Some Changes have been made", "Save changes?")
            if answer is True:
                for i in range(global_state, len(self.label_list)):
                    self.finished()
                self.paint_root_fr.grid_remove()
                self.__silent_abort()
                global_state = -1
                total_number_areas = 0
                self.start_bt.grid(sticky='NS')
                self.disable_save_bt()
            elif answer is False:
                self.paint_root_fr.grid_remove()
                global_state = -1
                total_number_areas = 0
                self.__silent_abort()
                self.back_save_cancel_fr.grid_remove()
                self.disable_save_bt()
                self.start_bt.grid(sticky='NS')
                self.filename = self.original_image
                self.load()
                return
            else:
                return
        else:  # nothing has been done. the image can still be annotated
            global_state = -1
            self.filename = self.original_image
            self.load()
            self.paint_root_fr.grid_remove()
            self.back_save_cancel_fr.grid_remove()
            self.start_bt.grid(sticky='NS')
            self.disable_save_bt()
            return

    def finish_polygons_key(self):
        global creating_polygon
        if creating_polygon:
            self.finish_polygon()
        else:
            self.polygon()

    """Function for the check-button (continue) -- uploads the imgs"""

    def finished(self): #TODO UPLOAD
        global global_state
        if global_state == -1:
            exit("ERROR WITH FINISHED")
        elif global_state < (len(self.label_list) - 1):
            self._save()
            self.continue_annotation()
            self.load()
        else:  # finished with the final label
            if not self.save_online:
                self.disable_save_bt()
                self._save()
                self.search_bt['state'] = 'disabled'
                self.start_bt['state'] = 'disabled'
                if not os.path.exists(self.offline_directory):
                    os.mkdir(self.offline_directory)
                if not os.path.exists(os.path.join(self.offline_directory, self.flattened_picture_name)):
                    os.mkdir(os.path.join(self.offline_directory, self.flattened_picture_name))
                for file in self.potential_upload_files:
                    shutil.copy2(file, os.path.join(self.offline_directory, self.flattened_picture_name, ""))
                self.__silent_abort()
                return self.home()

            answer = messagebox.askyesnocancel("Online or Local?", "Do you want to save the results online?")
            if answer is None:
                self.activate_save_bt()
                self.start_bt['state'] = 'normal'
                self.search_bt['state'] = 'normal'
                return
            elif answer is False:
                self.disable_save_bt()
                self._save()
                self.search_bt['state'] = 'disabled'
                self.start_bt['state'] = 'disabled'

                f = filedialog.askdirectory(mustexist=False, initialdir=".")
                if f == '':  # asksaveasfile return if dialog closed with "cancel".
                    return
                if not os.path.exists(os.path.join(f, self.flattened_picture_name, "")):
                    os.mkdir(os.path.join(f, self.flattened_picture_name, ""))
                for file in self.potential_upload_files:
                    shutil.copy2(file, os.path.join(f, self.flattened_picture_name, ""))

                self.activate_save_bt()
                self.start_bt['state'] = 'normal'
                self.search_bt['state'] = 'normal'
                return

            self.disable_save_bt()
            self._save()
            self.search_bt['state'] = ['disabled']
            self.start_bt['state'] = ['disabled']
            self.file_walker = tk.Toplevel(self.root)


            fh_answer = FileHandler(self.file_walker).save(self.flattened_picture_name,
                                                           self.potential_upload_files)
            if fh_answer == "overwrite":
                messagebox.showinfo("Success", "The previously annotated areas have been overwritten")
                self.home()
            elif fh_answer == "success":
                messagebox.showinfo("Success", "The annotated areas have been uploaded")
                self.home()
            elif fh_answer == "fuse":
                new_answer = self.__fuse()
                if new_answer == "success":
                    messagebox.showinfo("Success", "The old and new annotated areas have been fused and uploaded to " +
                                        self.flattened_picture_name)
                    self.home()
                else:
                    messagebox.showerror("ERROR", "there was an error while fusing both images")
            elif re.split("<", fh_answer)[0] == "new_file":
                messagebox.showinfo("Success", "The annotations have been written to a new file: " +
                                    re.split('<', fh_answer)[-1])
                self.home()
            elif fh_answer == "discard":
                new_answer = messagebox.askyesno("Attention",
                                                 "do you want to delete all the annotated areas from this session?")
                if new_answer:
                    self.__silent_abort()
                    self.home()
                    cleanup_and_create_working_dir()
                    self.search_bt['state'] = ['normal']
                    return
                else:
                    self.search_bt['state'] = ['normal']
                    return

            elif fh_answer == "cancel":
                self.search_bt['state'] = ['normal']
                self.start_bt['state'] = ['normal']
                return
            else:
                messagebox.showerror("ERROR", "ERROR")
                self.__silent_abort()
                self.home()

    def back(self):
        global global_state
        if global_state == -1:
            exit("BACK_ERROR")
        if global_state == 0:
            self.home()
        elif global_state == 1:
            if len(self.canvas.polygons) != 0 or len(self.canvas.rectangles) != 0:
                answer = messagebox.askyesnocancel("Warning",
                                                   "By returning to previous annotation ({}) ".format(
                                                       self.label_names[global_state-1]) +
                                                   "the current annotations on einzelreihe ares will " +
                                                   "be deleted! Continue?")
                if answer:
                    pass
                else:
                    return
            global_state = -1
            self.paint_root_fr.grid_remove()
            self.filename = self.original_image
            self.load()
            self.start_annotation()
        elif self.state > 1:
            if len(self.canvas.polygons) != 0 or len(self.canvas.rectangles) != 0:
                answer = messagebox.askyesnocancel("Warning",
                                                   "By returning to previous annotation ({}) ".format(
                                                       self.label_names[global_state-1]) +
                                                   "the current annotations on einzelreihe ares will " +
                                                   "be deleted! Continue?")
                if answer:
                    pass
                else:
                    return
            global_state -= 1
            self.update_label()
            if self.hierarchical:
                self.filename = os.path.join(os.getcwd(), WORKING_DIRECTORY, 'intermediate' +
                                             self.label_image_names[global_state-1])
                if os.path.exists(os.path.join(os.getcwd(), WORKING_DIRECTORY,
                                               'intermediate' + self.label_image_names[global_state])):
                    os.remove(os.path.join(os.getcwd(), WORKING_DIRECTORY,
                                           'intermediate' + self.label_image_names[global_state]))
                if os.path.exists(os.path.join(os.getcwd(), WORKING_DIRECTORY, self.label_image_names[global_state])):
                    os.remove(os.path.join(os.getcwd(), WORKING_DIRECTORY, self.label_image_names[global_state]))
                    self.potential_upload_files.remove(
                        os.path.join(os.getcwd(), WORKING_DIRECTORY, self.label_image_names[global_state]))
            self.load()

    """find out which label to create a rectangle with, then disable all other buttons"""

    def rectangle(self):
        global choose_rectangle, total_number_areas
        if self.paint_rectangle_bt.cget('relief') == 'sunken':
            self.back_bt['state'] = 'normal'
            self.home_bt['state'] = 'normal'
            self.finished_bt['state'] = 'normal'
            self.new_polygon_bt['state'] = 'normal'
            self.paint_rectangle_bt['relief'] = 'raised'
            self.paint_rectangle_bt['text'] = 'add a new rectangle'
            choose_rectangle = False
            total_number_areas += 1
        else:
            self.paint_rectangle_bt['relief'] = 'sunken'
            self.paint_rectangle_bt['text'] = 'Cancel'
            self.back_bt['state'] = 'disabled'
            self.home_bt['state'] = 'disabled'
            self.finished_bt['state'] = 'disabled'
            self.new_polygon_bt['state'] = 'disabled'
            choose_rectangle = True
            total_number_areas -= 1

    """find out which label to create a polygon with, then disable all other buttons"""

    def polygon(self):
        global choose_polygon, creating_polygon
        self.back_bt['state'] = 'disabled'
        self.home_bt['state'] = 'disabled'
        self.finished_bt['state'] = 'disabled'
        self.paint_rectangle_bt['state'] = 'disabled'
        self.undo_polygon_bt['state'] = 'normal'
        self.finish_polygon_bt['state'] = 'normal'
        self.new_polygon_bt['relief'] = 'sunken'
        self.new_polygon_bt['text'] = 'Cancel'
        self.new_polygon_bt['command'] = self.cancel_polygon
        choose_polygon = True
        creating_polygon = False

    def finish_polygon(self):
        global creating_polygon, choose_polygon, total_number_areas
        if self.canvas._finish_polygon() != 'too_few_points':
            choose_polygon = False
            creating_polygon = False
            total_number_areas += 1
            self.new_polygon_bt.configure(relief='raised', text="add new polygon", command=self.polygon)
            self.back_bt['state'] = 'normal'
            self.home_bt['state'] = 'normal'
            self.finished_bt['state'] = 'normal'
            self.paint_rectangle_bt['state'] = 'normal'
            self.undo_polygon_bt['state'] = 'disabled'
            self.finish_polygon_bt['state'] = 'disabled'

    def undo_polygon(self):
        global creating_polygon, choose_polygon
        if len(self.canvas.polygon_groundstructure) > 1:  # remove 2 (2 coordinates from the polygon)
            for i in range(2):
                self.canvas.canvas.delete(self.canvas.polygon_groundstructure.pop())
                self.canvas.polygon_points.pop()
        elif len(self.canvas.polygon_groundstructure) == 1:  # only one point painted on the canvas
            self.canvas.canvas.delete(self.canvas.polygon_groundstructure.pop())
            self.canvas.polygon_groundstructure.clear()
            self.canvas.polygon_points.clear()
            self.new_polygon_bt.configure(relief='raised', text="add new polygon", command=self.polygon)
            self.back_bt['state'] = 'normal'
            self.home_bt['state'] = 'normal'
            self.finished_bt['state'] = 'normal'
            self.paint_rectangle_bt['state'] = 'normal'
            self.undo_polygon_bt['state'] = 'disabled'
            self.finish_polygon_bt['state'] = 'disabled'
            choose_polygon = False
            creating_polygon = False

    def cancel_polygon(self):
        global creating_polygon, choose_polygon
        for p in self.canvas.polygon_groundstructure:
            self.canvas.canvas.delete(p)
        self.canvas.polygon_groundstructure.clear()
        self.canvas.polygon_points.clear()
        choose_polygon = False
        creating_polygon = False
        self.new_polygon_bt.configure(text='add a new polygon', relief='raised', command=self.polygon, state='normal')
        self.back_bt['state'] = 'normal'
        self.home_bt['state'] = 'normal'
        self.finished_bt['state'] = 'normal'
        self.paint_rectangle_bt['state'] = 'normal'
        self.undo_polygon_bt['state'] = 'disabled'
        self.finish_polygon_bt['state'] = 'disabled'

    def remove_latest(self):
        global undo_stack, total_number_areas
        if total_number_areas > 0:
            total_number_areas -= 1
        if len(undo_stack) == 0:
            return
        elif undo_stack_ear.pop() == 'p':  # last area was a polygon
            self.canvas.canvas.delete(self.canvas.polygons.pop())
        else:  # last area was a rectangle
            self.canvas.canvas.delete(self.canvas.rectangles.pop())

    """save files locally to .tmp"""

    def _save(self):
        global global_state
        img_name = self.label_image_names[global_state]
        something_to_save = False
        img = cv2.imread(self.filename, cv2.IMREAD_COLOR)
        height, width, channels = img.shape
        label_image = np.zeros((height, width, 1), dtype=np.uint8)
        old_label_image = None
        if len(self.canvas.rectangles) > 0 or len(self.canvas.polygons) > 0:
            something_to_save = True
            self.__paintPixelsInsideRectangles(self.canvas.rectangles, label_image, self.label_list[global_state])
            self.__getPixelsInsidePolygons(self.canvas.polygons, label_image, self.label_list[global_state])
            if self.hierarchical and global_state > 0:
                old_label_image = cv2.imread(os.path.join(os.getcwd(), WORKING_DIRECTORY,
                                                          self.label_image_names[global_state-1]), cv2.IMREAD_GRAYSCALE)
        if something_to_save:
            if not os.path.join(os.getcwd(), WORKING_DIRECTORY, img_name) in self.potential_upload_files:
                self.potential_upload_files.append(os.path.join(os.getcwd(), WORKING_DIRECTORY, img_name))
            if old_label_image is not None:
                converted_label_image = np.reshape(label_image, (img.shape[0], img.shape[1]))
                label_image = cv2.bitwise_and(old_label_image, converted_label_image)
            cv2.imwrite(os.path.join(os.getcwd(), WORKING_DIRECTORY, img_name), label_image)
        else:
            self.potential_upload_files.remove(os.path.join(os.getcwd(), WORKING_DIRECTORY, img_name))
        if self.hierarchical and global_state >= 0:  # the image has to be blackened
            converted_label_image = np.reshape(label_image, (img.shape[0], img.shape[1]))
            img = cv2.bitwise_and(img, img, mask=converted_label_image)
            self.filename = os.path.join(os.getcwd(), WORKING_DIRECTORY, 'intermediate' +
                                         self.label_image_names[global_state-1])
            cv2.imwrite(self.filename, img)
        self.__silent_abort()
        self.canvas.destroy()
        self.canvas = None

    """fuse all annotations that have been made on this picture already"""

    def __fuse(self):  #
        tl = tk.Toplevel(self.root)
        status = tk.StringVar()
        status.set("Reading both annotations")
        label = tk.Label(tl, textvariable=status)
        label.pack()
        labeled_image_old_ear = None
        labeled_image_old_einzelreihe = None
        labeled_image_old_parzelle = None
        labeled_image_new_ear = None
        labeled_image_new_einzelreihe = None
        labeled_image_new_parzelle = None
        ear_label_image = None
        einzelreihe_label_image = None
        parzelle_label_image = None
        if os.path.exists(os.path.join(os.getcwd(), WORKING_DIRECTORY, FUSION_INTERMEDIATE_EAR)):
            labeled_image_old_ear = cv2.imread(os.path.join(os.getcwd(), WORKING_DIRECTORY, FUSION_INTERMEDIATE_EAR),
                                               cv2.IMREAD_GRAYSCALE)
            ear_label_image = labeled_image_old_ear
        if os.path.exists(os.path.join(os.getcwd(), WORKING_DIRECTORY, FUSION_INTERMEDIATE_EINZELREIHE)):
            labeled_image_old_einzelreihe = cv2.imread(os.path.join(os.getcwd(), WORKING_DIRECTORY,
                                                                    FUSION_INTERMEDIATE_EINZELREIHE),
                                                       cv2.IMREAD_GRAYSCALE)
            einzelreihe_label_image = labeled_image_old_einzelreihe
        if os.path.exists(os.path.join(os.getcwd(), WORKING_DIRECTORY, FUSION_INTERMEDIATE_PARZELLE)):
            labeled_image_old_parzelle = cv2.imread(os.path.join(os.getcwd(), WORKING_DIRECTORY,
                                                                 FUSION_INTERMEDIATE_PARZELLE), cv2.IMREAD_GRAYSCALE)
            parzelle_label_image = labeled_image_old_parzelle
        if os.path.exists(os.path.join(os.getcwd(), WORKING_DIRECTORY, EAR_LABEL_IMAGE)):
            labeled_image_new_ear = cv2.imread(os.path.join(os.getcwd(), WORKING_DIRECTORY, EAR_LABEL_IMAGE),
                                               cv2.IMREAD_GRAYSCALE)
            ear_label_image = labeled_image_new_ear
        if os.path.exists(os.path.join(os.getcwd(), WORKING_DIRECTORY, EINZELREIHE_LABEL_IMAGE)):
            labeled_image_new_einzelreihe = cv2.imread(os.path.join(os.getcwd(), WORKING_DIRECTORY,
                                                                    EINZELREIHE_LABEL_IMAGE), cv2.IMREAD_GRAYSCALE)
            einzelreihe_label_image = labeled_image_new_einzelreihe
        if os.path.exists(os.path.join(os.getcwd(), WORKING_DIRECTORY, PARZELLE_LABEL_IMAGE)):
            labeled_image_new_parzelle = cv2.imread(os.path.join(os.getcwd(), WORKING_DIRECTORY, PARZELLE_LABEL_IMAGE),
                                                    cv2.IMREAD_GRAYSCALE)
            parzelle_label_image = labeled_image_new_parzelle
        status.set("Fusing both annotations")
        label.update()
        if labeled_image_new_ear is not None and labeled_image_old_ear is not None:
            ear_label_image = cv2.bitwise_or(labeled_image_new_ear, labeled_image_old_ear)
        if labeled_image_new_einzelreihe is not None and labeled_image_old_einzelreihe is not None:
            einzelreihe_label_image = cv2.bitwise_or(labeled_image_new_einzelreihe, labeled_image_old_einzelreihe)
        if labeled_image_old_parzelle is not None and labeled_image_new_parzelle is not None:
            parzelle_label_image = cv2.bitwise_or(labeled_image_new_parzelle, labeled_image_old_parzelle)
        if ear_label_image is not None:
            cv2.imwrite(os.path.join(os.getcwd(), WORKING_DIRECTORY, EAR_LABEL_IMAGE),
                        ear_label_image)
        elif (ear_label_image is None and
              os.path.join(os.getcwd(), WORKING_DIRECTORY, EAR_LABEL_IMAGE) in self.potential_upload_files):
            self.potential_upload_files.remove(os.path.join(os.getcwd(), WORKING_DIRECTORY, EAR_LABEL_IMAGE))
        if einzelreihe_label_image is not None:
            cv2.imwrite(os.path.join(os.getcwd(), WORKING_DIRECTORY, EINZELREIHE_LABEL_IMAGE),
                        einzelreihe_label_image)
        elif (einzelreihe_label_image is None and
              os.path.join(os.getcwd(), WORKING_DIRECTORY, EINZELREIHE_LABEL_IMAGE) in self.potential_upload_files):
            self.potential_upload_files.remove(os.path.join(os.getcwd(), WORKING_DIRECTORY, EINZELREIHE_LABEL_IMAGE))
        if parzelle_label_image is not None:
            cv2.imwrite(os.path.join(os.getcwd(), WORKING_DIRECTORY, PARZELLE_LABEL_IMAGE),
                        parzelle_label_image)
        elif (parzelle_label_image is None and
              os.path.join(os.getcwd(), WORKING_DIRECTORY, PARZELLE_LABEL_IMAGE) in self.potential_upload_files):
            self.potential_upload_files.remove(os.path.join(os.getcwd(), WORKING_DIRECTORY, PARZELLE_LABEL_IMAGE))
        status.set("Saved fused annotations. Now uploading them")
        label.update()
        label.destroy()
        fh = FileHandler(tl).save(self.flattened_picture_name, self.potential_upload_files,
                                  overwrite_old_files=True)
        return fh

    def __silent_abort(self):
        global choose_rectangle, choose_polygon, creating_polygon, undo_stack
        if self.canvas is not None:
            for r in self.canvas.rectangles:
                self.canvas.canvas.delete(r)
            for p in self.canvas.polygons:
                self.canvas.canvas.delete(p)
            for p in self.canvas.polygon_groundstructure:
                self.canvas.canvas.delete(p)
            self.canvas.polygon_groundstructure.clear()
            self.canvas.polygons.clear()
            self.canvas.rectangles.clear()
            self.canvas.clicked_rectangle = 0
            self.canvas.clicked_r_x = 0
            self.canvas.clicked_r_y = 0
            self.canvas.clicked_polygon = None
            self.canvas.clicked_point_on_polygon = None
            self.canvas.polygon_points_history.clear()
        choose_rectangle = False
        choose_polygon = False
        creating_polygon = False
        undo_stack = []

    def __getPixelsInsidePolygons(self, polygons, label_image, label_value):
        # get the position of the img box on the canvas
        box_img_int = tuple(map(int, self.canvas.canvas.coords(self.canvas.container)))[0:2]
        # set vector coords according to the img_box
        for poly in polygons:
            points = np.array(self.canvas.canvas.coords(poly), dtype=np.int).reshape(
                (int(len(self.canvas.canvas.coords(poly)) / 2), 2))
            points = points - box_img_int
            points = np.true_divide(points, self.canvas.imscale).astype(int)
            points = np.int32([points])
            print(points)
            cv2.fillPoly(img=label_image, pts=points, color=label_value)

    def __paintPixelsInsideRectangles(self, rectangles, label_image, label_value):
        box_img_int = tuple(map(int, self.canvas.canvas.coords(self.canvas.container)))[0:2]
        for rect in rectangles:
            vector = np.array(self.canvas.canvas.coords(rect), dtype=np.int)
            # set vector coords according to the img_box
            vector[0:2] = vector[0:2] - box_img_int
            vector[2:4] = vector[2:4] - box_img_int
            vector = np.true_divide(vector, self.canvas.imscale).astype(int)
            label_image = cv2.rectangle(img=label_image, pt1=tuple(vector[0:2]), pt2=tuple(vector[2:4]),
                                        color=label_value, thickness=-1)


"""FOLLOWING CODE WAS TAKEN FROM  
FooBar167 foobar167
Stackoverflow: https://stackoverflow.com/questions/41656176/tkinter-canvas-zoom-move-pan """


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

    def _finish_polygon(self):
        """remove the basic-structure and redraw a actual polygon"""
        global undo_stack, choose_polygon
        if len(self.polygon_points) < 6:
            messagebox.showinfo(title='Info', message='Too few points for a polygon')
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
        self.parent_class.activate_save_bt()

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


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)  # change to current directory
    root = tk.Tk()
    app = MeanWindow(root)
    root.mainloop()

"""
    def copy_object(self):
        return_copy = CanvasImage(self.placeholder, self.path, self.parent_class)
        return_copy.polygon_points = copy.deepcopy(self.polygon_points)
        return_copy.polygon_groundstructure = copy.deepcopy(self.polygon_groundstructure)
        return_copy.parzelle_polygons = copy.deepcopy(self.parzelle_polygons)
        return_copy.parzelle_rectangles = copy.deepcopy(self.parzelle_rectangles)
        return_copy.einzelreihe_polygons = copy.deepcopy(self.einzelreihe_polygons)
        return_copy.einzelreihe_rectangles = copy.deepcopy(self.einzelreihe_rectangles)
        return return_copy

    def restore_old_object(self, old):
        print(old.einzelreihe_rectangles)
        [self.canvas.create_rectangle(r) for r in old.parzelle_rectangles]
        [self.canvas.create_rectangle(r) for r in old.einzelreihe_rectangles]
        [self.canvas.create_polygon(p) for p in old.parzelle_polygons]
        [self.canvas.create_polygon(p) for p in old.einzelreihe_polygons]"""
