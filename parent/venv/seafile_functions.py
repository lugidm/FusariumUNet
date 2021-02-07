import seafileapi  # https://github.com/haiwen/python-seafile/blob/master/doc.md#repo
from constants import *
import tkinter as tk
import os
from PIL import Image, ImageTk
from tkinter import messagebox
import numpy as np
import re
from message_box import Mbox

def resize_img(image, maxsize):
    r1 = image.size[0] / maxsize[0]  # width ratio
    r2 = image.size[1] / maxsize[1]  # height ratio
    ratio = max(r1, r2)
    newsize = (int(image.size[0] / ratio), int(image.size[1] / ratio))
    image = image.resize(newsize, Image.ANTIALIAS)
    return image


class FileWalker:
    def __init__(self, root):
        self.client = seafileapi.connect(server=SERVER, username=USERNAME, password=PASSWORD)
        self.root = root
        self.success = False
        self.file_path = ""
        self.file_ending = ""
        self.current_files = None
        # self.root.geometry("600x400")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.current_dir_path = tk.StringVar()
        self.current_dir_path.set('')
        self.infotext = tk.StringVar()
        self.infotext.set("Double-click on directories or files, latter to download it")

        self.files_fr = tk.Frame(root)
        self.files_fr.pack(expand=True, fill='both')

        self.files_navigation_fr = tk.Frame(self.files_fr)
        # [self.files_navigation_fr.grid_columnconfigure(i, weight=1)for i in range(3)]
        img = Image.open(os.path.join(os.getcwd(), 'icons', 'back.png'))
        img = resize_img(img, [30, 30])
        self.img = ImageTk.PhotoImage(img)
        self.dir_up_bt = tk.Button(self.files_navigation_fr, image=self.img, command=self.back)
        # self.dir_up_bt.grid(column=0, row=0, sticky='W')
        self.dir_up_bt.pack(side='left')
        self.current_directory_lb = tk.Label(self.files_navigation_fr, textvariable=self.current_dir_path, bg='white')
        # self.current_directory_lb.grid(row=0, column=1, sticky='WE')
        self.current_directory_lb.pack(side='left', expand=True)
        img = Image.open(os.path.join(os.getcwd(), 'icons', 'info.png'))
        img = resize_img(img, [30, 30])
        self.info_img = ImageTk.PhotoImage(img)
        self.info_bt = tk.Button(self.files_navigation_fr, image=self.info_img, command=self.show_info)
        # self.info_bt.grid(row=0, column=2, sticky='E')
        self.info_bt.pack(side='right')
        self.files_navigation_fr.pack(fill='x', side='top')

        self.directory_fr = tk.LabelFrame(self.files_fr, text='DIRECTORIES')
        self.directory_fr.pack(expand=True, fill='both', side='left')
        self.directory_fr.grid_columnconfigure(0, weight=1)
        self.directory_fr.grid_rowconfigure(0, weight=1)
        self.scrollbar_dirs = tk.Scrollbar(self.directory_fr, orient='vertical')
        self.directory_listbox = tk.Listbox(self.directory_fr, selectmode='single',
                                            yscrollcommand=self.scrollbar_dirs.set)
        self.directory_listbox.bind('<Double-Button-1>', self.__double_click_dirs_listbox)
        self.directory_listbox.grid(column=0, row=0, sticky='NSEW')
        self.scrollbar_dirs.config(command=self.directory_listbox.yview)
        self.scrollbar_dirs.grid(column=1, row=0, sticky='NS')

        self.files_in_directory_fr = tk.LabelFrame(self.files_fr, text="FILES")
        self.files_in_directory_fr.pack(expand=True, fill='both', side='right')
        self.files_in_directory_fr.grid_columnconfigure(0, weight=1)
        self.files_in_directory_fr.grid_rowconfigure(0, weight=1)
        self.scrollbar_files = tk.Scrollbar(self.files_in_directory_fr, orient='vertical')
        self.files_listbox = tk.Listbox(self.files_in_directory_fr, selectmode='single',
                                        yscrollcommand=self.scrollbar_files.set, setgrid=2)
        self.scrollbar_files.config(command=self.files_listbox.yview)
        self.files_listbox.bind('<Double-Button-1>', self.__double_click_files_listbox)
        self.files_listbox.grid(column=0, row=0, sticky='NSEW')
        self.scrollbar_files.grid(column=1, row=0, sticky='NSEW')

        self.repo_list = [rp.name for rp in self.client.repos.list_repos()]

        self.repo = self.client.repos.get_repo(REPOSITORY)
        self.current_folders = self.repo.list_directories('/').entries
        self.current_dir_path.set('/')
        self.update_listboxes()

    def back(self):
        if self.current_dir_path == '/':
            return
        path_string = self.current_dir_path.get()
        path_list = path_string.split('/')
        path_list = path_list[0:-1]
        separator = '/'
        path_string = separator.join(path_list)
        if path_string == '':
            self.current_dir_path.set(separator)
        else:
            self.current_dir_path.set(path_string)
        self.update_listboxes()

    def update_listboxes(self):
        """update the listboxes after current dir_path has been reset"""
        self.directory_listbox.delete(0, 'end')
        self.files_listbox.delete(0, 'end')
        self.current_folders = self.repo.list_directories(self.current_dir_path.get()).entries
        self.current_files = self.repo.get_dir(self.current_dir_path.get()).ls(force_refresh=True)
        if self.current_files:
            files = []
            [files.append(d.path) for d in self.current_files]
            for i in range(len(files)):
                files[i] = files[i].split('/')[-1]
        if self.current_folders and not self.current_files:  # only subdirs and no files
            [self.directory_listbox.insert('end', d.path) for d in self.current_folders]
        elif self.current_folders and self.current_files:  # some subdirs and some files in current folder
            [self.directory_listbox.insert('end', d.path) for d in self.current_folders]
            [self.files_listbox.insert('end', d) for d in files]
        elif self.current_files:  # only files in current folder
            [self.files_listbox.insert('end', d) for d in files]

    def __double_click_dirs_listbox(self, event):
        """ update label + listbox and go deeper into directory"""
        if self.directory_listbox.size() == 0:
            return
        items = self.directory_listbox.curselection()
        items = [self.current_folders[int(item)] for item in items]
        self.current_dir_path.set(items[0].path)
        self.update_listboxes()

    def __double_click_files_listbox(self, event):
        """ update files listbox and ask user if he wants to download the chosen file"""
        if self.files_listbox.size() == 0:
            return
        items = self.files_listbox.curselection()
        items = [self.current_files[int(item)] for item in items]
        self.file_path = items[0].path
        if messagebox.askyesno("DOWNLOAD", "Do you want to download the file " + self.file_path + "?"):
            if not os.path.exists(WORKING_DIRECTORY):
                os.mkdir(WORKING_DIRECTORY)
            seaffile = self.repo.get_file(self.file_path)
            content = seaffile.get_content()
            if (self.file_path.endswith('.JPG') or self.file_path.endswith('.jpg') or self.file_path.endswith(
                    '.JPEG') or
                    self.file_path.endswith('.jpeg') or self.file_path.endswith('.jpe') or self.file_path.endswith(
                        '.jfi')
                    or self.file_path.endswith('.jif') or self.file_path.endswith('.jfif')):
                f = open(os.path.join(os.getcwd(), WORKING_DIRECTORY, DOWNLOADED_IMAGE + '.jpg'), "wb")
                self.success = True
                self.file_ending = '.jpg'
            elif self.file_path.endswith('.png'):
                f = open(os.path.join(os.getcwd(), WORKING_DIRECTORY, DOWNLOADED_IMAGE + '.png'), "wb")
                self.success = True
                self.file_ending = '.png'
            elif self.file_path.endswith('.bmp') or self.file_path.endswith('.dib') or self.file_path.endswith('.BMP'):
                f = open(os.path.join(os.getcwd(), WORKING_DIRECTORY, DOWNLOADED_IMAGE + '.bmp'), "wb")
                self.success = True
                self.file_ending = '.bmp'
            else:
                messagebox.showerror('FILE-ERROR',
                                     'The downloaded file is either not a picture or not of a known image-format' +
                                     'please convert it to jpg, bmp, or bitmap!')
                return
            f.write(content)
            f.close()
        else:
            self.files_listbox.select_clear(0, 'end')
            return
        self.root.destroy()

    def show_info(self):
        messagebox.showinfo("INFO", "double-click on directories or files to either walk through them or download /" +
                            "annotate them if connection errors appear, check the file constants")

    def show(self):
        self.root.deiconify()
        self.root.wait_window()
        if self.success:
            return self.file_path, self.file_ending
        else:
            return "ERROR", self.file_ending


class FileHandler:
    def __init__(self, root):
        self.return_value = ""
        self.root = root
        self.status = tk.StringVar()
        self.status.set("Building up connection...")

        try:
            self.client = seafileapi.connect(server=SERVER, username=USERNAME, password=PASSWORD)
        except:
            answer = messagebox.askretrycancel("Connection Error!", "The internet-connection could not be established!"+
                                      " Please check your connection and hit retry. Or upload the annotation-results" +
                                      "by hand. They are in the .tmp Folder where the .exe file is")
            if answer is None:
                exit(1)
            elif answer == "retry":
                self.__init__(root)
                return
        self.repo_list = [rp.name for rp in self.client.repos.list_repos()]
        self.repo = self.client.repos.get_repo(REPOSITORY)
        self.current_folders = self.repo.list_directories('/')
        self.status = tk.StringVar()
        self.status.set("Connected to server...")
        self.label = tk.Label(self.root, textvariable=self.status)
        self.label.pack()

    def save(self, flattened_picture_name: str, upload_data_paths: list, overwrite_old_files=False):
        results_dir = '/'+ONLINE_RESULTS_DIRECTORY
        if not self.path_exists(results_dir):
            self.current_folders = self.current_folders.mkdir(results_dir)
        self.current_folders = self.repo.get_dir(results_dir)
        if not self.path_exists(flattened_picture_name):
            self.current_folders = self.current_folders.mkdir(flattened_picture_name)
            for path in upload_data_paths:
                self.status.set("Uploading "+str(path)+"...")
                self.label.update()
                self.current_folders.upload(open(path, 'rb'), os.path.split(path)[-1])
            self.return_value = "success"
        elif overwrite_old_files:  # when fusing
            self.current_folders = self.repo.get_dir(results_dir + '/' + flattened_picture_name)
            self.current_folders.delete()
            self.current_folders = self.repo.get_dir(results_dir)
            self.current_folders = self.current_folders.mkdir(flattened_picture_name)
            for path in upload_data_paths:
                self.status.set("Uploading "+str(path)+"...")
                self.label.update()
                self.current_folders.upload(open(path, 'rb'), os.path.split(path)[-1])
            self.return_value = "success"
        else:  # image with the same name was already uploaded
            answer = Mbox(tk.Toplevel(self.root), "This image was already annotated!").show()
            if answer == "fuse": # fuse
                self.return_value = answer
                self.status.set("Downloading old annotations...")
                self.label.update()
                self.current_folders = self.repo.get_dir(results_dir + '/'+ flattened_picture_name)
                if self.path_exists(EAR_LABEL_IMAGE):
                    self.download_bmp(results_dir + '/' + flattened_picture_name + '/' + EAR_LABEL_IMAGE,
                                  os.path.join(os.getcwd(), WORKING_DIRECTORY, FUSION_INTERMEDIATE_EAR))
                if self.path_exists(EINZELREIHE_LABEL_IMAGE):
                    self.download_bmp(results_dir + '/' + flattened_picture_name + '/' + EINZELREIHE_LABEL_IMAGE,
                                      os.path.join(os.getcwd(), WORKING_DIRECTORY, FUSION_INTERMEDIATE_EINZELREIHE))
                if self.path_exists(PARZELLE_LABEL_IMAGE):
                    self.download_bmp(results_dir + '/' + flattened_picture_name + '/' + PARZELLE_LABEL_IMAGE,
                                      os.path.join(os.getcwd(), WORKING_DIRECTORY, FUSION_INTERMEDIATE_PARZELLE))
                self.status.set("Fusing...")
                self.label.update()
            elif answer == "new_file":
                self.current_folders = self.repo.get_dir(results_dir)
                nr = 0
                for i in range(999999):
                    if not self.path_exists(flattened_picture_name+'('+str(i)+')'):
                        nr = i
                        break
                self.current_folders = self.current_folders.mkdir(flattened_picture_name + '('+str(nr)+')')
                for path in upload_data_paths:
                    self.status.set("Uploading " + str(path) + "...")
                    self.label.update()
                    self.current_folders.upload(open(path, 'rb'), os.path.split(path)[-1])
                self.return_value = "success"
                self.return_value = answer + "<" + self.current_folders.path

            elif answer == "overwrite":  # overwrite
                self.status.set("Deleting old annotations")
                self.label.update()
                self.current_folders = self.repo.get_dir(results_dir + '/' + flattened_picture_name)
                self.current_folders.delete()
                self.current_folders = self.repo.get_dir(results_dir)
                self.current_folders = self.current_folders.mkdir(flattened_picture_name)
                for path in upload_data_paths:
                    self.status.set("Uploading " + str(path) + "...")
                    self.label.update()
                    f = open(path, 'rb')
                    self.current_folders.upload(f, os.path.split(path)[-1])
                    f.close()
                self.return_value = "overwrite"
            elif answer == "discard" :
                self.return_value = "discard"
            elif answer == "cancel" or answer is None:
                self.return_value = "cancel"
        self.root.destroy()
        return self.return_value

    def download_bmp(self, online_path, local_path):
        seaffile = self.repo.get_file(online_path)
        content = seaffile.get_content()
        f = open(local_path, "wb")
        f.write(content)
        f.close()

    """something like os.pathexists"""
    def path_exists(self, path):
        path = re.split('/', path)[-1]
        exists = False
        for folders in self.current_folders.entries:
            if re.split('/', folders.path)[-1] == path:
                exists = True
                break
        return exists
"""

root = tk.Tk()
fh= FileHandler(root)
fh.save("bla", [os.path.join(os.getcwd(), "1o.jpg")])
root.mainloop()"""