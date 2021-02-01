import tkinter


class Mbox(object):
    def __init__(self, root, msg):
        tki = tkinter
        self.top = root
        self.top.attributes("-topmost", True)
        self.return_value = None
        frm = tki.Frame(self.top, borderwidth=4, relief='ridge')
        frm.pack(fill='both', expand=True)

        label = tki.Label(frm, text=msg)
        label.pack(padx=4, pady=4)

        fuse_bt = tki.Button(frm, text='fuse both annotations', command=self.fuse, fg='green')
        fuse_bt.pack()
        new_file_bt = tki.Button(frm, text='create a new file', command=self.new_file, fg='green')
        new_file_bt.pack()
        overwrite_bt = tki.Button(frm, text='overwrite old annotations', command=self.overwrite, fg='orange')
        overwrite_bt.pack()
        discard_bt = tki.Button(frm, text='discard all new changes', command=self.discard, fg='red')
        discard_bt.pack()
        b_cancel = tki.Button(frm, text='Cancel', command=self.cancel)
        b_cancel.pack(padx=4, pady=4)

    def fuse(self):
        self.top.destroy()
        self.return_value = 'fuse'

    def new_file(self):
        self.top.destroy()
        self.return_value = 'new_file'

    def overwrite(self):
        self.top.destroy()
        self.return_value = 'overwrite'

    def discard(self):
        self.top.destroy()
        self.return_value = 'discard'

    def cancel(self):
        self.top.destroy()
        self.return_value = 'cancel'

    def show(self):
        self.top.deiconify()
        self.top.wait_window()
        return self.return_value
