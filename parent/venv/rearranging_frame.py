import tkinter as tk


class RearrangingFrame(tk.Frame):
    def __init__(self, parent_class, container, *args, **kwargs):
        self.parent_class = parent_class
        super().__init__(container, *args, **kwargs)
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def on_resize(self, event):
        hscale = int(float(event.height) / self.height)
        if self.parent_class.diseased_area_fr is not None:  # all the widgets have been added to the frame
            if (self.parent_class.diseased_area_fr.winfo_height() * 4) < hscale:
                self.parent_class.healthy_area_fr.grid(column=0, row=2)
                self.parent_class.diseased_area_fr.grid(column=0, row=3)
            else:
                self.parent_class.healthy_area_fr.grid(column=1, row=0)
                self.parent_class.diseased_area_fr.grid(column=1, row=1)
