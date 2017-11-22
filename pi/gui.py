import tkinter as tk
from PIL import Image, ImageTk
from itertools import count
import time

#ImageLabel class for gifs
class ImageLabel(tk.Label):
    def load(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        self.loc = 0
        self.frames = []

        try:
            for i in count(1):
                self.frames.append(ImageTk.PhotoImage(img.copy()))
                img.seek(i)
        except EOFError:
            pass
        
        try:
            #If a delay is specified in the gif, use it
            self.delay = img.info['duration']
        except:
            #Otherwise sets delay between images
            self.delay = 10
        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None
        
    #Method that plays the gifs
    def next_frame(self):
        if self.frames:
            self.loc += 1
            if self.loc == len(self.frames):
                time.sleep(0.25)
                self.config(image='')
                return self.loc
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)
        return -1

#Class for the tkinter GUI
class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("1050x750")
        self.root.configure(background='white')
        self.label = ImageLabel(self.root)
        self.label.pack()
    def show_move(self, filename):
        self.label.unload()
        self.label.load(filename)
        self.root.after(1300, lambda: self.root.quit())
        self.root.mainloop()
