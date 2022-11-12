import tkinter as tk
from tkinter import *
import random
import numpy as np
from simple_nn import NN, Layer, load_data
import pickle
import csv
from tensorflow import keras



class cell:
    def __init__(self, canvas, x, y):
        self.x = x
        self.y = y
        self.canvas = canvas
        self.left = 0
        self.top = 0
        # the value store the color value 1 for white and 0 for black
        self.value = 0

    
    def reset_value(self):
        self.value = 0
    
    def draw(self, size, margin = 0, clr = "white"):
        start, end = self.x * size + margin, self.y * size + margin
        self.left, self.top = start, end
        self.canvas.create_rectangle(start, end, start + size, end + size, fill = clr, outline = '')
        
    def draw_(self, size, set_val = 0.989, clr = "white"):
        self.canvas.create_rectangle(self.left, self.top, self.left + size, self.top + size, fill = clr, outline = "")
        self.value = set_val

class grid:
    def __init__(self, canvas, size = 10):
        self.size = size
        self.margin = 50    
        self.load_nn()
        self.is_rand = False
        self.labels = None
        self.img = [
            [0 for i in range(28)] for j in range(28)
        ]
        self.canvas = canvas
        self.g = [   
                [cell(canvas, j, i) for i in range(28)] for j in range(28)
        ]
        self.last_x, self.last_y = 0, 0
        self.load_img()
       
    def load_img(self):
        #train, labels = load_data("imgs2.csv")
        (train, labels), (x_test, y_test) = keras.datasets.mnist.load_data()
        train = [
            x.reshape(784, 1) for x in train
        ]
        self.testset = [
            (x, y) for x, y in zip(train, labels)
        ]

    def load_nn(self):
        file = open("nn1.pkl", "rb")
        self.nn = pickle.load(file)
        file.close()

    def set_labels(self, labels):
        self.labels = labels

    def reset_redraw(self):
        self.is_rand = False
        self.canvas.delete('all')
        self.draw()
        for i in range(28):
            for j in range(28):
                self.img[i][j] = 0
                self.g[i][j].reset_value()
        result_labels[0].config(text = "Actual - " )
        result_labels[1].config(text = "Predicted - ")
        result_labels[1].config(bg = "green")
        for i, label in enumerate(labels):
                label.config(text = f"{i} - {0:.2f}")

    def draw(self): 
        for i in range(28):
          for j in range(28):
            self.g[j][i].draw(self.size, self.margin, "black")
    
    
    
    def fill_rect(self, event):
        if not self.is_rand:
            x = int((self.last_x  - self.margin) / self.size)
            y = int((self.last_y  - self.margin) / self.size)
            if x >= 0 and x < 28 and y >= 0 and y < 28:
                x0, x1 = (x - 1 if x - 1 >= 0 else 0), (x + 1 if x + 1 < 28 else 27)
                y0, y1 = (y - 1 if y - 1 >= 0 else 0), (y + 1 if y + 1 < 28 else 27)
                self.g[x][y].value = 0.9
                for i in range(x0, x1):
                    for j in range(y0, y1):
                        self.g[i][j].draw_(self.size, 1 / 0.9, "white")
                self.g[x][y].draw_(self.size, 0.9, "white")
                self.last_x, self.last_y = event.x, event.y
        
    def random_test(self):
        self.is_rand = True
        inp, lab = random.choice(self.testset)
        a = self.predict(inp)
        inp = inp.reshape(28, 28)
        inp = inp.tolist()
        actual = str(lab)
        pred = str(np.argmax(a))
        bg = "red"
        if actual == pred:
            bg = "green"
        result_labels[0].config(text = "Actual - " + actual)
        result_labels[1].config(text = "Predicted - " + pred)
        result_labels[1].config(bg = bg)
        for i in range(28):
            for j in range(28):
                self.g[i][j].value = inp[i][j]
                clr_code = "#" + ("f"  * 3) if inp[i][j] != 0 else "#" + (str(int(inp[i][j])) * 3)
                self.g[j][i].draw(self.size, self.margin, clr_code) 


    def handle(self, event):
        self.last_x, self.last_y = event.x, event.y

    def extract_img(self, event):
        if not self.is_rand:
            for i in range(28):
                for j in range(28):
                    self.img[j][i] = self.g[i][j].value
            inp = np.matrix(self.img).reshape(784, 1)
            a = self.predict(inp).tolist()
            a = [i[0] for i in a]
            index = [i for i in range(10)]
            if self.labels:
                bubble_sort(a, index)
                for label, i, act in zip(labels, index, a):
                    label.config(text = f"{i} - {act:.2f}")


    def predict(self, inp):
        prev_m = 0
        m = 0 
        a = self.nn.feedforward(inp)
        for i, j in enumerate(a):
            if j > prev_m:
                prev_m = j
                m = i
        return a
    

def bubble_sort(acts, index):
    counter = 1
    while counter != 0:
        counter = 0
        for i in range(len(acts) - 1):
            if acts[i] < acts[i + 1]:
                index[i], index[i + 1] = index[i + 1], index[i]
                acts[i], acts[i + 1] = acts[i + 1], acts[i]
                counter += 1
root = tk.Tk()
root.config(bg = "white")
root.geometry("800x600")
canvas = Canvas(root, width = 400, height = 400, background = "white", bd=0, highlightthickness=0)
last_x, last_y = 0, 0
#canvas.pack(padx = 0, pady = 0)
canvas.grid(row= 0, column = 1)

def save_img():
    
    tp  = Toplevel()
    tp.title("Save the image!")
    lab = Label(tp, text = "Label").pack(pady = 10)   
    lab2 = Label(tp, text = "").pack(pady = 10)  
    ent = Entry(tp)
    ent.pack(pady = 5)
    btn_save = Button(tp, text = "save", command = lambda : save_csv_file(ent, tp)).pack(pady = 10)
    


def save_csv_file(label, tp):
    import csv
    try:
        label = int(label.get())
        file = open("imgs.csv", "a+")
        writer = csv.writer(file)
        data = [g.img[i][j] for i in range(28) for j in range(28)]
        writer.writerow(data + [label])
        file.close()
        tp.destroy()
    except:
        pass


# 10 label text for score 
labels = []
can3 = Canvas(root, width = 100, height=150, background = "white", bd=0, highlightthickness=0)
can3.grid(row = 0, column = 2)
for i in range(10):
    labels.append(Label(can3, text = str(i) + " - 0.00", bg = "white", fg = "black",  font = ("Arial", 20)))
    labels[i].pack(padx = (0,30))

result_labels = []
can4 = Canvas(root, width = 50, height=550, background = "white", bd=0, highlightthickness=0)
can4.grid(row = 0, column = 0)
result_labels.append(Label(can4, text = "Actual - ", bg = "green", fg = "white", font = ("Arial", 25)))
result_labels.append(Label(can4, text = "Predicted - ", bg = "green", font = ("Arial", 25)))
for l in result_labels:
    l.pack(padx = (20, 10), pady = (20, 20))

g = grid(canvas, 10)
g.set_labels(labels)
g.draw()

can2 = Canvas(root, width = 400, height = 500, background = "white", bd=0, highlightthickness=0)
can2.grid(row = 1, column = 1, padx = 10)

but_clean = tk.Button(can2, text = "clean", bg = "white", bd = 0, command = g.reset_redraw, width = 10, height = 2)
but_clean.grid(row = 1, column = 0, padx = 10)

# let the user save the digit
but_save = tk.Button(can2, text = "save", bd = 0, command = save_img, width = 10, height=2)
but_save.grid(row = 1, column = 1, padx = 10)


but_ran = tk.Button(can2, text = "random", bd = 0, command = g.random_test,width = 10, height=2)
but_ran.grid(row = 1, column = 2)


canvas.bind("<Button-1>", g.handle)
canvas.bind("<B1-Motion>", g.fill_rect)
canvas.bind("<ButtonRelease-1>", g.extract_img)
root.mainloop()



