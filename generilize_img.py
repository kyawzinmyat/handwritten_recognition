from PIL import Image
import csv
import numpy as np
import tensorflow as tf
import random
from tensorflow import keras
from shift import *



def gen(x, y, writer):
    for data,y_ in zip(x, y):
        img = Image.fromarray(data)
        if random.randint(0, 2) == 0:
            rot = random.choice([i for i in range(0, 90, 1)])
            img = img.rotate(rot)
            img = np.array(img)
        else:
            img = np.array(img)
            shift(random.choice(["left",'right','top','bottom'])
            , img, random.randint(0, 3))
        img = img.reshape(784,1).tolist()
        img = [i[0] for i in img]
        write_row(writer, img + [y_])

def gen_all_rand(x, y, writer):
    for data,y_ in zip(x, y):
        img = Image.fromarray(data)
        rot = random.choice([i for i in range(0, 90, 1)])
        img = img.rotate(rot)        
        img = np.array(img)
        shift(random.choice(["left",'right','top','bottom'])
            , img, random.randint(0, 3))
        img = img.reshape(784,1).tolist()
        img = [i[0] for i in img]
        write_row(writer, img + [y_])

def write_row(writer, row):
    writer.writerow(row)

def rotate(deg, img_obj):
    return img.rotate(deg)

def translate(direction, steps,img_arr):
    shift(dir, img, steps)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test/255.0
file = open("imgs1.csv", "a+")
writer = csv.writer(file)
gen(x_train, y_train, writer)
gen(x_test, y_test, writer)

file.close()

