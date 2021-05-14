#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

filename = 'train-images-idx3-ubyte'
index = int(input("Please input which picture do you want? "))
with open(filename, 'rb') as f_obj:
    f_obj.seek(16 + (index - 1) * 28 * 28)
    image1 = f_obj.read(28 * 28)
    image2 = np.zeros(28 * 28, dtype=float)
    print(type(image1), type(image2))
    for i in range(28 * 28):
        image2[i] = image1[i] / 256
    image2 = image2.reshape(28, 28)

plt.imshow(image2, cmap='binary')
plt.show()
