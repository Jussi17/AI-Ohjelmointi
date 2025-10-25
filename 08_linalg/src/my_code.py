# M8 Kirjoita funktio, joka palauttaa x·y
#                                     --- y
#                                     y·y  , kun x ja y annetaan
# funktion parametreina. Tätä kutsutaan vektorin x projektioksi y:lle

import sys
import time
import numpy as np

def project(x, y):
    dot_product_xy = np.dot(x, y)
    dot_product_yy = np.dot(y, y)
    projection = (dot_product_xy / dot_product_yy) * y
    return projection
