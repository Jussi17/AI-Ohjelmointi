# M5 Kirjoita funktio, joka tarkastaa onko kaksi vektoria toisiaan
# vastaan kohtisuorassa. Funktio palauttaa True tai False.

import sys
import time
import numpy as np


def is_orthogonal(a, b):
    return np.isclose(np.dot(a, b), 0)

print (is_orthogonal(np.array([1, 2, 3]), np.array([-2, 1, 0])))  # True
print (is_orthogonal(np.array([1, 2, 3]), np.array([4, 5, 6])))    # False