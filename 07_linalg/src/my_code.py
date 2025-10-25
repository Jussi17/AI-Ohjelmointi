# M7 Kirjoita funktio, joka palauttaa annetun vektorin suuntaisen
# yksikkövektorin.

import sys
import time
import numpy as np

def unit(a):
    norm_a = np.linalg.norm(a)
    if norm_a == 0:
        raise ValueError("Ei voida luoda yksikkövektoria nollavektorista.")
    return a / norm_a
