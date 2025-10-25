# M3 Lue taulukot tiedostoista a.npy ja b.npy Talleta ne molemmat
# taulukot tiedostoon ab.npz. Tallenna ensimmäinen taulukko
# nimellä a ja jälkimmäinen nimellä b.

import numpy as np

filename = 'a.npy'
a = np.load(filename)

filename = 'b.npy'
b = np.load(filename)

np.savez('ab.npz', a=a, b=b)