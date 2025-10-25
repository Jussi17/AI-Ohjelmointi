# M4 Lue tiedosto m4in.npz ja talleta siitä taulukko jonka nimi on b
# tiedostoon m4out.npy.

import numpy as np

filename = 'm4in.npz'
b = np.load(filename)['b']
np.save('m4out.npy', b)
