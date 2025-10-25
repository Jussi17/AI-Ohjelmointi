# M2 Lue taulukko tiedostosta m2in.npy ja aseta siinä indeksin 0, 0
# arvo 1 ja oikeaan alakulmaan arvo -1. Talleta muokattu taulukko
# tiedostoon m2out.npy. 

import numpy as np

filename = 'm2in.npy'
m2 = np.load(filename)

m2[0, 0] = 1
m2[-1, -1] = -1
np.save('m2out.npy', m2)