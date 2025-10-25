# M1 Luo taulukko a, jossa on 3 riviä ja 4 saraketta ja kaikkien
# alkioiden arvo on 0. Talleta taulukko tiedostoon m1out.npy.

import numpy as np

A=np.zeros((3, 4))
np.save("m1out.npy", A)

print(A)

