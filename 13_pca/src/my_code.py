# M13 Toteuta ohjelma, joka lukee mittaustulokset in.npy-tiedostosta
# ja pakkaa ne PCA:n avulla pienempään dimensioon. Määrää
# pienempi dimensio n siten, että explained variance :n
# sisältämistä(ominais)arvoista pienin mukaan otettava on 1/10
# suurimmasta arvosta. Talleta pakattu data tiedostoon out.npy.

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

basedir = os.path.dirname(__file__)
inputfile=os.path.join(basedir, 'in.npy')
outputfile=os.path.join(basedir, 'out.npy')

data=np.load(inputfile)

pca = PCA()
pca.fit(data)

explained_variance = pca.explained_variance_
min_threshold = explained_variance.max() / 10
n_components = (explained_variance >= min_threshold).sum()

pca = PCA(n_components=n_components)
packed_data = pca.fit_transform(data)

np.save(outputfile, packed_data)
