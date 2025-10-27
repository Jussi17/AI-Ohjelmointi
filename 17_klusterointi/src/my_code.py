# M17 Tiedostossa painteddata.csv on pisteiden x- ja
# y-koordinaatteja. Kirjoita ohjelma, joka tulosta montako
# klusteria pisteet muodostavat. (Tehtävä pohjassa olevassa datassa
# klustereita on 6)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

basedir = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(basedir, "painteddata.csv"))

#Your code here
X = data[["x", "y"]].values
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X)
n_clusters = len(np.unique(kmeans.labels_))


print(n_clusters)
