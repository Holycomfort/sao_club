import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn import decomposition, manifold


font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc', size=8)
#plt.rcParams['font.sans-serif']=['STSong']  
#plt.rcParams['axes.unicode_minus']=False 

df = pd.read_excel('city_dist.xlsx')
cities = list(df.keys())[1:]
dis = np.array(df.iloc[:, 1:])

mds = manifold.MDS(n_components=2)
mds.fit(dis)
coords = mds.fit_transform(dis)
print(coords)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
colors = np.random.uniform(0, 1, (len(cities), 3))
ax.scatter(coords[:, 0], coords[:, 1], color=colors, label=cities)
for i, city in enumerate(cities):
    ax.text(coords[i, 0], coords[i, 1], city, fontproperties=font)
plt.show()