import numpy as np
from sklearn import decomposition, manifold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


data = np.load('mnist.npz')
X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X = np.concatenate([X_train, X_test])

y_train = np.array([np.argmax(one_hot) for one_hot in y_train])
y_test = np.array([np.argmax(one_hot) for one_hot in y_test])

'''
### Q1
X_train_0 = X_train[y_train == 0]
X_train_8 = X_train[y_train == 8]
X_train_08 = np.concatenate([X_train_0, X_train_8])
#tool = decomposition.PCA(n_components=2)
#tool = manifold.Isomap(n_components=2)
#tool = manifold.LocallyLinearEmbedding(n_components=2)
tool = manifold.TSNE(n_components=2)
coords = tool.fit_transform(X_train_08)
num_0 = len(X_train_0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(coords[:num_0, 0], coords[:num_0, 1], color=(0.5, 0.5, 0.5))
ax.scatter(coords[num_0:, 0], coords[num_0:, 1], color=(0.1, 0.9, 0.5))
plt.show()
'''

### Q2
num_train = len(X_train)
pca = decomposition.PCA(n_components=300)
new_X = pca.fit_transform(X)
X_train = new_X[:num_train]
X_test = new_X[num_train:]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

clf = LogisticRegression()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))