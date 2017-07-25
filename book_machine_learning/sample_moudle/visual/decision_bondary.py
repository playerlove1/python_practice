from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, resolution=0.02,ax=None):


    if ax is None:
        ax = plt.gca()
    
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    Z=classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    
    Z=Z.reshape(xx1.shape)
    
    # plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # plt.xlim(xx1.min(), xx1.max())
    # plt.ylim(xx2.min(), xx2.max())
    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    ax.axis(xmin=xx1.min(),xmax=xx1.max(),ymin=xx2.min(),ymax=xx2.max())
    
    # plot class sample
    
    for idx, cl in enumerate(np.unique(y)):
        # plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx) ,marker=markers[idx], label=cl)
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx) ,marker=markers[idx], label=cl)
    
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return ax
    # plt.xlabel('petal length [cm]')
    # plt.ylabel('sepal length [cm]')
    # plt.legend(loc='upper left')
    # plt.show()