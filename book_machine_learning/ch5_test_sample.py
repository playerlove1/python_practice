import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles

from matplotlib.ticker import FormatStrFormatter

from sample_moudle.visual.plot_decision_regions import plot_decision_regions
from sample_moudle.algorithm.ch5.rbf_kernel_pca import rbf_kernel_pca
from sample_moudle.algorithm.ch5.rbf_kernel_pca_project import rbf_kernel_pca_project

uci_wine_url='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine = pd.read_csv(uci_wine_url, header=None)
X, y = df_wine.iloc[:,1:].values ,df_wine.iloc[:,0].values
#使用sklearn的cross_validation模組的train_test_split函數  將原始訓練資料切割為訓練集與測試集  (將資料切成30%測試 70%訓練)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#使用sklearn的preprocessing模組的StandardScaler來做標準化 (標準化:讓特徵值滿足標準常態分配 Standard normal distribution 並且每個特徵的平均值都是0)
stdsc = StandardScaler()
stdsc.fit(X_train)
X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)


'''
PCA之步驟
1.標準化d維dataset
2.建立共變異數矩陣 (covariance matrix)
3.分解「共變異數矩陣」為「特徵向量」(eigenvector)與「特徵值」(eigenvalues)
4.選取k個最大的特徵值 與其對應的 特徵向量， k是新「特徵空間」的維度數 (k<=d)
5.使用被選出的k個「特徵向量」，建立「投影矩陣」W
6.使用投影矩陣W，轉換輸入是d維的dataset，輸出新的k維「特徵子空間」
'''
#利用NumPy中的linalg.eig的函數來計算 共變異矩陣中的特徵值 與特徵向量 數對
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

#利用NumPy的cumsum函數計算  解釋變異數的 總和
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()


#將原先的資料轉換到PCA的主成分軸上
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse = True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n',w)

print('將一個樣本利用投影矩陣轉換的結果：',X_train_std[0].dot(w))

#劃出投影後的資料2維呈現
X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],X_train_pca[y_train==l, 1], c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower loft')
plt.show()


#利用scukit-learn中的PCA 所產生之圖 (與自己做的呈鏡像 因處理特徵解的工具不同 若在意的話 將資料乘以-1即為相同) 
pca = PCA(n_components = 2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
plot_decision_regions(X = X_train_pca,y = y_train, clf = lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

#決策區域在測試集的表現
plot_decision_regions(X = X_test_pca,y = y_test, clf = lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

#顯示各特徵的解釋變異
pca = PCA(n_components = None)
X_train_pca = pca.fit_transform(X_train_std)
print('pca解釋變異數:',pca.explained_variance_ratio_)


#利用LDA(Linear discriminant analysis, LDA)做特徵提取  藉由 降維 來處理維度災難，對非正規化的模型可以提高計算效率並降低over fit的程度
'''
PCA是試圖找出在一個dataset中 最大化變異數的 正交成分軸  而 LDA是嘗試找出可以最佳化類別分離 的 特徵子空間
兩者都是利用 線性轉換技術  來減少dataset當中的維度  而PCA是非監督式演算法  而LDA是監督式演算法
LDA的假設是 資料是常態分佈的，假設類別具有相同的「共變異數矩陣」，「特徵」在統計上也應該要是彼此獨立的

LDA之步驟
1.標準化d維dataset
2.對訓練資料中的每個類別，計算d維的「平均值向量」(mean vector)
3.建立 類別間(Between-class) 的 散佈矩陣(scatter matrix)Sb 與 類別內(within-class)的散佈矩陣(scatter matrix) Sw
4.由 inverse(Sw)Sb 矩陣中計算 「特徵向量」(eigenvector)與「特徵值」(eigenvalues)
5.選擇最大的k個  特徵值 與對應的 特徵向量，建立 d x k維的轉換矩陣w
6. 使用轉換矩陣w 將資料投影到新的 「特徵子空間」中
'''

np.set_printoptions(precision=4)
#平均向量list
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
    print('MV %s: %s\n' %(label, mean_vecs[label-1]))
#計算類別內的散佈矩陣Sw

#LDA假設 類別標籤 是均勻分佈的 但實際情況 如下所示
# print('Class label distribution: %s' %  np.bincount(y_train)[1:])

print('Class label distribution: %s'       % np.bincount(y_train.astype(np.int32))[1:])


#因此在計算sw前要先將各別的散佈矩陣除以類別樣本個數

#特徵維度為13
d=13
S_W = np.zeros((d, d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print('Within-class scatter matrix %sx%s' % (S_W.shape[0], S_W.shape[1]))

#計算類別間的散佈矩陣Sb
mean_overall = np.mean(X_train_std, axis=0)
#特徵維度為13
d=13
S_B = np.zeros((d, d))
for i,mean_vec in enumerate(mean_vecs):
    n = X[y==i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d,1)
    mean_overall = mean_overall.reshape(d,1)
    S_B += n* (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print('Between-class scatter matrix %sx%s' % (S_B.shape[0], S_B.shape[1]))

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs =sorted(eigen_pairs, key=lambda k:k[0], reverse = True)
print('Eigenvalues un decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

#為了衡量多少的 類別判別資訊  被線性判別式(特徵向量) 所描述  繪製解釋變異數徒圖形
tot = sum (eigen_vals.real)
discr = [(i/tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1,14), discr, alpha=0.5, align='center', label='cumlative "discriminability"')
plt.step(range(1,14), cum_discr, where='mid', label='cumlative "discriminability"')

plt.ylabel('"discriminability ratio"')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()


#將最具判別力的兩個 特徵向量疊起來，建立轉換矩陣w
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)
#將資料投影到新的特徵子空間
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0] * (-1),
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#使用sklearn的LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

#用羅吉斯回歸模型建立決策邊界 (訓練資料)
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, clf=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

#用羅吉斯回歸模型建立決策邊界 (測試資料)
X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, clf=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

#Kernel skill  利用核函數將非線性可分的資料轉換到新的維度空間中 使其成為可以線性分離的

#半月形的dataset
X,y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()

#將半月形dataset用標準PCA呈現   (無法使用線性分類器有效分類)
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))


ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
plt.show()

#使用 RBF KERNEL PCA 將半月型dataset轉換為線性可分
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

plt.tight_layout()
plt.show()

#同心圓dataset
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)

plt.tight_layout()
plt.show()

#同心圓dataset用標準PCA呈現   (無法使用線性分類器有效分類)
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
plt.show()

#使用 RBF KERNEL PCA 將同心圓dataset轉換為線性可分
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
plt.show()

#將新資料投影到轉換的維度上

#半月形dataset
X, y = make_moons(n_samples=100, random_state=123)
#回傳核矩陣的特徵值
alphas, lambdas = rbf_kernel_pca_project(X, gamma=15, n_components=1)
x_new = X[25]
print(x_new)
x_proj = alphas[25] # original projection
print(x_proj )

#定義投影的函數
def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

# projection of the "new" datapoint
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
print(x_reproj)

#顯示投影 (原投影的點 跟新投影的點重合)
plt.scatter(alphas[y == 0, 0], np.zeros((50)), color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y == 1, 0], np.zeros((50)), color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green', label='remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)

plt.tight_layout()
plt.show()

#利用 sklearn的PCA 產生一樣的結果
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1], color='blue', marker='o', alpha=0.5)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()