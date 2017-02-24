#引入函式庫
from sklearn.svm import SVR
import math
import numpy as np
import matplotlib.pyplot as plt


clf = SVR(kernel="rbf",C=10)


#由list轉為np array
# feature_np=np.array(feature,dtype=np.int).reshape(len(feature),1) 
# target_np=np.array(target,dtype=np.float)

feature_np=np.sort(np.random.random((20, 1)), axis=0)

target=[]
for i in feature_np:
    target.append(math.sin(i*math.pi*2))
target_np=np.array(target,dtype=np.float)

clf.fit(feature_np,target_np)


print("f(0):",clf.predict([[0]]),"y:",math.sin(0),"margin:",clf.decision_function([[0]]))
print("f(1):",clf.predict([[1]]),"y:",math.sin(1*math.pi*2),"margin:",clf.decision_function([[1]]))

#取得預測線
y_predict=clf.predict(feature_np)

#print(y_predict)


plt.scatter(feature_np, target_np, color='darkorange', label='data')
plt.hold('on')
plt.plot(feature_np, y_predict, color='navy',lw=2,label='predict')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()



# f(0): [ -3.91353616e-15] y: 0.0
# f(1): [ -3.91353616e-15] y: -2.4492935982947064e-16

#onlineSVR
# f(0)=0.20864     y(0)=0     margin=0.20864
# f(1)=-0.20058     y(1)=-2.4493e-016     margin=-0.20058