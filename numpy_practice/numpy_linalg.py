from numpy import *

#矩陣
A=mat([[1,2,4,5,7],[9,12,11,8,2],[6,4,3,2,1],[9,1,3,4,5],[0,2,3,4,1]])

print("行列式 det(A):",linalg.det(A))

print("反矩陣 inv(A):",linalg.inv(A))

print("對稱矩陣 :",A*A.T)

print("矩陣的秩 :",linalg.matrix_rank(A))

b = [1,0,1,0,1]
S = linalg.solve(A,b)

print("反矩陣求解 :",S)


#求向量的範數
A=[8,1,6]
#手動計算  (向量中各元素平方和開根號)
print("手動計算 :",sqrt(sum(power(A,2))))

print("函式呼叫 :",linalg.norm(A))



#矩陣線性轉換  特徵值與特徵向量
#A=[[8,1,6],[3,5,7],[4,9,2]]
A=[[5,4,2,1],[0,1,-1,-1],[-1,-1,3,0],[1,1,-1,2]]
evals, evecs =linalg.eig(A)
print("特徵值:",evals,"\n特徵向量:",evecs)


