import numpy as np

t_d=np.array([5,5,5])
p_d=np.array([4,5,6])
print(np.sum((abs(t_d-p_d)/t_d) *100)//len(t_d))
