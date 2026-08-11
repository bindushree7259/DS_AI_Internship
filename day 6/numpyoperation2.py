import numpy as np

a=np.array([[2,1,1],[2,1,3]])
b=np.array([[1,3,1],[1,4,1]])

print(np.vstack((a,b)))
print(np.hstack((a,b)))
print(np.concatenate((a,b),axis=0))
print(np.concatenate((a,b),axis=1))
