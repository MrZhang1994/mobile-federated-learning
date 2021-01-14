
XI = 0.999
# initialize ETA
ETA = 0.03

## for cnn + FederatedMNIST
# set epsilon
EPSILON = 1.4
# set KAI
KAI = 300

import matplotlib.pyplot as plt
import numpy as np

rho = 5
beta = 65
delta = 8.171


A3 = KAI*(1-XI)/(2*beta)
B3 = ETA*beta+1
C3 = (rho*delta)/(beta*(EPSILON**2))
# 
# C3 = 1.1
print(A3)
print(B3)
print(C3)
x = np.linspace(0, 100, 50) # 从0到1，等分50分
y = A3*x
z = B3*(C3**x-1)
plt.figure() # 定义一个图像窗口
plt.plot(x, y) # 绘制曲线 y
plt.plot(x, z) # 绘制曲线 y
plt.show()

# def calculate_itr_method_2(rho, beta, delta):

#     A3 = KAI*(1-XI)/(2*beta)
#     B3 = ETA*beta+1
#     C3 = (rho*delta)/(beta*(EPSILON**2))

#     n_i = 0
#     f_last = 0
#     f = 0
#     while f<=f_last:
#         f_last = f
#         n_i = n_i+1
#         f = A3*n_i - B3*(C3**n_i-1)

#     itr = n_i-1
#     return itr


# itr = calculate_itr_method_2(4.321,65,8.171)
# print(itr)