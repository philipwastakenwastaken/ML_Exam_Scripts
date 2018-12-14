import sim


O1 = [4.39, 3.62, 4.53, 3.83, 4.36, 4.14, 4.06, 4.31]
O2 = [4.39, 1.43, 0.74, 1.85, 1.17, 1.01, 0.59, 2.33]
O3 = [3.62, 1.43, 1.94, 1.02, 0.83, 1.32, 1.38, 2.80]
O4 = [4.53, 0.74, 1.94, 2.26, 1.69, 1.15, 0.81, 2.66]
O5 = [3.83, 1.85, 1.02, 2.26, 1.27, 1.32, 1.67, 2.68]
O6 = [4.36, 1.17, 0.83, 1.69, 1.27, 1.22, 1.33, 2.91]
O7 = [4.14, 1.01, 1.32, 1.15, 1.32, 1.22, 0.62, 2.41]
O8 = [4.06, 0.59, 1.38, 0.81, 1.67, 1.33, 0.62, 2.17]
O9 = [4.31, 2.33, 2.80, 2.66, 2.68, 2.91, 2.41, 2.17]

x = sim.knn_density(O1, 1) / sim.knn_density(O3, 1)
#print("ard: = ", x)

y = sim.gauss_kernel_density(O1, 6, 1)
#print("kernel density:", y)

LIS1 = [1, 0, 0, 1, 1, 0]
LIS2 = [1, 0, 1, 0, 1, 0]
LIS3 = [0, 1, 0, 1, 0, 1]
LIS4 = [0, 1, 0, 1, 0, 1]
LIS5 = [1, 0, 1, 0, 0, 1]
LIS6 = [1, 0, 1, 0, 1, 0]
OPO1 = [1, 0, 1, 0, 0, 1]
OPO2 = [0, 1, 0, 1, 1, 0]
OPO3 = [0, 1, 1, 0, 0, 1]
OPO4 = [0, 1, 0, 1, 0, 1]
OPO5 = [0, 1, 1, 0, 0, 1]
OPO6 = [1, 0, 1, 0, 1, 0]


a = sim.j_coeff(LIS1, LIS2) < sim.j_coeff(LIS1, OPO1)

#print("A:", a)
#print("B:", sim.j_coeff(LIS1, OPO1) < sim.j_coeff(LIS1, OPO2))
#print("C:", sim.j_coeff(LIS1, OPO2) < sim.j_coeff(LIS1, OPO3))
#print("D:", sim.j_coeff(LIS1, OPO2) < sim.j_coeff(LIS1, OPO4))

c1 = [58, 39, 6, 147]
c2 = [95, 2, 2, 151]
c3 = [94, 3, 6, 147]
c4 = [87, 10, 10, 143]

print("A:", sim.prec(c1) > sim.prec(c4))
print("B:", sim.recall(c2) < sim.recall(c3))
print("C", sim.er(c3) > sim.er(c4))
