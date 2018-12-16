import sim


o1 = [3.85, 4.51, 4.39, 4.08, 3.97, 2.18, 3.29, 5.48]
o2 = [3.85, 2.19, 3.46, 3.66, 3.93, 3.15, 3.47, 4.11]
o3 = [4.51, 2.19, 3.70, 4.30, 4.83, 3.86, 4.48, 4.19]
o4 = [4.39, 3.46, 3.70, 1.21, 3.09, 4.12, 3.22, 3.72]
o5 = [4.08, 3.66, 4.30, 1.21, 2.62, 4.30, 2.99, 4.32]
o6 = [3.97, 3.93, 4.83, 3.09, 2.62, 4.15, 1.29, 3.38]
o7 = [2.18, 3.15, 3.86, 4.12, 4.30, 4.15, 3.16, 4.33]
o8 = [3.29, 3.47, 4.48, 3.22, 2.99, 1.29, 3.16, 3.26]
o9 = [5.48, 4.11, 4.19, 3.72, 4.32, 3.38, 4.33, 3.26]

# ard
ard = sim.knn_density(o1, 2) / ((sim.knn_density(o7, 2) + sim.knn_density(o8, 2)) / 2)
print('ard:', ard)


P1 = [1, 0, 1, 0, 1, 0]
P2 = [1, 1, 1, 1, 1, 0]
P3 = [1, 1, 0, 0, 1, 0]
P4 = [1, 1, 1, 0, 0, 1]
P5 = [1, 0, 1, 0, 0, 1]
P6 = [0, 0, 1, 1, 0, 1]
P7 = [1, 1, 1, 1, 1, 1]
P8 = [0, 0, 1, 1, 1, 1]
P9 = [0, 1, 0, 1, 0, 1]

p_list = [P1, P2, P3, P4, P5, P6, P7, P8, P9]

# support
set1 = [[0], [2], [5]]
set2 = [[0], [1], [2], [0, 2], [3], [4], [5], [2, 5]]
set3 = [[0], [1], [0, 1], [2], [0, 2], [3], [2, 3], [4], [0, 4], [2, 4], [5], [2, 5], [3, 5]]
set4 = ([[0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2], [3], [1, 3], [2, 3], [4], [0, 4], [1, 4], [0, 1, 4],
        [2, 4], [0, 2, 4], [3, 4], [2, 3, 4], [5], [0, 5], [1, 5], [2, 5], [0, 2, 5], [3, 5], [2, 3, 5]])

print('A:', sim.support_mult(p_list, set1))
print('B:', sim.support_mult(p_list, set2))
print('C:', sim.support_mult(p_list, set3))
print('D:', sim.support_mult(p_list, set4))

# naive bayes
class_list = [0, 0, 0, 1, 1, 1, 2, 2, 2]
indx_list = [0, 1]
equal_to = [1, 0]

p = sim.naive_bayes(p_list, class_list, 2, indx_list, equal_to, 3)
print('p:', p)

# similarity measures
smc = sim.smc(P2, P6)
j = sim.j_coeff(P2, P7)
cos6 = sim.cos(P2, P6)
cos7 = sim.cos(P2, P7)

print('A:', smc > j)
print('B:', smc > cos6)
print('C:', cos7 > j)
print('D:', cos6 > cos7)

# confidence
# X --> Y
# a_left = X, a_right = Y
a_left = [1, 3, 5]
a_right = [0, 4]

# lift, no script for this but very easy to calculate
# with the given conf and support script
conf = sim.conf(p_list, a_left, a_right)
supp = sim.support(p_list, a_right)
print('Lift:', conf / supp)

# kmeans where you are given a list of centroids
# and are asked which of these centroid lists are correct
X = [1, 3, 4, 6, 7, 8, 13, 15]

c1 = sim.k_means_1d_no_init([[1, 3, 4], [6, 7, 8], [13, 15]])
c2 = sim.k_means_1d_no_init([[1], [3, 4, 6], [7, 8], [13, 15]])
c3 = sim.k_means_1d_no_init([[1, 3, 4], [6, 7], [8, 13, 15]])
c4 = sim.k_means_1d_no_init([[1, 3, 4], [6, 7], [8, 13], [15]])

print('A:', c1)
print('B:', c2)
print('C:', c3)
print('D:', c4)

# adaboost weights

# error lists contains one list as for this problem there is only 1 round.
# the one list elements corresponding to if the problem was classified wrongly.
# if x_i is classified wrongly e[i] = True and vice versa.
error_lists = [[False, True, True, False]]

ada = sim.ada_boost(error_lists)
print('ada:', ada)