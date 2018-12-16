import sim

root = [3, 1, 3]
splits = [[1, 1, 0], [2, 0, 3]]

print('dec:', sim.dec_tree_ce(root, splits))

o1 = [4, 7, 9, 5, 5, 5, 6]
o2 = [4, 7, 7, 7, 3, 7, 8]
o3 = [7, 7, 10, 6, 6, 4, 9]
o4 = [9, 7, 10, 8, 6, 10, 9]
o5 = [5, 7, 6, 8, 8, 6, 7]
o6 = [5, 3, 6, 6, 8, 8, 11]
o7 = [5, 7, 4, 10, 6, 8, 7]
o8 = [6, 8, 9, 9, 7, 11, 7]
o_list = [o1, o2, o3, o4, o5, o6, o7, o8]

print('A:', sim.cos(o1, o3))
print('B:', sim.j_coeff(o1, o3))
print('C:', sim.smc(o1, o3))

class_list = [1, 1, 1, 1, 0, 0, 0, 0]
print('knn:', sim.knn(o_list, class_list, 1, [0, 1]))

ard = sim.knn_density(o1, 1) / sim.knn_density(o2, 1)
print('ard:', ard)

s1 = [0, 1, 1, 0, 1, 0]
s2 = [0, 1, 1, 1, 0, 1]
s3 = [1, 1, 1, 0, 1, 0]
s4 = [1, 1, 1, 0, 1, 0]
s5 = [0, 1, 1, 0, 1, 1]
s6 = [0, 0, 1, 1, 1, 1]
s7 = [1, 1, 0, 1, 1, 1]
s8 = [1, 1, 1, 0, 0, 0]
s9 = [1, 0, 1, 1, 0, 0]
s10 = [1, 1, 1, 0, 0, 1]
s_list = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]

class_list2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
indx_list = [0, 1]
equal_to = [0, 1]
C = 2

p = sim.naive_bayes(s_list, class_list2, 0, indx_list, equal_to, C)
print('p:', p)

f1 = 0
f2 = 1
f3 = 2
f4 = 3
f5 = 4
f6 = 5

A = [[f2], [f3], [f2, f3]]
B = [[f1], [f2], [f3], [f2, f3], [f5]]
C = [[f1], [f2], [f1, f2], [f3], [f1, f3], [f2, f3], [f5], [f2, f5], [f3, f5], [f6]]
D = [[f1], [f2], [f1, f2], [f3], [f1, f3], [f2, f3], [f1, f2, f3], [f4], [f5], [f2, f5], [f3, f5], [f2, f3, f5], [f6], [f2, f6], [f3, f6]]

print('A:', sim.support_mult(s_list, A))
print('B:', sim.support_mult(s_list, B))
print('C:', sim.support_mult(s_list, C))
print('D:', sim.support_mult(s_list, D))

a_left = [f1, f6]
a_right = [f4, f5]

print('conf:', sim.conf(s_list, a_left, a_right))

X = [3, 6, 7, 9, 10, 11, 14]
init = [4, 7, 14]

print('kmeans:', sim.k_means_1d_init(X, init))

error_lists = [[False, True, True, True]]

print('ada weights:', sim.ada_boost(error_lists))