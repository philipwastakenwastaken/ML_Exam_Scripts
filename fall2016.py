import sim


# question 6
acc = (31 + 30 + 29) / (210)
print(acc)

O1 = [0.534, 1.257, 1.671, 1.090, 1.315, 1.484, 1.253, 1.418]
O2 = [0.534, 0.727, 2.119, 1.526, 1.689, 1.214, 0.997, 1.056]
O3 = [1.257, 0.727, 2.809, 2.220, 2.342, 1.088, 0.965, 0.807]
O4 = [1.671, 2.119, 2.809, 0.601, 0.540, 3.135, 2.908, 3.087]
O5 = [1.090, 1.526, 2.220, 0.601, 0.331, 2.563, 2.338, 2.500]
O6 = [1.315, 1.689, 2.342, 0.540, 0.331, 2.797, 2.567, 2.708]
O7 = [1.484, 1.214, 1.088, 3.135, 2.563, 2.797, 0.275, 0.298]
O8 = [1.253, 0.997, 0.965, 2.908, 2.338, 2.567, 0.275, 0.343]
O9 = [1.418, 1.056, 0.807, 3.087, 2.500, 2.708, 0.298, 0.343]


o_list = [O1, O2, O3, O4, O5, O6, O7, O8, O9]

# question 11, knn density + ard

ard = sim.knn_density(O4, 1) / sim.knn_density(O6, 1)
print('ard:', ard)

# k means with given init centroids
obs = [-0.4, 0.0, 0.6, -2.1, -1.5, -1.7, 1.1, 0.8, 1.0]
obs1 = -0.4
obs2 = 0.0
obs3 = 0.6
obs4 = -2.1
obs5 = -1.5
obs6 = -1.7
obs7 = 1.1
obs8 = 0.8
obs9 = 1.0
sim.print_euclid_1d(obs, True)

y = sim.euclid_table_1d(obs, False)

init_list = [3, 5, 4]
k = 3

print('centroids:', sim.k_means_1d_init(obs, init_list))


P1 = [1, 1, 1, 1, 0, 1]
P2 = [0, 0, 0, 0, 0, 0]
P3 = [1, 1, 0, 1, 0, 0]
P4 = [0, 1, 1, 0, 1, 0]
P5 = [1, 1, 1, 1, 1, 1]
P6 = [0, 0, 0, 0, 0, 0]
P7 = [1, 1, 0, 1, 0, 0]
P8 = [0, 1, 1, 0, 1, 0]
P9 = [1, 1, 1, 1, 0, 1]
P10 = [0, 1, 1, 0, 1, 0]
P11 = [0, 0, 0, 0, 0, 0]
P12 = [1, 1, 0, 1, 0, 0]
P13 = [0, 1, 1, 0, 1, 0]
P14 = [0, 1, 1, 0, 1, 0]

p_list = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14]

set1 = [[0], [1], [2], [3], [4], [0, 1], [1, 2], [1, 3], [1, 4], [2, 4]]
set2 = [[0], [1], [2], [3], [4], [0, 1], [0, 3], [1, 2], [1, 3], [1, 4], [2, 4]]
set3 = [[0], [1], [2], [3], [4], [0, 1], [0, 3], [1, 2], [1, 3], [1, 4], [2, 4], [1, 2, 4]]
set4 = [[0], [1], [2], [3], [4], [0, 1], [0, 3], [1, 2], [1, 3], [1, 4], [2, 4], [0, 1, 3], [1, 2, 4]]


print('A:', sim.support_mult(p_list, set1))
print('B:', sim.support_mult(p_list, set2))
print('C:', sim.support_mult(p_list, set3))
print('D:', sim.support_mult(p_list, set4))

a_left = [0, 1, 2, 3, 4]
a_right = [5]
print('conf:', sim.conf(p_list, a_left, a_right))


indx_list = [0, 1]
class_list = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
equal_to = [1, 1]

p = sim.naive_bayes(p_list, class_list, 1, indx_list, equal_to, 2)
print('p:', p)


j = sim.j_coeff(P1, P3)
smc = sim.smc(P1, P3)
cos = sim.cos(P1, P3)

print('A:', j < smc)
print('B:', j > cos)
print('C:', smc > cos)
print('D:', cos == 3/15)

print('d', sim.p_norm([-0.25, -0.25], 1))


e = [True, True, True, True, True]
e1 = [False] * 20
print('ada:', sim.ada_boost([e + e1]))

