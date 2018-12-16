import sim

vars = [10.2, 6.1, 2.8, 2.2, 1.6]

print('ind var:', sim.pca_var(vars))
print('comb var:', sim.pca_var_combined(vars))

# adaboost
error = [True] * 2 + [False] * 30


print('ada:', sim.ada_boost([error]))

root = [17, 15]
split1 = [[13, 2], [4, 13]]
split2 = [[15, 12], [2, 3]]
split3 = [[13, 2], [2, 10], [2, 3]]

print('split1:', sim.dec_tree_ce(root, split1))
print('split2:', sim.dec_tree_ce(root, split2))
print('split3:', sim.dec_tree_ce(root, split3))


# k means

X = [42, 38.3, 40.1, 34.2, 50.9, 30.3, 68.6, 19.4]
k = 2
centroids = [30.3, 19.4]
print('kmeans:', sim.k_means_1d_init(X, centroids))

x1 = [1, 1, 1, 0, 1]
x2 = [1, 0, 0]

print('j:', sim.j_clus(x1, x2))

O1 = [1, 0, 1, 0, 0, 1]
O2 = [1, 0, 1, 0, 0, 1]
O3 = [1, 0, 1, 0, 0, 1]
O4 = [1, 0, 1, 0, 1, 0]
O5 = [0, 1, 0, 1, 1, 0]
O6 = [1, 0, 0, 1, 1, 0]
O7 = [0, 1, 0, 1, 1, 0]
O8 = [1, 0, 1, 0, 1, 0]
o_list = [O1, O2, O3, O4, O5, O6, O7, O8]

print('j:', sim.j_coeff(O1, O4))

w1 = [-4, 1, 0.01, 1, -1, -1]
w2 = [-10, 1, -0.02, 1, 1, 1]
x = [6, 120, 3.2, 0, 4]

print('sum w1:', sim.dot_product([1] + x, w1))
print('sum w2:', sim.dot_product([1] + x, w2))

hpL = 0
hpH = 1
wtL = 2
wtH = 3
am0 = 4
am1 = 5

A = [[hpL], [wtL], [wtH], [am0], [am1]]
B = [[hpL], [wtL], [wtH], [am0], [am1], [hpL, wtL], [hpL, am0], [hpL, am1], [wtL, am1], [wtH, am0]]
C = [[hpL], [wtL], [wtH], [am0], [am1], [hpL, wtL], [hpL, am0], [hpL, am1], [wtL, am1], [wtH, am0], [hpL, wtL, am1]]
D = [[hpL], [wtL], [wtH], [am0], [am1], [hpL, wtL], [hpL, am0], [hpL, am1], [wtL, am1], [wtH, am0], [hpL, wtL, am1], [hpL, wtL, am0]]

a_left = [wtH, am0]
a_right = [hpH]

print('conf:', sim.conf(o_list, a_left, a_right))


print('A:', sim.support_mult(o_list, A))
print('B:', sim.support_mult(o_list, B))
print('C:', sim.support_mult(o_list, C))
print('D:', sim.support_mult(o_list, D))

a = [1, 0, 1, 0, 0, 1]
b = [1, 0, 1, 0, 1, 0]
c = [0, 0, 0, 0, -1, 1]
print('A:', sim.euclid_norm(c))
print('B:', sim.p_norm(c, 1) < sim.euclid_norm(c))
print('C:', sim.j_coeff(a, b))
print('D:', sim.cos(a, b) == sim.smc(a, b))

# high: z = 0
# low: z = 1

class_list = [1, 1, 1, 1, 0, 0, 0, 1]
C = 2
indx_list = [hpL, am0]
class_check = 0
p = sim.naive_bayes(o_list, class_list, class_check, indx_list, [1, 1], C)
print('p:', p)

p1 = [0, 0.2606, 1.1873, 2.4946, 2.9510, 2.5682, 3.4535, 2.4698]
p2 = [0.2606, 0, 1.2796, 2.4442, 2.8878, 2.4932, 3.3895, 2.4216]
p3 = [1.1873, 1.2796, 0, 2.8294, 3.6892, 2.9147, 4.1733, 2.2386]
p4 = [2.4946, 2.4442, 2.8294, 0, 1.4852, 0.2608, 2.2941, 1.8926]
p5 = [2.9510, 2.8878, 3.6892, 1.4852, 0, 1.5155, 1.0296, 3.1040]
p6 = [2.5682, 2.4932, 2.9147, 0.2608, 1.5155, 0, 2.3316, 1.8870]
p7 = [3.4535, 3.3895, 4.1733, 2.2941, 1.0296, 2.3316, 0, 3.7588]
p8 = [2.4698, 2.4216, 2.2386, 1.8926, 3.1040, 1.8870, 3.7588, 0]
p_list = [p1, p2, p3, p4, p5, p6, p7, p8]
print('knn error:', sim.knn(p_list, class_list, 1, [0, 1]))

root2 = [55 + 75, 44 + 59, 88 + 74]
split1 = [55, 44, 88]
split2 = [75, 59, 74]
splits = [split1, split2]
print('imp:', sim.dec_tree_gini(root2, splits))