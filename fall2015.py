import sim

# variance explained by PCA components
s_diag = [27.9, 18.0, 15.8, 14.5, 11.1, 8.4]

print('var ex by each PCA:', sim.pca_var(s_diag))
print('var combined:', sim.pca_var_combined(s_diag))


# classifier accuracy (only for 2 class problems)

# contents of each index is available in sim.py
# other wise enter element in this order:
# first row: left to right
# second row: left to right
confuse_matrix = [75, 3, 4, 65]

print('A:', sim.acc(confuse_matrix), sim.er(confuse_matrix))
print('B:', sim.prec(confuse_matrix))


O1 = [0, 69, 55, 117, 50, 326, 36]
O2 = [69, 0, 36, 128, 104, 303, 85]
O3 = [55, 36, 0, 129, 94, 314, 78]
O4 = [117, 128, 129, 0, 85, 220, 91]
O5 = [50, 104, 94, 85, 0, 303, 23]
O6 = [326, 303, 314, 220, 303, 0, 307]
O7 = [36, 85, 78, 91, 23, 307, 0]

o_list = [O1, O2, O3, O4, O5, O6, O7]
class_list = [1, 1, 1, 1, 0, 0, 0]
classes = [0, 1]
k = 5

error_rate = sim.knn(o_list, class_list, k, classes)
print('error:', error_rate)

P1 = [0, 0, 0, 0, 1, 0, 0]
P2 = [0, 1, 1, 1, 0, 0, 1]
P3 = [0, 0, 0, 0, 0, 0, 0]
P4 = [0, 1, 0, 0, 1, 0, 1]
P5 = [0, 1, 1, 1, 1, 0, 0]
P6 = [1, 1, 1, 1, 1, 0, 0]
P7 = [1, 1, 1, 1, 1, 0, 1]
P8 = [0, 1, 1, 1, 1, 1, 1]
P9 = [0, 1, 0, 1, 0, 0, 0]
P10 = [1, 1, 0, 0, 0, 1, 0]
P11 = [0, 0, 0, 0, 1, 0, 0]
P12 = [0, 0, 0, 0, 0, 0, 0]
P13 = [0, 0, 0, 0, 0, 0, 0]
P14 = [0, 0, 0, 0, 0, 0, 0]
P15 = [0, 0, 0, 0, 0, 0, 0]
p_list = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15]

PC = 1
PCC = 2
HTN = 3
DM = 4
CAD = 5
RBC = 0

A = [[PC], [PCC], [HTN], [DM]]
B = [[PC], [PCC], [HTN], [DM], [PC, PCC], [PC, HTN], [PCC, HTN]]
C = [[PC], [PCC], [HTN], [DM], [PC, PCC], [PC, HTN], [PC, DM], [PCC, HTN]]
D = [[PC], [PCC], [HTN], [DM], [PC, PCC], [PC, HTN], [PC, DM], [PCC, HTN], [PC, PCC, HTN]]

print('A:', sim.support_mult(p_list, A))
print('B:', sim.support_mult(p_list, B))
print('C:', sim.support_mult(p_list, C))
print('D:', sim.support_mult(p_list, D))

supp = sim.support(p_list, [RBC, PC, CAD])
conf = sim.conf(p_list, [RBC, PC], [CAD])

print('supp:', supp)
print('conf:', conf)

# knn with custom distance measure
output_list = []
for val in p_list:
    output_list.append(round(1 / sim.smc(val, P5), 3))

print(output_list)

# naive bayes

class_list2 = ([1] * 9) + ([0] * 6)
class_check = 1
indx_list = [RBC, PC, DM, CAD]
equal_to = [1, 1, 1, 1]
C = 2

p = sim.naive_bayes(p_list, class_list2, class_check, indx_list, equal_to, C)
print('p:', p)


indx_list2 = [RBC, PC, DM]
equal_to2 = [1, 1, 1]
p2 = sim.bayes(p_list, class_list2, class_check, indx_list2, equal_to2, C)
print('p2:', p2)


# decision tree with classification error
pc0_list = [0, 0]
pc1_list = [0, 0]
for indx, obs in enumerate(p_list):
    if obs[PC] == 1 and class_list2[indx] == 0:
        pc1_list[0] = pc1_list[0] + 1
    if obs[PC] == 1 and class_list2[indx] == 1:
        pc1_list[1] = pc1_list[1] + 1
    if obs[PC] == 0 and class_list2[indx] == 1:
        pc0_list[1] = pc0_list[1] + 1
    if obs[PC] == 0 and class_list2[indx] == 0:
        pc0_list[0] = pc0_list[0] + 1
root = [9, 6]
print(pc0_list)
print(pc1_list)

print('dec:', sim.dec_tree_ce(root, [pc0_list, pc1_list]))

X = [2, 4, 8, 11, 15, 19, 20, 27, 30, 31]
centroids = [2, 4, 8]

print('kmeans', sim.k_means_1d_init(X, centroids))