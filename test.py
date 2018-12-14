import sim


O1 = [1, 0, 1, 0, 1, 0, 1, 0]
O2 = [1, 0, 1, 0, 1, 0, 1, 0]
O3 = [1, 0, 1, 0, 1, 0, 1, 0]
O4 = [1, 0, 1, 0, 1, 0, 0, 1]
O5 = [1, 0, 1, 0, 0, 1, 0, 1]
O6 = [1, 0, 0, 1, 0, 1, 1, 0]
O7 = [0, 1, 1, 0, 0, 1, 0, 1]
O8 = [0, 1, 1, 0, 1, 0, 0, 1]
O9 = [0, 1, 0, 1, 1, 0, 1, 0]
O10 = [0, 1, 0, 1, 0, 1, 1, 0]

o_list = [O1, O2, O3, O4, O5, O6, O7, O8, O9, O10]
indx_list = [[0], [1], [2], [4], [5], [6], [7]]
indx_list2 = [[0], [1], [2], [4], [5], [6], [7], [0, 2], [0, 4], [0, 6], [2, 4], [2, 7], [4, 6]]
indx_list3 = [[0], [1], [2], [4], [5], [6], [7], [0, 2], [0, 4], [0, 6], [2, 4], [2, 7], [4, 6], [0, 2, 4]]
indx_list4 = [[0], [1], [2], [4], [5], [6], [7], [0, 2], [0, 4], [0, 6], [2, 4], [2, 7], [4, 6], [0, 2, 4], [0, 2, 6]]

print("supp A:", sim.support_mult(o_list, indx_list))
print("supp B:", sim.support_mult(o_list, indx_list2))
print("supp C:", sim.support_mult(o_list, indx_list3))
print("supp D:", sim.support_mult(o_list, indx_list4))

a_left = [0, 2]
a_right = [4, 6]
print('conf:', sim.conf(o_list, a_left, a_right))
