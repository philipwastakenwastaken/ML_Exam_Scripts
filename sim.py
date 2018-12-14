import math

# indices for classifier accuracy
TP = 0
FN = 1
FP = 2
TN = 3

# In K-fold cross-validation the data is
# split into K-datasplits where in each split 1/K is hold
# out for testing and (K-1)/K is used for training.

# p infinity norm = square
# p 2 norm = circle
# p 1 norm = diamond


# computes f00
def f00(a, b):
    count = 0
    for v1, v2 in zip(a, b):
        if v1 == 0 and v2 == 0:
            count += 1
    return count


# computes f11
def f11(a, b):
    count = 0
    for (v1, v2) in zip(a, b):
        if v1 == 1 and v2 == 1:
            count += 1
    return count


# euclidian norm for a vector.
def euclid_norm(a):
    sum = 0
    for val in a:
        sum += val ** 2
    return sum ** (1 / 2)


# p norm for a vector
def p_norm(a, p):
    sum = 0
    for val in a:
        sum += (abs(val)) ** p
    return sum ** (1 / p)


# jaccard coefficient for two vectors
def j_coeff(a, b):
    return f11(a, b) / (len(a) - f00(a, b))


# SMC
def smc(a, b):
    return (f00(a, b) + f11(a, b)) / len(a)


# cosine similarity
def cos(a, b):
    top = 0
    for v1, v2 in zip(a, b):
        top += v1 * v2
    return top / (p_norm(a, 2) * p_norm(b, 2))


# EJ
def ej(a, b):
    top = 0
    for v1, v2 in zip(a, b):
        top += v1 * v2
    return top / ((p_norm(a, 2) ** 2) + (p_norm(a, 2) ** 2) - top)


# requires list of distances
def knn_density(x, k):
    average = 0
    x.sort()
    for i in range(0, k):
        average += x[i]
    return 1.0 / (average / float(k))


def dot_product(a, b):
    sum = 0
    for v1, v2 in zip(a, b):
        sum += v1 * v2
    return sum


# requires list of distances
def gauss_kernel_density(x, M, var):
    sum = 0
    var_frac_top = -(1 / (2 * (var ** 2)))
    var_frac_bot = math.sqrt((2 * math.pi * (var ** 2)) ** M)
    for val in x:
        sum += math.exp(var_frac_top * (val ** 2)) / var_frac_bot
    return sum / len(x)


# support for item set. x = list of lists with binarized data
# indx_list = indexes of the item sets which are to be calculated
def support(x, indx_list):
    count = 0
    for obs in x:
        inner_count = 0
        for indx in indx_list:
            if obs[indx] == 1:
                inner_count += 1
        if inner_count == len(indx_list):
            count += 1
    return count / len(x)


# calculates support for each item set included in indx_list
def support_mult(x, indx_list):
    supp_list = []
    for indx in indx_list:
        supp_list.append(support(x, indx))
    return supp_list


# confidence of association rule.
# x = list of lists with binarized data
# association rule = X --> Y
# a_left = X, a_right = Y
def conf(x, a_left, a_right):
    top = support(x, a_left + a_right)
    bot = support(x, a_left)
    return top / bot


# gmm probability, 1d.
# x = observation to be predicted, wlist = list of weights, mlist = list of mu
# vlist = list of var, K = the cluster x should be predicted in
def gmm_1d(x, wlist, mlist, vlist, K):
    sum = 0
    pre = 0
    ls = []
    for w, mu, var in zip(wlist, mlist, vlist):
        frac = 1 / math.sqrt(2 * math.pi * var)
        ex = math.exp(-1 / (2 * var) * ((x - mu) ** 2))
        pre = w * frac * ex
        sum += pre
        ls.append(pre)
    return ls[K - 1] / sum


# for assignment with scalar invariant etc..
def sim_test(a, b):
    dot_sum = 0
    for v1, v2 in zip(a, b):
        dot_sum += v1 * v2

    dot_sum = dot_sum ** 2
    top1 = (euclid_norm(a) ** 2) * (euclid_norm(b) ** 2)
    return math.sqrt((top1 - dot_sum) / top1)


# creates a 2d list containing the euclidean distances to each point
def euclid_table_1d(x, inc_zero):
    output = []
    for calc_x in x:
        temp = []
        for val in x:
            el = abs(calc_x - val)
            if inc_zero:
                temp.append(el)
            elif el != 0:
                temp.append(el)
        output.append(temp)
    return output


def print_euclid_1d(x, inc_zero):
    y = euclid_table_1d(x, inc_zero)
    output = ''
    top = '  '
    for indx, v1 in enumerate(y):
        top += '     ' + str(indx)
        output += str(indx) + ' '
        for v2 in v1:
            output += '   ' + str(round(abs(v2), 2))
        output += '\n'
    top += '\n'
    output = top + output
    print(output)


# ANN model with rectified linear function.
# x = obs, h_units = list of h. units, w0 = weight 0.
def ann_rect(x, h_units, w0):
    sum = 0
    len_diff = len(h_units[0]) - len(x)
    w_dot = []
    if (len_diff > 0):
        w_dot = [1] * len_diff
    for indx, w in enumerate(h_units):
        sum += w0[indx + 1] * max(dot_product(w_dot + x, w), 0)
    return w0[0] + sum


# Evaluating classifier
# each function takes a list consisting of [TP, FN, FP, TN]
def prec(x):
    return x[0] / (x[0] + x[2])


def recall(x):
    return x[0] / (x[0] + x[1])


def acc(x):
    return (x[0] + x[3]) / sum(x)


def er(x):
    return 1 - acc(x)


def prob_class(class_list, class_check):
    count = 0
    for cl in class_list:
        if cl == class_check:
            count += 1
    return count / len(class_list)


def prob_given_class(x, class_list, class_check, indx_check):
    count = 0
    for indx, obs in enumerate(x):
        if class_list[indx] == class_check and obs[indx_check] == 1:
            count += 1
    return count / len([j for j in class_list if j == class_check])


# x: observation list. class_list: 
# class of x[i] = class_list[i]. 
# naive bayes i p(y = c | x1, ..., xn)
# class_check: c
# indx_list: the indexes of x1 ... xn
def naive_bayes(x, class_list, class_check, indx_list, C):
    top = 1
    for indx in indx_list:
        top *= prob_given_class(x, class_list, class_check, indx)
    top *= prob_class(class_list, class_check)

    bot_sum = 0
    for cl in range(0, C):
        bot = 1
        for indx in indx_list:
            bot *= prob_given_class(x, class_list, cl, indx)
        bot *= prob_class(class_list, cl)
        bot_sum += bot
    return top / bot_sum


# ada boost algorithm assuming equal probability of initial weights
def ada_boost(x, rounds, alpha_e_list, class_list):
    w_list = [1 / len(x)] * len(x)
    
    for error in alpha_e_list:
        alpha = 0.5 * math.log2((1 - error) / error)




