import math

# indices for classifier accuracy
TP = 0
FN = 1
FP = 2
TN = 3

# In K-fold cross-validation the data is
# split into K-datasplits where in each split 1/K is hold
# out for testing and (K-1)/K is used for training.

# for decision tree boundries
# p infinity norm = square
# p 2 norm = circle
# p 1 norm = diamond

# how many models are trained in two-layer cross validation?
# K1 * (K2 *  L + 1)
# K1 = outer cv
# 2 = inner cv
# L = antal parametre, man tester


# only works with 2 classes, same with f11
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


# diff is used for clustering..
def f00_diff(a, b):
    count = 0
    for o1 in a:
        for o2 in b:
            if o1 != o2:
                count += 1
    return count


def f11_diff(a, b):
    count = 0
    v_list = [a, b]
    for v in v_list:
        for indx, o1 in enumerate(v):
            for i in range(indx + 1, len(v)):
                if o1 == v[i]:
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


# jaccard coefficient for clustering
def j_clus(a, b):
    sum_len = len(a) + len(b)
    K = (sum_len * (sum_len - 1)) / 2
    return f11_diff(a, b) / (K - f00_diff(a, b))


# jaccard sim for two vectors
def j_coeff(a, b):
    M = len(a)
    return f11(a, b) / (M - f00(a, b))


# SMC
def smc(a, b):
    return (f00(a, b) + f11(a, b)) / len(a)


# cosine similarity
def cos(a, b):
    top = 0
    for v1, v2 in zip(a, b):
        top += v1 * v2
    return f11(a, b) / (p_norm(a, 2) * p_norm(b, 2))


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
    return round(count / len(x), 3)


# amount of variance explained for a single PCA component
# input: a list containing the diagonal of the S matrix
def pca_var(s_diag):
    output_list = []
    for var_test in s_diag:
        top = var_test ** 2
        sum = 0
        for var_denom in s_diag:
            sum += var_denom ** 2
        output_list.append(round(top / sum, 3))
    return output_list


# total variance explained by PCA1, PCA1 + PCA2, ..., PCA1 + ..., PCAN
# input: a list containing the diagonal of the S matrix
def pca_var_combined(s_diag):
    var_ex_list = pca_var(s_diag)
    sum = 0
    output_list = []
    for var_ex in var_ex_list:
        sum += var_ex
        output_list.append(sum)
    return output_list


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
        top += '     ' + str(indx + 1)
        output += str(indx + 1) + ' '
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


def prob_given_class(x, class_list, class_check, indx_check, equal):
    count = 0
    for indx, obs in enumerate(x):
        if class_list[indx] == class_check and obs[indx_check] == equal:
            count += 1
    return count / len([j for j in class_list if j == class_check])


# x: observation list. class_list:
# class of x[i] = class_list[i].
# naive bayes: p(y = c | x1, ..., xn)
# class_check: c
# indx_list: the indexes of x1 ... xn
# equal_to: what each index in indx_list should be equal to.
def naive_bayes(x, class_list, class_check, indx_list, equal_to, C):
    top = 1
    for indx, equal in zip(indx_list, equal_to):
        top *= prob_given_class(x, class_list, class_check, indx, equal)
    top *= prob_class(class_list, class_check)

    bot_sum = 0
    for cl in range(0, C):
        bot = 1
        for indx, equal in zip(indx_list, equal_to):
            bot *= prob_given_class(x, class_list, cl, indx, equal)
        bot *= prob_class(class_list, cl)
        bot_sum += bot
    return top / bot_sum


def bayes(x, class_list, class_check, indx_list, equal_to, C):
    top = prob_given_many_class(x, class_list, class_check, indx_list, equal_to)
    top *= prob_class(class_list, class_check)

    bot_sum = 0
    for cl in range(0, C):
        bot = prob_given_many_class(x, class_list, cl, indx_list, equal_to)
        bot *= prob_class(class_list, cl)
        bot_sum += bot
    print(top)
    print(bot)
    return top / bot


def prob_given_many_class(x, class_list, class_check, indx_list, equal_to):
    count = 0
    for indx, obs in enumerate(x):
        sub_count = 0
        if class_list[indx] == class_check:

            for index, equal in zip(indx_list, equal_to):
                if obs[index] == equal:
                    sub_count += 1
            if sub_count == len(indx_list):
                count += 1
    return count / len([j for j in class_list if j == class_check])


# ada boost algorithm assuming equal probability of initial weights
# error_lists: list of lists of booleans, one list for each round of ada boost.
# each list contains elements that are True is they were misclassfied, and
# false otherwise.
def ada_boost(error_lists):
    w_list = [1 / len(error_lists[0])] * len(error_lists[0])

    for error_list in error_lists:
        e = 0
        new_weights = []

        # calc epsilon
        for w, error in zip(w_list, error_list):
            delta = 1
            if error:
                delta = 0
            e += w * (1 - delta)

        alpha = 0.5 * math.log((1 - e) / e)

        for w, error in zip(w_list, error_list):
            top_alpha = -alpha
            if error:
                top_alpha = alpha
            top_w = w * math.exp(top_alpha)
            bot_w_sum = 0

            for w, error in zip(w_list, error_list):
                top_alpha = -alpha
                if error:
                    top_alpha = alpha
                bot_w_sum += w * math.exp(top_alpha)
            new_weights.append(top_w / bot_w_sum)
        w_list = new_weights
    return [round(z, 4) for z in w_list]


# use this if you are NOT given a set of initial centroids.
def k_means_1d_no_init(cluster_list):
    # calc mean of clusters
    mean_list = []
    for cl in cluster_list:
        sum = 0
        for o in cl:
            sum += o
        mean_list.append(sum / len(cl))

    # find mismatches
    for indx, cl in enumerate(cluster_list):

        for o in cl:

            for mean in mean_list:

                if abs(o - mean) < abs(o - mean_list[indx]):
                    return False
    return True


# use this if you are given a set of initial centroids
def k_means_1d_init(obs_list, init_list):
    mean_list = []
    # init mean list with intial centroids
    for init in init_list:
        mean_list.append(init)
    cond = True

    while (cond):
        old_mean_list = mean_list
        centroid_list = []
        for i in range(0, len(init_list)):
            centroid_list.append([])
        # assign each point to each nearest cluster
        for obs in obs_list:

            # calc distance to each cluster
            dist_min = 1000000
            indx = 0
            for cl, mean in enumerate(mean_list):
                if abs(obs - mean) < dist_min:
                    dist_min = abs(obs - mean)
                    indx = cl
            centroid_list[indx].append(obs)

        # calculate new means
        mean_list = []
        for centroid in centroid_list:
            sum = 0
            for obs in centroid:
                sum += obs
            mean_list.append(sum / len(centroid))

        # check if centroids changed
        if mean_list == old_mean_list:
            cond = False
    return centroid_list


# knn algorithm
# ties are decided by looking at the class of nearest element
# contained in the tied classes list.
# USE D_LIST WITH ZEROES INCLUDED
# classes is values of classes
def knn(d_list, class_list, k, classes):
    errors = 0
    n_class = [0] * len(classes)

    for obs_num, obs in enumerate(d_list):
        sorted_obs = obs
        sorted_obs.sort()
        smallest_vals = []
        # find k smallest values
        for i in range(1, k + 1):
            smallest_vals.append(sorted_obs[i])
        indx_list = []
        # find the indxs of the smallest values
        for val in smallest_vals:
            indx_list.append(obs.index(val))
        
        # count the neighbours classes
        for indx in indx_list:
            for cl_index, cl in enumerate(classes):
                if class_list[indx] == cl:
                    n_class[cl_index] = n_class[cl_index] + 1
        
        # handle ties by choosing the class which has the closest element
        tie = False
        tie_list = []
        if n_class.count(max(n_class) > 1):
            tie = True
            # find which classes that are tied
            max_ele = max(n_class)
            for indx, cl in enumerate(n_class):
                if max_ele == cl:
                    tie_list.append(indx)
        if tie:
            nearest_index = find_min_class(obs, class_list, tie_list)
            assigned_class = class_list[nearest_index]
            if class_list[obs_num] != assigned_class:
                errors += 1
        else:
            assigned_class = n_class.index(max(n_class))
            if class_list[obs_num] != assigned_class:
                errors += 1
    return errors / len(d_list)
        

def find_min_class(obs, class_list, classes):
    min_ele = 1000000
    min_indx = 0

    for indx, dist in enumerate(obs):
        if dist < min_ele and dist != 0 and class_list[indx] in classes:
            min_ele = dist
            min_indx = indx
    return min_indx


def dec_tree_ce(root, splits):
    # impurity of root
    n_root = sum(root)
    max_class = max(root)
    imp_root = 1 - (max_class / n_root)

    # impurity of splits
    imp_list = []
    for split in splits:
        max_class = max(split)
        impurity = 1 - (max_class / sum(split))
        imp_list.append(impurity)

    # purity gain
    p_sum = 0
    for split, imp in zip(splits, imp_list):
        frac = sum(split) / n_root
        p_sum += frac * imp
    return imp_root - p_sum


def dec_tree_gini(root, splits):
    # impurity of root
    imp_root = gini_imp(root)
    n_root = sum(root)

    # impurity of splits
    imp_list = []
    for split in splits:
        impurity = gini_imp(split)
        imp_list.append(impurity)

    # purity gain
    p_sum = 0
    for split, imp in zip(splits, imp_list):
        frac = sum(split) / n_root
        p_sum += frac * imp
    return imp_root - p_sum


def gini_imp(split):
    n = sum(split)
    sum1 = 0
    for s in split:
        sum1 += (s / n) ** 2
    return 1 - sum1
