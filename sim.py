import math


def f00(a, b):
    count = 0
    for v1, v2 in zip(a, b):
        if v1 == 0 and v2 == 0:
            count += 1
    return count


def f11(a, b):
    count = 0
    for (v1, v2) in zip(a, b):
        if v1 == 1 and v2 == 1:
            count += 1
    return count


def euclid_norm(a):
    sum = 0
    for val in a:
        sum += val ** 2
    return sum ** (1 / 2)


def p_norm(a, p):
    sum = 0
    for val in a:
        sum += (abs(val)) ** p
    return sum ** (1 / p)


def j_coeff(a, b):
    return f11(a, b) / (len(a) - f00(a, b))


def smc(a, b):
    return (f00(a, b) + f11(a, b)) / len(a)


def cos(a, b):
    top = 0
    for v1, v2 in zip(a, b):
        top += v1 * v2
    return top / (p_norm(a, 2) * p_norm(b, 2))


def ej(a, b):
    top = 0
    for v1, v2 in zip(a, b):
        top += v1 * v2
    return top / ((p_norm(a, 2) ** 2) + (p_norm(a, 2) ** 2) - top)


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


def sim_test(a, b):
    dot_sum = 0
    for v1, v2 in zip(a, b):
        dot_sum += v1 * v2

    dot_sum = dot_sum ** 2
    top1 = (euclid_norm(a) ** 2) * (euclid_norm(b) ** 2)
    return math.sqrt((top1 - dot_sum) / top1)
