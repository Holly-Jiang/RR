import os
import random
from typing import List
from random_sat_generator import CNF_Observing


def subsets(nums):
    res = [[], ]
    if not nums:
        return res
    for num in nums:
        res += [arr + [num] for arr in res]
    return res


def gcd(nums: List[int]) -> int:
    num1, num2 = min(nums), max(nums)
    while num1:
        num1, num2 = num2 % num1, num1
    return num2


'''
    variables:[]
    list: operator []
    eqlist: eq operator
    degree: the maximum degree
    coes: the maximun coefficent 
'''


def calculator(variables, list, eqlist, degree, maxcoe_from, maxcoe_to, max_cons):
    constant = random.randint(maxcoe_from, maxcoe_to)  # 随机产生一个maximum以内整数
    exp = []
    order = random.sample(range(0, len(variables)), len(variables))
    coelist = []
    for i in range(len(order)):  # 循环n次，每次产生一个新问题
        coe = random.randint(maxcoe_from, maxcoe_to)
        coelist.append(coe)
    coelist.append(constant)
    convention = gcd(coelist) if len(order) > 1 else 1
    for i in range(len(order)):  # 循环n次，每次产生一个新问题
        deg = random.randint(1, degree)  # 随机产生一个maximum以内整数
        if deg == 1 or len(variables[order[i]]) > 2:
            exp.append(f'{int(coelist[i] / convention)}*{variables[order[i]]}')
        else:
            exp.append(f'{int(coelist[i] / convention)}*{variables[order[i]]}**{deg}')
    observing = ''
    for i in range(len(exp)):
        if i > 0:
            sign = random.choice(list)
            observing += f'{sign}{exp[i]}'
        else:
            observing += f'{exp[i]}'

    sign = random.choice(list)
    if constant>0:
        observing += f'{sign}{constant/convention}'
    eq = random.choice(eqlist)
    observing += f'{eq}0'
    # w.write('%s\n' % observing)
    # w.flush()
    return observing


def degree_product(degree, vars):
    res = []
    if degree > 1 and len(vars) > 1:
        degvlist = []
        varset = subsets(vars)
        for vl in varset:
            if len(vl) == degree:
                vv = ''
                for i in vl:
                    if vv == '':
                        vv += i
                    else:
                        vv += f'*{i}'
                degvlist.append(vv)
        degv_count = random.sample(range(0, len(degvlist)), 1)
        if degv_count[0] > 0:
            degv_index = random.sample(range(0, len(degvlist)), degv_count[0])
            for i in degv_index:
                res.append(degvlist[i])
    return res


def random_expression(k, vars, clause, degree, maxcoe_from, maxcoe_to, constant):
    list = ['+', '-']
    eqlist = [' > ', ' < ', ' >= ', ' <= ']
    # path = './exp.txt'
    # w = open(path, 'w')
    varlist = ['x1', 'x2', 'x3', 'x4']
    sets = subsets(varlist)
    exp = []
    for v in sets:
        if len(v) > 0:
            v.extend(degree_product(degree, varlist))
            exp.append(calculator(v, list, eqlist, degree, maxcoe_from, maxcoe_to, constant))
    # w.close()
    obs = random.sample(range(0, len(exp)), vars)
    os.system(f'cnfgen randkcnf {k} {vars} {clause} > cnf.txt')
    observing = []
    for i in range(len(obs)):
        observing.append(exp[obs[i]])
    print('observing expression:\n', observing)
    cnf, cnf_exp, cnf_exp_list = CNF_Observing('./cnf.txt', observing)
    print('CNF:\n', cnf)
    print('final:\n', cnf_exp)
    return observing, cnf_exp,cnf_exp_list


if __name__ == '__main__':
    list = ['+', '-']
    eqlist = ['>', '<', '>=', '<=', '==', '!=']
    path = './exp.txt'
    w = open(path, 'w')
    varlist = ['x1', 'x2', 'x3', 'x4']
    sets = subsets(varlist)
    exp = []
    k = 3
    vars = 3
    clause = 4
    degree = 1
    maxcoe = 100
    for v in sets:
        if len(v) > 0:
            v.extend(degree_product(degree, varlist))
            exp.append(calculator(v, list, eqlist, degree, maxcoe, w))
    w.close()
    obs = random.sample(range(0, len(exp)), vars)
    os.system(f'cnfgen randkcnf {k} {vars} {clause} > cnf.txt')
    observing = []
    for i in range(len(obs)):
        observing.append(exp[obs[i]])
    print('observing expression:\n', observing)
    cnf, cnf_exp = CNF_Observing('./cnf.txt', observing)
    print('CNF:\n', cnf)
    print('final:\n', cnf_exp)
