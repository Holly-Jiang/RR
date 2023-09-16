import os
import random
import time
from typing import List
import re
from sympy import Interval, symbols, trace, simplify, factor_list

from QCTMC import qctmc_rho_t
from ConflictDrivenSolving import Factor_Polynomial, FactorPolynomial, Factor_Polynomial_By_X
from random_sat_generator import CNF_Observing


def poly_analysis(f_e,rho_t, P,height_from, height_to):
    factor = f_e.split(' ')
    Q_poly = factor[0]
    interval = Interval(int(factor[2]), 99999999999999, left_open=True, right_open=True)
    if factor[1].__eq__('>'):
        interval = Interval(int(factor[2]), 99999999999999, left_open=True, right_open=True)
    elif factor[1].__eq__('>='):
        interval = Interval(int(factor[2]), 99999999999999, left_open=False, right_open=True)
    elif factor[1].__eq__('<'):
        interval = Interval(-99999999999999, int(factor[2]), left_open=True, right_open=True)
    elif factor[1].__eq__('<='):
        interval = Interval(-99999999999999, int(factor[2]), left_open=True, right_open=False)
    elif factor[1].__eq__('=='):
        interval = Interval(int(factor[2]), int(factor[2]), left_open=False, right_open=False)
    reg = r"x\d+"
    params = list(set(re.findall(reg, Q_poly)))
    substi = {}
    for item in params:
        r = r"\d+"
        ind = eval(re.search(r, item).group()) - 1
        locals()[item] = symbols(item, real=True)
        substi[locals()[item]] = trace(P[ind] * rho_t)
    Q_poly = eval(Q_poly).subs(substi)

    # get the observing expression
    if len(interval.args) == 1:
        phi_t = (Q_poly - interval.args[0]) * (Q_poly - interval.args[0])
    else:
        if interval.start == 99999999999999 or interval.start == -99999999999999:
            phi_t = (Q_poly - interval.end)
        elif interval.end == 99999999999999 or interval.end == -99999999999999:
            phi_t = (Q_poly - interval.start)
        else:
            phi_t = (Q_poly - interval.start) * (Q_poly - interval.end)
    phi_t = simplify(phi_t)
    if str(phi_t).__contains__('exp') and (not str(phi_t).__contains__('+')) and (
            not str(phi_t).__contains__('-')):
        cnf_flag = False
        return []
    if (not str(phi_t).__contains__('exp')) and (not str(phi_t).__contains__('cos')):
        cnf_flag = False
        return []
    fac = Factor_Polynomial(str(phi_t), height_from, height_to)
    return fac
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


def calculator(variables, list, eqlist, degree, maxcoe_from, maxcoe_to, coe_bound_from, coe_bound_to):
    exp = []
    order = random.sample(range(0, len(variables)), len(variables))
    coelist = []
    maxcoe=0
    random_constant=random.choice([0,1])
    while maxcoe<maxcoe_from or maxcoe>maxcoe_to:
        coelist1 = []
        coelist = []
        len_coe=len(order)
        if random_constant:
            len_coe=len_coe+1
        for i in range(len_coe):  # 循环n次，每次产生一个新问题
            coe = random.randint(coe_bound_from, coe_bound_to)
            coelist1.append(coe)
        convention = gcd(coelist1) if len_coe > 1 else 1
        for i in range(len_coe):  # 循环n次，每次产生一个新问题
            coe = int(coelist1[i]/convention)
            coelist.append(coe)
        maxcoe=max(coelist)
    max_var=0
    for vi in range(len(variables)):
        vnum=len(variables[order[vi]].split('*'))
        if vnum>max_var:
            max_var=vnum
    for i in range(len(order)):  # 循环n次，每次产生一个新问题
        deg = random.randint(1, degree)  # 随机产生一个maximum以内整数
        vnum = len(variables[order[i]].split('*'))
        if max_var < degree:
            exp.append(f'{int(coelist[i])}*{variables[order[i]]}**{degree-vnum+1}')
            max_var = degree
        elif vnum >1:
            exp.append(f'{int(coelist[i])}*{variables[order[i]]}')
        else:
            dd=random.choice([1,deg])
            if degree-dd+1==1:
                exp.append(f'{int(coelist[i])}*{variables[order[i]]}')
            else:
                exp.append(f'{int(coelist[i])}*{variables[order[i]]}**{degree-dd+1}')
    observing = ''
    for i in range(len(exp)):
        if i > 0:
            sign = random.choice(list)
            observing += f'{sign}{exp[i]}'
        else:
            observing += f'{exp[i]}'
    if random_constant:
        constant = coelist[len(coelist)-1]
        sign = random.choice(list)
        observing += f'{sign}{constant}'
    eq = random.choice(eqlist)
    observing += f'{eq}0'
    # w.write('%s\n' % observing)
    # w.flush()
    return observing


def degree_product1(degree, vars):
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
        else:
            print('-------')
    return res

def degree_product(degree, vars):
    res = []
    if degree > 1 and len(vars) > 1:
        degvlist = []
        varset = subsets(vars)
        for vl in varset:
            if len(vl) >0 and len(vl) <= degree:
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
        else:
            # print('-------')
            pass
    return res


def random_expression(k, vars, clause, degree, maxcoe_from, maxcoe_to,is_bool,coe_bound_from, coe_bound_to):

    rho_t, P, qctmc_time = qctmc_rho_t(0)
    list = ['+', '-']
    eqlist = [' > ', ' < ', ' >= ', ' <= ']
    # path = './exp.txt'
    # w = open(path, 'w')
    varlist = ['x1', 'x2', 'x3', 'x4']
    sets = subsets(varlist)
    exp = []
    for v in sets:
        if len(v) >0:
            v.extend(degree_product(degree, varlist))
            flag=True
            while flag:
                exp_temp=calculator(v, list, eqlist, degree, maxcoe_from, maxcoe_to,coe_bound_from, coe_bound_to)
                factor=exp_temp.split(' ')
                fac = Factor_Polynomial_By_X(factor[0],P,rho_t)
                if len(fac) > 0:
                    flag=False
            exp.append(exp_temp)
    # w.close()
    obs = random.sample(range(0, len(exp)), vars)
    cnf_file=f'./cnf/{k}_{vars}_{clause}_{round(time.time(),10)}.txt'
    os.system(f'cnfgen randkcnf {k} {vars} {clause} > {cnf_file}')
    observing = []
    for i in range(len(obs)):
        observing.append(exp[obs[i]])
    print('OBSERVING EXPRESSION:\n', observing)
    cnf, cnf_exp, cnf_exp_list = CNF_Observing(f'{cnf_file}', observing)
    print('CNF:\n', cnf)
    print('FINAL:\n', cnf_exp)
    return observing, cnf_exp, cnf_exp_list


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
    os.system(f'cnfgen randkcnf {k} {vars} {clause} > cnf_{time.time()}.txt')
    observing = []
    for i in range(len(obs)):
        observing.append(exp[obs[i]])
    print('OBSERVING EXPRESSION:\n', observing)
    cnf, cnf_exp = CNF_Observing('./cnf.txt', observing)
    print('CNF:\n', cnf)
    print('FINAL:\n', cnf_exp)
