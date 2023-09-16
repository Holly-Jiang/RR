import math
import time

from sympy import *
from sympy.physics.quantum import TensorProduct
from sympy.polys import factor_list
import random, re

def trace(mat):
    dim = shape(mat)[0]
    returnTrace = 0
    for i in range(0, dim):
        returnTrace += mat[i, i]
    return returnTrace

def Factor_Polynomial_By_X(f,P,rho_t):

    t = symbols('t', real=True)
    reg = r"x\d+"
    params = list(set(re.findall(reg, f)))
    substi = {}
    for item in params:
        r = r"\d+"
        ind = eval(re.search(r, item).group()) - 1
        locals()[item] = symbols(item)
        substi[locals()[item]] = trace(P[ind] * rho_t)
    f = eval(f).subs(substi)
    f=str(f)
    f=str(expand(eval(f)))
    factor=FactorPolynomial(f)
    return factor

def Factor_Polynomial(f, coe_from, coe_to):

    t = symbols('t', real=True)
    # f = '(0.125 - exp(-4*t)/8)*(0.375 + 0.5*exp(-2*t)*cos(2*t) + 0.125*exp(-4*t)) + (0.375 - 0.5*exp(-2*t)*cos(2*t) + 0.125*exp(-4*t))*(0.375 + 0.5*exp(-2*t)*cos(-2*t) + 0.125*exp(-4*t)) + 0.125 - exp(-4*t)/8'
    reg = r'exp\(.\d*.t\)'
    reg1 = r'cos\(.\d*.t\)'
    params1 = list(set(re.findall(reg, f)))
    tvars_all = []
    tvars = []
    tvars_map = {}
    params=[]
    if len(params1)>0:
        for i in range(len(params1)):
            if params1[i].__contains__('-'):
                params.append('exp(-t)')
                break
        for i in range(len(params1)):
            if not params1[i].__contains__('-'):
                params.append('exp(t)')
                break
    flag1 = False
    flag2 = False
    for i in range(len(params1)):
        if params1[i].__contains__('-'):
            str1 = params1[i].split('exp(-')[1]
            if str1.startswith('t'):
                str1 = 1
            else:
                str1 = int(str1.split('*')[0])
            str2 = f't0**{str1}'
            tvars_all.append(str2)
            if not flag1:
                tvars.append('t0')
                locals()[f't0'] = symbols(f't0', real=True)
                flag1 = True
        if not params1[i].__contains__('-'):
            str1 = params1[i].split('exp(')[1]
            if str1.startswith('t'):
                str1 = 1
            else:
                str1 = int(str1.split('*')[0])
            str2 = f't0**{str1}'
            tvars_all.append(str2)
            if not flag2:
                tvars.append('t1')
                locals()[f't1'] = symbols(f't1', real=True)
                flag2 = True
    params2 = list(set(re.findall(reg1, f)))

    for i in range(len(params2)):
        tvars.append(f't{i + len(params)}')
        locals()[f't{i + len(params)}'] = symbols(f't{i + len(params)}', real=True)
    for i in range(len(params1)):
        item = params1[i]
        f = f.replace(item, tvars_all[i])
    for i in range(len(params2)):
        item = params2[i]
        f = f.replace(item, tvars[i + len(params)])
    params.extend(params2)
    Q_poly = eval(f)
    f1=expand(Q_poly*10000000)
    if str(f1).__contains__('/'):
        return []
    coe, coelist = coe_poly(f1)
    convention=gcd(coelist)
    max_coe=0
    for c in coelist:
        if Abs(c/convention)>max_coe:
            max_coe=Abs(c/convention)
    f2=(f1)/convention
    print(f"f2 {max_coe} {f2}")
    if max_coe<coe_from or max_coe>coe_to:
        return []

    expstr = str(f2)
    for i in range(len(tvars)):
        expstr = expstr.replace(tvars[i], params[i])

    print(f"phi_t: {expstr}")
    # return [expstr]
    factorF = factor_list(str(f2))[1]
    factors = []
    coe = 0

    for item in factorF:
        coe,coelist=coe_poly(item[0])
        expstr = str(item[0])
        for i in range(len(tvars)):
            expstr = expstr.replace(tvars[i], params[i])

        factors.append(eval(expstr))
    if coe < coe_to:
        print(factorF)
        return []
    return factors

def coe_poly(item):
    coe=0
    coelist=[]
    for it in item.args:
        strit = str(it)
        if strit.__contains__('t'):
            stritt = strit.split('t')[0]
            if stritt == '':
                itf = 1
            elif stritt.__contains__('*'):
                itf = float(stritt.split('*')[0])
            else:
                itf = float(strit)
            coe = itf if Abs(itf) > coe else coe
            coelist.append(int(itf))
        else:
            itf = float(strit)
            coe = itf if Abs(itf) > coe else coe
            coelist.append(int(itf))
    return coe, coelist

def FactorPolynomial(f):
    t = symbols('t', real=True)
    f1=eval(f)
    # f = '(0.125 - exp(-4*t)/8)*(0.375 + 0.5*exp(-2*t)*cos(2*t) + 0.125*exp(-4*t)) + (0.375 - 0.5*exp(-2*t)*cos(2*t) + 0.125*exp(-4*t))*(0.375 + 0.5*exp(-2*t)*cos(-2*t) + 0.125*exp(-4*t)) + 0.125 - exp(-4*t)/8'
    reg = r'exp\(-*\d*\**t\)'
    reg1 = r'cos\(-*\d*\**t\)'
    params1 = list(set(re.findall(reg, f)))
    tvars_all = []
    tvars = []
    tvars_map = {}
    params = []
    if len(params1) > 0:
        for i in range(len(params1)):
            if params1[i].__contains__('-'):
                params.append('exp(-t)')
                break
        for i in range(len(params1)):
            if not params1[i].__contains__('-'):
                params.append('exp(t)')
                break
    flag1 = False
    flag2 = False
    for i in range(len(params1)):
        if params1[i].__contains__('-'):
            str1 = params1[i].split('exp(-')[1]
            if str1.startswith('t'):
                str1 = 1
            else:
                str1 = int(str1.split('*')[0])
            str2 = f't0**{str1}'
            tvars_all.append(str2)
            if not flag1:
                tvars.append('t0')
                locals()[f't0'] = symbols(f't0', real=True)
                flag1 = True
        if not params1[i].__contains__('-'):
            str1 = params1[i].split('exp(')[1]
            if str1.startswith('t'):
                str1 = 1
            else:
                str1 = int(str1.split('*')[0])
            str2 = f't0**{str1}'
            tvars_all.append(str2)
            if not flag2:
                tvars.append('t1')
                locals()[f't1'] = symbols(f't1', real=True)
                flag2 = True
    params2 = list(set(re.findall(reg1, f)))

    for i in range(len(params2)):
        tvars.append(f't{i + len(params)}')
        locals()[f't{i + len(params)}'] = symbols(f't{i + len(params)}', real=True)
    for i in range(len(params1)):
        item = params1[i]
        f = f.replace(item, tvars_all[i])
    for i in range(len(params2)):
        item = params2[i]
        f = f.replace(item, tvars[i + len(params)])
    params.extend(params2)
    Q_poly = simplify(eval(f))
    factorF = factor_list(f)[1]

    factors = []
    for item in factorF:
        expstr = str(item[0])
        for i in range(len(tvars)):
            expstr = expstr.replace(tvars[i], params[i])
        ff = simplify(eval(expstr))
        for ff1 in ff.args:
            restr = list(set(re.findall(r'-*\d*\**exp\(-*\d*\**t\)', str(ff1))))
            if len(restr) == 1 and restr[0] == str(ff1):
                continue
            factors.append(eval(expstr))
    if len(factors)==1:
        return [f1]
    return factors




def IntervalMinusPoint(invl1, t_star, samples):
    accu=0.000001
    # if len(invl1) < 200:
    #     accu = 0.000001
    # elif len(invl1) < 300:
    #     accu = 0.0001
    # elif len(invl1) < 500:
    #     accu = 0.01
    # elif len(invl1) < 700:
    #     accu = 0.05
    # elif len(invl1) < 1000:
    #     accu = 0.1
    # elif len(invl1) < 1500:
    #     accu = 0.5
    # else:
    #     print('2000------------')
    #     accu = 1
    returnInvl = invl1[:]
    for invl in invl1:
        l, u = invl[0], invl[1]
        if Abs(u - l) < accu:
            for i in samples:
                if invl in returnInvl:
                    if Abs(i-l) < accu and Abs(i-u) < accu:
                        # print(f'{len(returnInvl)}  invl:{invl}')
                        returnInvl.remove(invl)

    return returnInvl


def IntervalMinus1(invl1, invl2,J):
    delta=invl1[:]
    for intvl1 in invl1:
        flag = True
        inf = intvl1[0] - J[1]
        sup = intvl1[1] - J[0]
        for i in range(len(invl2)):
            sec=Intersection(Interval(invl2[i][0], invl2[i][1], False, False), Interval(inf, sup, False, False))
            if len(sec.args)>0:
                flag = False
                break

        if flag is True:
            delta.remove(intvl1)
    return delta



def IntervalMinus(invl1, invl2):
    '''
    :param invl1: [[l1, u1], [l2, u2], ...] disjoint intervals
    :param invl2: [ll, uu] single interval
    :return:
    '''
    returnInvl = invl1[:]
    if invl2 is None or len(invl2) == 0:
        return returnInvl
    for invl in invl1:
        l, u, ll, uu = invl[0], invl[1], invl2[0], invl2[1]  # implicitly uu > ll
        if ll < l:
            if uu < l:
                continue
            elif uu < u and uu >= l:
                returnInvl.remove(invl)
                returnInvl.append([uu, u])
            elif uu > u:
                returnInvl.remove(invl)
            elif uu==u:
                returnInvl.remove(invl)
                returnInvl.append([uu, uu])
        elif ll == l:
            if uu < l:
                print(f'Exception: {ll,uu}')
                continue
            elif uu < u and uu >= l:
                returnInvl.remove(invl)
                returnInvl.append([uu, u])
                returnInvl.append([ll, ll])
            elif uu > u:
                returnInvl.remove(invl)
                returnInvl.append([ll, ll])
            elif uu == u:
                returnInvl.remove(invl)
                returnInvl.append([uu, uu])
                returnInvl.append([ll, ll])
        elif ll > l and ll <= u:
            if uu > l and uu < u:
                returnInvl.remove(invl)
                returnInvl.append([l, ll])
                returnInvl.append([uu, u])
            elif uu > u:
                returnInvl.remove(invl)
                returnInvl.append([l, ll])
            elif uu == u:
                returnInvl.remove(invl)
                returnInvl.append([l, ll])
                returnInvl.append([uu, uu])
        elif ll > u:
            pass
    returnInvl.sort()
    if len(returnInvl) == 0:
        return []
    else:
        return returnInvl


def IntervalMinusClose(invl1, invl2):
    '''
    :param invl1: [[l1, u1], [l2, u2], ...] disjoint intervals
    :param invl2: [ll, uu] single interval
    :return:
    '''
    returnInvl = invl1[:]
    if invl2 is None or len(invl2) == 0:
        return returnInvl
    for invl in invl1:
        l, u, ll, uu = invl[0], invl[1], invl2[0], invl2[1]  # implicitly uu > ll
        if ll <= l:
            if uu < l:
                continue
            elif uu < u and uu >= l:
                returnInvl.remove(invl)
                returnInvl.append([uu, u])
            elif uu >= u:
                returnInvl.remove(invl)
        elif ll > l and ll <= u:
            if uu > l and uu < u:
                returnInvl.remove(invl)
                returnInvl.append([l, ll])
                returnInvl.append([uu, u])
            elif uu >= u:
                returnInvl.remove(invl)
                returnInvl.append([l, ll])
        elif ll > u:
            pass
    returnInvl.sort()
    if len(returnInvl) == 0:
        return []
    else:
        return returnInvl


def IntervalMerge(invl):
    ind, lenth = 0, len(invl)
    while ind < lenth - 1:
        if invl[ind][1] == invl[ind + 1][0]:
            invl[ind + 1][0] = invl[ind][0]
            invl.pop(ind)
            lenth = len(invl)
        else:
            ind += 1
    return invl


def IntervalPlus(invl1, invl2):
    '''
        :param invl1: [[l1, u1], [l2, u2], ...] disjoint intervals
        :param invl2: [ll, uu] single interval
        :return:
        '''
    returnInvl = invl1[:]
    ll, uu = invl2[0], invl2[1]
    if len(invl1) == 0:
        if ll < uu:
            returnInvl.append(invl2)
            return returnInvl
        else:
            print('uu < ll')
    for invl in invl1:
        l, u = invl[0], invl[1]  # implicitly uu > ll

        if ll <= l:
            if uu < l:
                returnInvl.append([ll, uu])
                break
            elif uu <= u and uu >= l:
                returnInvl.remove(invl)
                returnInvl.append([ll, u])
                break
            elif uu > u:
                returnInvl.remove(invl)
                returnInvl.append([ll, u])
                returnInvl = IntervalPlus(returnInvl, [u, uu])
                break
        elif ll > l and ll < u:
            if uu >= l and uu <= u:
                break
            elif uu > u:
                returnInvl.append([u, uu])
                # returnInvl = IntervalPlus(returnInvl, [u, uu])
                break
        elif ll >= u and uu > u:
            returnInvl.append([ll, uu])
            pass
    returnInvl.sort()
    returnInvl = IntervalMerge(returnInvl)
    return returnInvl


def SupValue4AbsFuncAppro(func, var, invl):
    '''
    :param func:
    :param var:
    :param invl: closed interval
    :return:
    '''
    l, u = invl[0], invl[1]
    M = max(Abs(func.subs({var: l})), Abs(func.subs({var: u})))
    N = 20
    delta = (u - l) / N
    i = l
    while i < u:
        funcVal = Abs(func.subs({var: i}))
        M = funcVal if funcVal >= M else M
        i += delta
    return M


def ChooseElement(B, left_open, right_open):
    '''
    :param B: [[l1,u1], [l2,u2], ...]list of intervals
    :return:
    '''
    if random.choice([0, 1]):
        ind=0
        max=0
        for i in range(len(B)):
            if B[i][1]-B[i][0]>max:
                ind=i
                max=B[i][1]-B[i][0]
    else:
        ind = random.randint(0, len(B) - 1)
    if Abs(B[ind][1] - B[ind][0])  <= 0.000000000001:
        return B[ind][0]+(B[ind][1] - B[ind][0])/2
    n = 10
    delta = (B[ind][1] - B[ind][0]) / n
    if left_open:
        i = B[ind][0] + delta
    else:
        i = B[ind][0]
    if right_open:
        tt = B[ind][1] - delta
    else:
        tt = B[ind][1]
    cc = []
    while i - tt <= 0.000001:
        cc.append(i)
        i += delta
    t = random.choice(cc)
    return float(t)
    # t = round(random.uniform(B[ind][0], B[ind][1]),6)
    # # TODO: 1.check whether the end-points can be selected (the close problem of intervals)
    # #       2.the chosen element is int, or float is also ok?
    # return t


def Q_poly_sub(Q_poly, P, rho_t):
    reg = r"x\d+"
    params = list(set(re.findall(reg, Q_poly)))
    substi = {}
    for item in params:
        r = r"\d+"
        ind = eval(re.search(r, item).group()) - 1
        locals()[item] = symbols(item)
        substi[locals()[item]] = trace(P[ind] * rho_t)
    Q_poly1 = eval(Q_poly).subs(substi)
    return Q_poly1


def SignInvariantNeighbor(rho_t, t_star, Phi, B, phi_t, factors):
    # TODO: how to express polynomial of (x_s)_{s \in S}?
    '''
    :param rho_t: density operator (in matrix form)
    :param t_star:
    :param Phi: {'projectors': [P1, P2, ...], 'poly': '3*x1**2+x1*x2+4*x2>=0', 'interval': [l, u]}
    :param B: [[l1,u1], [l2,u2], ...] list of intervals
    :return:
    '''
    P, Q_poly, intvl = Phi['projector'], Phi['poly'], Phi['interval']

    Q_poly=Q_poly_sub(Q_poly, P, rho_t)
    # get the observing expression
    # print(f'phi_t: {phi_t}')
    if len(Q_poly.free_symbols)==0:
        return Interval(B[0][0], B[0][1], False, False)
    # get the variable of polynomial
    variable = list(Q_poly.free_symbols)[0]
    # get the factors of the observing expression
    # factors = [phi_t]
    factors_first_deriv = [diff(item, variable) for item in factors]
    factors_second_deriv = [diff(item, variable) for item in factors_first_deriv]
    # main body for computing the sign-invariant neighbourhood of t*
    if phi_t.evalf(subs={variable: t_star}) == 0:
        # equation-type
        return [t_star]
    else:
        # inequality-type
        delta = []
        for i in range(0, len(factors)):
            sup_Psi_j, sup_Psi_jj = 0, 0
            for item in B:
                sup_Psi_invl1 = Abs(float(SupValue4AbsFuncAppro(factors_first_deriv[i], variable, item)))
                sup_Psi_j = sup_Psi_j if sup_Psi_j >= sup_Psi_invl1 else sup_Psi_invl1
                sup_Psi_invl2 = Abs(float(SupValue4AbsFuncAppro(factors_second_deriv[i], variable, item)))
                sup_Psi_jj = sup_Psi_jj if sup_Psi_jj >= sup_Psi_invl2 else sup_Psi_invl2
            # compute \epsilon_j
            epsilon_j = Rational(Abs(factors[i].subs({variable: t_star})), sup_Psi_j)
            # compute \varepsilon_j
            varepsilon_j = Rational(Abs(factors_first_deriv[i].subs({variable: t_star})), sup_Psi_jj)
            # print(f'{Abs(factors[i].subs({variable: t_star}))} {Abs(factors_first_deriv[i].subs({variable: t_star}))}')
            # print(f'epsilon_j: {epsilon_j}, varepsilon_j:{varepsilon_j}')
            # compute delta_j 's end-points
            if varepsilon_j <= epsilon_j:
                inf_delta_j = t_star - epsilon_j
                sup_delta_j = t_star + epsilon_j
            else:
                if varepsilon_j > epsilon_j and factors[i].subs({variable: t_star}) * factors[i].subs(
                        {variable: t_star - varepsilon_j}) > 0:
                    inf_delta_j = t_star - varepsilon_j
                else:
                    s_star = Interval_zero_root_left(factors[i], variable, t_star - varepsilon_j, t_star - epsilon_j)
                    inf_delta_j = s_star
                    pass  # TODO: l = s_star

                if varepsilon_j > epsilon_j and factors[i].subs({variable: t_star}) * factors[i].subs(
                        {variable: t_star + varepsilon_j}) > 0:
                    sup_delta_j = t_star + varepsilon_j
                else:
                    s_star = Interval_zero_root_right(factors[i], variable, t_star + epsilon_j, t_star + varepsilon_j)
                    sup_delta_j = s_star
                    pass  # TODO: l = s_star
            if inf_delta_j > sup_delta_j:
                print('the interval inf >sup: [%s,%s]' % (inf_delta_j, sup_delta_j))
            # str_inf_delta_j=str(round(inf_delta_j,4))
            # str_sup_delta_j = str(round(sup_delta_j,4))
            # delta_j = Interval(float(str_inf_delta_j[:len(str_inf_delta_j)-2])+0.01, float(str_sup_delta_j[:len(str_sup_delta_j)-2]), left_open=True, right_open=True)
            delta_j = Interval(inf_delta_j, sup_delta_j, left_open=True, right_open=True)
            delta.append(delta_j)
        # the intersection betwwen \delta and B
        # for intvl2 in B:
        #     intvl3 = Interval(intvl2[0], intvl2[1], left_open=False, right_open=False)
        #     delta.append(intvl3)
        res = delta[0]
        for delta_i in delta:
            res = Intersection(delta_i, res)
        if res is EmptySet:
            return [t_star]

        return res


def Interval_zero_root(phi_t, variable, left, right):
    medium = left + (right - left) / 40
    med_root = phi_t.subs({variable: medium})
    left_root = phi_t.subs({variable: left})
    right_root = phi_t.subs({variable: right})
    res = -1
    if med_root == 0:
        res = medium
    elif right_root * med_root < 0:
        res = Interval_zero_root(phi_t, variable, medium, right)
    elif left_root * med_root < 0:
        res = Interval_zero_root(phi_t, variable, left, medium)
    return res


def Interval_zero_root_right(phi_t, variable, left, right):
    delta = (right - left) / 40
    medium = left + delta
    left_root = phi_t.subs({variable: left})
    root = left
    while medium < right:
        med_root = phi_t.subs({variable: medium})
        if left_root * med_root > 0:
            root = medium
        medium += delta
    return root


def Interval_zero_root_left(phi_t, variable, left, right):
    delta = (right - left) / 40
    medium = right - delta
    right_root = phi_t.subs({variable: right})
    root = right
    while medium > left:
        med_root = phi_t.subs({variable: medium})
        if right_root * med_root > 0:
            root = medium
        medium -= delta
    return root


# def Interval_zero_root_right(phi_t, variable, left, right, iter):
#     medium = left+(right - left) / 2
#     med_root = phi_t.subs({variable: medium})
#     left_root = phi_t.subs({variable: left})
#     right_root = phi_t.subs({variable: right})
#     iter+=1
#     res=-1
#     if med_root == 0:
#         res = medium
#     elif iter >=2 and left_root * med_root>0:
#         res = medium
#     elif right_root * med_root < 0:
#         res=Interval_zero_root_right(phi_t, variable, medium, right,iter)
#     elif left_root * med_root < 0:
#         res=Interval_zero_root_right(phi_t, variable, left, medium,iter)
#     return res
def SatisfiedRR(rho_t, t_star, Phi):
    # TODO: how to express polynomial of (x_s)_{s \in S}?
    '''
    :param rho_t: density operator (in matrix form)
    :param t_star:
    :param Phi: {'projectors': [P1, P2, ...], 'poly': '...', 'interval': [l, u]}
    TODO: here interval is temporarily closed in both end-points
    :return:
    '''
    P, Q_poly, intvl = Phi['projector'], Phi['poly'], Phi['interval']
    reg = r"x\d+"
    params = list(set(re.findall(reg, Q_poly)))
    substi = {}
    for item in params:
        r = r"\d+"
        ind = eval(re.search(r, item).group()) - 1
        locals()[item] = symbols(item)
        substi[locals()[item]] = trace(P[ind] * rho_t)
    Q_poly = eval(Q_poly).subs(substi)
    # get the variable of polynomial
    if len(Q_poly.free_symbols)>0:
        variable = list(Q_poly.free_symbols)[0]
        Q_t_star = Q_poly.subs({variable: t_star})
    else:
        Q_t_star=Q_poly
    if len(intvl.args) == 1:
        if Abs(Q_t_star - intvl.args[0]) < 0.000001:
            # print(f'Q_t_star:{Q_t_star}')
            return True
        return False
    elif Q_t_star > intvl.start:
        if Q_t_star < intvl.end:
            return True
        elif (not intvl.right_open) and Q_t_star == intvl.end:
            return True
        else:
            return False
    elif (not intvl.left_open) and Q_t_star == intvl.start:
        if Q_t_star < intvl.end:
            return True
        elif (not intvl.right_open) and Q_t_star == intvl.end:
            return True
        else:
            return False
    else:
        return False


def FindSamplePoints(l, u, difference):
    pointsList = []
    if l > u:
        return pointsList
    if l == u:
        pointsList.append(l)
        return pointsList
    lenth = u - l
    i = 1
    while True:
        if Rational(lenth, i) <= difference:
            break
        i += 1
    diff = Rational(lenth, i)
    for j in range(0, i):
        pointsList.append(l + j * diff)
    if u - (l + i * diff)>0.0001:
        raise Exception('the right-end is not matched')
    return pointsList

def IntervalMinusCloseSet(invl1, invl2):
    returnInvl = invl1[:]
    for invl in invl2:
        if returnInvl is None or len(returnInvl)==0:
            # print(f'I-I\': None ')
            return None
        returnInvl = IntervalMinusClose(returnInvl, invl)
    # print(f'I-I\': {returnInvl}')
    if returnInvl is None or len(returnInvl) == 0:
        # print(f'I-I\': None ')
        return None
    return returnInvl

def IntervalMinusSet(invl1, invl2):
    returnInvl = invl1[:]
    for invl in invl2:
        if returnInvl is None or len(returnInvl)==0:
            # print(f'I-I\': None ')
            return None
        returnInvl = IntervalMinus(returnInvl, invl)
    # print(f'I-I\': {returnInvl}')
    if returnInvl is None or len(returnInvl) == 0:
        # print(f'I-I\': None ')
        return None
    return returnInvl


def tStarCandidate(B):
    cc = []
    for invl in B:
        n = 40
        delta = (invl[1] - invl[0]) / n
        i = invl[0]
        while i <= invl[1]:
            cc.append(i)
            i += delta
    return cc


def chooseTStar(C: list):
    t = random.choice(C)
    C.remove(t)
    return t


def ConflictDrivenSolving(rho_t, rr, phi_t, factors):
    '''
    :param rho_t: polynomial of 'dynamics of a QCTMC'
    :param rr: [interval of 'box', interval of 'diamond', Phi]
    :return: [t_1*, t_2*, ...] finitely many absolute times
    '''
    I, J, Phi = rr[0], rr[1], rr[2]
    inf_I = I[0]
    sup_I = I[1]
    inf_J = J[0]
    sup_J = J[1]
    B = [[inf_I + inf_J, sup_I + sup_J]]  # the post_monitoring period of □^I♢^J Phi
    T = set()
    I = [I]
    I_prime = []
    count = 0
    # main body of the algorithm
    samples = []
    II=IntervalMinusSet(I, I_prime)

    while (B is not None and len(B) != 0) and II is not None:
        # print("B:", len(B), B)
        # print("I_prime:", I_prime)
        # print('T:', len(T), T)
        # 1.560936204101162
        # samples = [2.5, 3.0, 2.2, 1.9, 1.7, 1.561,1.56, 1.56093620410116, 1.56093620410117, 1.45, 1.3, 1.2, 1.0]
        # samples = [ 3.0,2.5, 2.2, 1.9, 1.7, 1.6,1.5,1.4, 1.3, 1.2, 1.0]

        # samples=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
        # samples=[1.0,1.5]
        # t_star = samples[count]
        count += 1
        t_star = ChooseElement(B, False, False)
        # print('t_star:',t_star)

        delta = SignInvariantNeighbor(rho_t, t_star, Phi, [[inf_I + inf_J, sup_I + sup_J]], phi_t, factors)
        if isinstance(delta, Interval):
            delta = [delta.start, delta.end]
        elif len(delta) == 1:
            delta = list(delta)
            delta = [delta[0], delta[0]]
        if SatisfiedRR(rho_t, t_star, Phi):
            if delta[1] - delta[0] > sup_J - inf_J:
                    l = ChooseElement([[delta[0], delta[0] + (sup_J - inf_J) / 2]], False, True)
                    u = ChooseElement([[delta[1] - (sup_J - inf_J) / 2, delta[1]]], True, False)
            else:
                l = t_star
                u = t_star
            if l < u:
                T_prime = FindSamplePoints(l, u, sup_J - inf_J)
                I_prime = IntervalPlus(I_prime, [l - sup_J, u - inf_J])
                T.update(T_prime)
            else:
                T_prime = [t_star]
                I_prime = IntervalPlus(I_prime, [t_star - sup_J, t_star - inf_J])
                T.update(T_prime)

        B = IntervalMinusPoint(B, t_star, samples)
        B = IntervalMinus(B, delta)
        II=IntervalMinusSet(I, I_prime)
        # print(f'I: {I} J: {J}')
        if II is not None:
            B= IntervalMinus1(B, II, J)
        # print(f"B:{len(B)}, delta:{delta} II:{II}")
        samples.append(t_star)
    if IntervalMinusSet(I, I_prime) is None:
        print("TRUE SAMPLE-DRIVEN:")
        # print('I:', I, "I_prime:", I_prime)
        # print('T:', len(T), T)
        return 1
    else:
        print("FALSE SAMPLE-DRIVEN:")
        # print('I:', I, "I_prime:", I_prime)
        # print('T:', len(T), T)
        return 0


if __name__ == '__main__':
    start = time.time()
    t0 = symbols('t0', real=True)
    t2 = symbols('t2', real=True)
    t1 = symbols('t1', real=True)
    t = symbols('t', real=True)
    f=eval('-2812500.0*t0**8 + 11875000.0*t0**6*t1 + 17500000.0*t0**4*t1**2 - 1250000.0*t0**4 + 23125000.0*t0**2*t1 - 68437500.0')
    coe_poly(expand(f))
    # FactorPolynomial('')
    rho_t = Matrix([[0.375 + 0.5 * exp(-2 * t) * cos(2 * t) + 0.125 * exp(-4 * t), 0, 0,
                     0.125 + 0.5 * I * exp(-2 * t) * sin(2 * t) - 0.125 * exp(-4 * t)],
                    [0, 1 / 8 - exp(-4 * t) / 8, 1 / 8 - exp(-4 * t) / 8, 0],
                    [0, 1 / 8 - exp(-4 * t) / 8, 1 / 8 - exp(-4 * t) / 8, 0],
                    [0.125 - 0.5 * I * exp(-2 * t) * sin(2 * t) - 0.125 * exp(-4 * t), 0, 0,
                     0.375 - 0.5 * exp(-2 * t) * cos(2 * t) + 0.125 * exp(-4 * t)]])
    ket_0 = Matrix([[1, 0], [0, 0]])
    ket_1 = Matrix([[0, 0], [0, 1]])
    ket_0_0 = TensorProduct(ket_0, ket_0)
    ket_1_1 = TensorProduct(ket_1, ket_1)
    ket_0_1 = TensorProduct(ket_0, ket_1)
    ket_1_0 = TensorProduct(ket_1, ket_0)
    # phi='-1/64-exp(-8*t)/64-7*exp(-4*t)/32-1/8*exp(-6*t)*cos(2*t)-3/8*exp(-2*t)*cos(2*t)-1/4*exp(-4*t)*cos(2*t)*cos(2*t)'
    # phi='-1/64-3/16*exp(-(2+2*I)*t)-3/16*exp(-(2-2*I)*t)-1/16*exp(-(4+4*I)*t)-1/16*exp(-(4-4*I)*t)-1/16*exp(-(6+2*I)*t)-1/16*exp(-(6-2*I)*t)-11/32*exp(-4*t)-1/64*exp(-8*t)'
    phi='-(3*exp(4*t) + 4*exp(2*t)*cos(2*t) + 1)**2*exp(-8*t)/64 + 1/8 - exp(-4*t)/8'
    # phi = '2.015625 - 4.6875*exp(-2*t)*cos(2*t) + 0.78125*exp(-4*t) + 0.6875*exp(-6*t)*cos(2*t) + 0.203125*exp(-8*t)'
    t=symbols('t', real=True)
    f=eval(phi)
    i=0
    while i<=3:
        value=f.subs({t: i})
        if value>-0.01:
            print(f'{i:.6f} {value*10-0.3:.6f}')
            pass
        else:
            # print(f'{i:.6f} {value-0.09:.6f}')
            pass
        i+=0.001

    P = [ket_0_0, ket_0_1, ket_1_0, ket_1_1]
    box_I = [0, 2]
    diamond_J = [0, 5]
    interval = Interval(0, 9999, left_open=False, right_open=False)
    Phi = {'projector': P, 'poly': phi, 'interval': interval}
    rr = [box_I, diamond_J, Phi]
    print(ConflictDrivenSolving(rho_t, rr))
    # B[ind]: [1.56093620410116, 1.56093620410116] delta: 1.05471187339390e-16
    print(time.time() - start)
