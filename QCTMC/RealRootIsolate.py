import os
import time
from functools import wraps
from sympy import *
from sympy.calculus.util import *
from scipy.optimize import fsolve


# TODO: check 'maximum' applied to 'Abs function'

# Global Variable
import psutil


class fancy_timed(object):
    def __init__(self, f):
        self.f = f
        self.active = False

    def __call__(self, *args):
        if self.active:
            return self.f(*args)
        start = time.time()
        self.active = True
        res = self.f(*args)
        end = time.time()
        self.active = False
        print("the total time consumed for real root isolation: ", end - start, "s")
        return res


class RealRootIsolate:
    def __init__(self, variable, func, invl, N=2):
        self.t = symbols(variable)
        locals()[variable] = symbols(variable)
        self.f = eval(func).simplify()
        self.invl = invl
        self.N = N
        self.solution = list()
        self.cache=0

    def RenewInterval(self, invl):
        self.invl = invl

    def SupValue4AbsFunc(self, func, var, invl):
        l, u = invl[0], invl[1]

        M = max(Abs(func.subs({var: l})), Abs(func.subs({var: u})))
        df = diff(func, var).simplify()
        print('SupValue4AbsFunc df:', latex(df))
        sln = solve(df, var)
        for value in sln:
            if value in invl:
                funcVal = Abs(func.subs({var: value}))
                M = funcVal if funcVal >= M else M
        return M

    def SupValue4AbsFuncAppro(self, func, var, invl):
        l, u = invl[0], invl[1]
        M = max(Abs(func.subs({var: l})), Abs(func.subs({var: u})))
        N = 40
        delta = (u - l) / N
        i = l + delta
        while i < u:
            funcVal = Abs(func.subs({var: i}))
            M = funcVal if funcVal >= M else M
            i += delta
        return M

    def JointInterval(self, invl1, invl2):
        l = max(invl1[0], invl2[0])
        u = min(invl1[1], invl2[1])
        if l >= u:
            return None
        else:
            return [l, u]

    @fancy_timed
    def RealRootIsolation(self, interval):
        # print("======== enter the function ========")
        l, u = interval[0], interval[1]
        # invl = Interval(l, u)
        # if self.f or self.f or self.f:
        #     raise Exception('The real-valued function does not satisfy pre-conditions, please check!')

        first_order_f = diff(self.f, self.t).simplify()
        second_order_f = diff(first_order_f, self.t).simplify()

        # M = max(Abs(maximum(first_order_f, self.t, invl)), Abs(minimum(first_order_f, self.t, invl)))
        # M_prime = max(Abs(maximum(second_order_f, self.t, invl)), Abs(minimum(second_order_f, self.t, invl)))

        M = self.SupValue4AbsFuncAppro(first_order_f, self.t, interval)
        M_prime = self.SupValue4AbsFuncAppro(second_order_f, self.t, interval)

        i = 0
        # print('u:  ', u,'  l:  ', l)
        delta = Rational(u - l, self.N)
        # print('the value of delta:  ', delta)

        # main body of algorithm
        while i < self.N:
            iso_cache = (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
            if iso_cache>self.cache:
                self.cache=iso_cache
            # print(" the iteration number: ", i)
            f_val1 = self.f.subs({self.t: l + i * delta})
            f_val2 = self.f.subs({self.t: l + i * delta + delta})
            f_val3 = self.f.subs({self.t: l + i * delta + 2 * delta})
            f_prime_val1 = first_order_f.subs({self.t: l + i * delta})
            f_prime_val2 = first_order_f.subs({self.t: l + i * delta + delta})
            if Abs(f_val1) > M * delta:
                i += 1
            elif Abs(f_val2) > M * delta:
                i += 2
            elif Abs(f_prime_val1) >= M_prime * delta:
                # print('f_val2', f_val1)
                # print('f_val2', f_val2)
                if f_val1 * f_val2 < 0:
                    self.solution.append([l + i * delta, l + i * delta + delta])
                    # print('here1:  ', [l + i * delta, l + i * delta + delta])
                i += 1
            elif Abs(f_prime_val2) >= M_prime * delta:
                if f_val1 * f_val2 < 0:
                    self.solution.append([l + i * delta, l + i * delta + delta])
                    # print('here2:  ', [l + i * delta, l + i * delta + delta])
                # print('f_val3', f_val1 * f_val3, f_val1 * f_val3 < 0)
                if i + 2 <= self.N and f_val2 * f_val3 < 0:
                    self.solution.append([l + i * delta + delta, l + i * delta + 2 * delta])
                    # print('here3:  ', [l + i * delta + delta, l + i * delta + 2 * delta])
                i += 2
            else:
                invl2sol = self.JointInterval(interval, [l + i * delta, l + i * delta + delta])
                # print('next interval to isolate: ',invl2sol)
                if invl2sol is not None:
                    self.RealRootIsolation(self, invl2sol)
                # print('-------- exit the while loop -------',invl2sol)
                i += 1
                # self.RealRootIsolation(self, [l + i * delta, l + i * delta + delta])
                # print('-------- exit the while loop -------')
                # i += 1


if __name__ == '__main__':
    t = symbols('t')

    f = '-Rational(1,5) - sqrt(2) * Rational(1,2) * exp(-(2+sqrt(2))*Rational(1,2) * t) + sqrt(2)*Rational(1,2)*exp(-(2-sqrt(2))*Rational(1,2)* t )'
    # print(f)
    f2 = '1 + (sqrt(2) - Rational(1,2)) * exp(-(2+sqrt(2))*Rational(1,2)* t) - (sqrt(2)+Rational(1,2))*exp(-(2-sqrt(2))*Rational(1,2)* t)'

    x1 = 'Rational(3,8) +  Rational(1,4)*exp(-(2+2*I)*t)+  Rational(1,4)*exp(-(2-2*I)*t) + Rational(1,8)*exp(-4*t)'
    x2 = 'Rational(1,8) - Rational(1,8)*exp(-4*t)'
    x3 = 'Rational(1,8) - Rational(1,8)*exp(-4*t)'
    x4 = 'Rational(3,8) - Rational(1,4)*exp(-(2+2*I)*t)- Rational(1,4)*exp(-(2-2*I)*t) + Rational(1,8)*exp(-4*t)'
    f3 = x2 + '-(' + x1 + ')*(' + x1 + ')'

    a1 = 'Rational(3,8) +  Rational(1,2)*exp(-2*t)*cos(2*t) + Rational(1,8)*exp(-4*t)'
    a2 = 'Rational(1,8) - Rational(1,8)*exp(-4*t)'
    f3_prime = a2 + '-(' + a1 + ')*(' + a1 + ')'
    # print(latex(eval(f3).simplify()))

    add_add = '0.375 - 0.125*exp(-4*t)'
    sub_sub = '0.375 - 0.125*exp(-4*t)'
    sub_add = '1/8 + exp(-4*t)/8'
    add_sub = '1/8 + exp(-4*t)/8'
    f4 = sub_add + '-(' + add_add + ')*(' + add_add + ')'
    print((eval(f3).simplify()))

    RRI = RealRootIsolate('t', '-6.1875 - 3.25*exp(-2*t)*cos(2*t) + 0.25*exp(-4*t) - 0.75*exp(-6*t)*cos(2*t) - 0.0625*exp(-8*t)', [0, 12], 2)
    # RRI = RealRootIsolate('t', f2, [1/1000, 6], 2)
    RRI.RealRootIsolation(RRI, RRI.invl)
    print(RRI.solution)
