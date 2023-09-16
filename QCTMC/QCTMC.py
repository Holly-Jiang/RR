import math
import profile
import random
import re
import time
import psutil
import os

import sympy
from sympy.matrices import Matrix, Identity
from sympy import sqrt, I, Rational, shape, exp, symbols, cos, E, sin, latex, trace, Interval, Transpose, simplify, \
    expand, zeros
from sympy.physics.quantum import TensorProduct
import numpy as np
import sys
from ConflictDrivenSolving import Q_poly_sub, ConflictDrivenSolving, IntervalMinusSet, FactorPolynomial, \
    IntervalMinusCloseSet
from RealRootIsolate import RealRootIsolate

sys.path.append('../')


class QCTMC:
    def __init__(self, dim, num_states, labels, Hermitian_Operator, Linear_Operators, ini_state=None):
        # ini_state : D_[H_[cq]]
        # Hermitian_Operator : Matrix
        # Linear_Operators : list of Matrices
        # ini_state : Matrix

        self.dimension = dim
        self.num_states = num_states
        self.labels = labels
        self.Hermitian_Operator = Hermitian_Operator
        self.Linear_Operators = Linear_Operators
        self.ini_state = ini_state
        self.governingMatrix = self.GoverningMatrix()

        self.variable = symbols('t', real=True)  # requires that the corresponding input variable is 't'

    def SimplifyMatrixEle(self, matrix):
        for row in range(0, shape(matrix)[0]):
            for col in range(0, shape(matrix)[1]):
                matrix[row, col] = matrix[row, col].simplify()
        return matrix

    def GoverningMatrix(self):
        governingMatrix = -I * TensorProduct(self.Hermitian_Operator, Matrix(Identity(self.dimension))) + \
                          I * TensorProduct(Matrix(Identity(self.dimension)), self.Hermitian_Operator.T)
        for linear_operator in self.Linear_Operators:
            governingMatrix += TensorProduct(linear_operator, linear_operator.conjugate()) - \
                               Rational(1, 2) * TensorProduct(linear_operator.T.conjugate() * linear_operator,
                                                              Matrix(Identity(self.dimension))) - \
                               Rational(1, 2) * TensorProduct(Matrix(Identity(self.dimension)),
                                                              linear_operator.T * linear_operator.conjugate())
        returnMatrix = self.SimplifyMatrixEle(governingMatrix)
        return returnMatrix

    def KetI(self, ind, dim):
        # return_type: vector in the form of matrix
        vec_state = [0] * dim
        vec_state[ind] = 1
        res = Matrix(np.array(vec_state)[None, :]).T
        return res

    def ZeroMatrix(self, dim):
        # return_type: Matrix
        ZeroVector = zeros(1,dim)
        ZeroMatrix = ZeroVector.T * ZeroVector
        return ZeroMatrix

    def trace(self, mat):
        dim = shape(mat)[0]
        returnTrace = 0
        for i in range(0, dim):
            returnTrace += mat[i, i]
        return returnTrace

    def L2V(self, LinearMat):
        # LinerMat_type: Matrix
        # return_type: column vector in the form of Matrix
        dim_LM = shape(LinearMat)[0]
        if dim_LM != self.dimension:
            raise Exception('The dimension of linear matrix is not matched, please check!')
        # sum = Matrix([np.zeros(self.dimension * self.dimension)]).T
        sum=zeros(self.dimension * self.dimension,1)
        for i in range(0, self.dimension):
            ketI = self.KetI(i, self.dimension)
            sum += Matrix(TensorProduct(ketI, ketI))
        vector = TensorProduct(LinearMat, Matrix(Identity(self.dimension))) * sum
        return vector

    def V2L(self, vector):
        # vector_type: column vector in the form of Matrix
        # return_type: Matrix
        vector = vector.T  # row vector
        dim_vec = shape(vector)[1]
        if dim_vec != self.dimension * self.dimension:
            raise Exception('The dimension of vector is not matched, please check!')
        returnMatrix = self.ZeroMatrix(self.dimension)
        for i in range(0, self.dimension):
            for j in range(0, self.dimension):
                traceM = self.trace(TensorProduct(self.KetI(i, self.dimension), self.KetI(j, self.dimension)) * vector)
                returnMatrix[i, j] = traceM.simplify()
        return returnMatrix

    def Sol2LME(self,is_compute):
        '''
        :return: density operator (in matrix form)
        '''
        ini_state_vector = self.L2V(self.ini_state)
        if is_compute==1:
            governMat_X_XH = expand(simplify(exp(self.governingMatrix*self.variable)))
            f = open('./EXPM.txt', 'w')
            f.write(governMat_X_XH.__str__())
            f.close()
        elif is_compute==3:
            matrix ='Matrix([[(3*exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (exp(8*t) - exp(4*t) - 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 1/8 - exp(-4*t)/8, 1/8 - exp(-4*t)/8, 0, 0, 1/8 - exp(-4*t)/8, 1/8 - exp(-4*t)/8, 0, (exp(8*t) - exp(4*t) + 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (3*exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8], [0, I*t*exp(-2*t)/2 + exp(-2*I*t)/4 + 3*exp(-2*t)/4, I*t*exp(-2*t)/2 - exp(-2*I*t)/4 + exp(-2*t)/4, 0, -t*exp(-2*t)/2, 0, 0, t*exp(-2*t)/2, -t*exp(-2*t)/2, 0, 0, t*exp(-2*t)/2, 0, -I*t*exp(-2*t)/2 + exp(-2*I*t)/4 - exp(-2*t)/4, -I*t*exp(-2*t)/2 - exp(-2*I*t)/4 + exp(-2*t)/4, 0], [0, I*t*exp(-2*t)/2 - exp(-2*I*t)/4 + exp(-2*t)/4, I*t*exp(-2*t)/2 + exp(-2*I*t)/4 + 3*exp(-2*t)/4, 0, -t*exp(-2*t)/2, 0, 0, t*exp(-2*t)/2, -t*exp(-2*t)/2, 0, 0, t*exp(-2*t)/2, 0, -I*t*exp(-2*t)/2 - exp(-2*I*t)/4 + exp(-2*t)/4, -I*t*exp(-2*t)/2 + exp(-2*I*t)/4 - exp(-2*t)/4, 0], [(exp(8*t) - exp(4*t) - 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (3*exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, -1/8 + exp(-4*t)/8, -1/8 + exp(-4*t)/8, 0, 0, -1/8 + exp(-4*t)/8, -1/8 + exp(-4*t)/8, 0, (3*exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (exp(8*t) - exp(4*t) + 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8], [0, -t*exp(-2*t)/2, -t*exp(-2*t)/2, 0, (-2*I*t + exp(2*t*(1 + I)) + 3)*exp(-2*t)/4, 0, 0, (2*I*t + exp(2*t*(1 + I)) - 1)*exp(-2*t)/4, (-2*I*t - exp(2*t*(1 + I)) + 1)*exp(-2*t)/4, 0, 0, (2*I*t - exp(2*t*(1 + I)) + 1)*exp(-2*t)/4, 0, t*exp(-2*t)/2, t*exp(-2*t)/2, 0], [1/8 - exp(-4*t)/8, 0, 0, -1/8 + exp(-4*t)/8, 0, (3*exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, (-exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (-exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, (3*exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, -1/8 + exp(-4*t)/8, 0, 0, 1/8 - exp(-4*t)/8], [1/8 - exp(-4*t)/8, 0, 0, -1/8 + exp(-4*t)/8, 0, (-exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, (3*exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (3*exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, (-exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, -1/8 + exp(-4*t)/8, 0, 0, 1/8 - exp(-4*t)/8], [0, t*exp(-2*t)/2, t*exp(-2*t)/2, 0, (2*I*t + exp(2*t*(1 + I)) - 1)*exp(-2*t)/4, 0, 0, (-2*I*t + exp(2*t*(1 + I)) + 3)*exp(-2*t)/4, (2*I*t - exp(2*t*(1 + I)) + 1)*exp(-2*t)/4, 0, 0, (-2*I*t - exp(2*t*(1 + I)) + 1)*exp(-2*t)/4, 0, -t*exp(-2*t)/2, -t*exp(-2*t)/2, 0], [0, -t*exp(-2*t)/2, -t*exp(-2*t)/2, 0, (-2*I*t - exp(2*t*(1 + I)) + 1)*exp(-2*t)/4, 0, 0, (2*I*t - exp(2*t*(1 + I)) + 1)*exp(-2*t)/4, (-2*I*t + exp(2*t*(1 + I)) + 3)*exp(-2*t)/4, 0, 0, (2*I*t + exp(2*t*(1 + I)) - 1)*exp(-2*t)/4, 0, t*exp(-2*t)/2, t*exp(-2*t)/2, 0], [1/8 - exp(-4*t)/8, 0, 0, -1/8 + exp(-4*t)/8, 0, (-exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, (3*exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (3*exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, (-exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, -1/8 + exp(-4*t)/8, 0, 0, 1/8 - exp(-4*t)/8], [1/8 - exp(-4*t)/8, 0, 0, -1/8 + exp(-4*t)/8, 0, (3*exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, (-exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (-exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, (3*exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, -1/8 + exp(-4*t)/8, 0, 0, 1/8 - exp(-4*t)/8], [0, t*exp(-2*t)/2, t*exp(-2*t)/2, 0, (2*I*t - exp(2*t*(1 + I)) + 1)*exp(-2*t)/4, 0, 0, (-2*I*t - exp(2*t*(1 + I)) + 1)*exp(-2*t)/4, (2*I*t + exp(2*t*(1 + I)) - 1)*exp(-2*t)/4, 0, 0, (-2*I*t + exp(2*t*(1 + I)) + 3)*exp(-2*t)/4, 0, -t*exp(-2*t)/2, -t*exp(-2*t)/2, 0], [(exp(8*t) - exp(4*t) + 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (3*exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, -1/8 + exp(-4*t)/8, -1/8 + exp(-4*t)/8, 0, 0, -1/8 + exp(-4*t)/8, -1/8 + exp(-4*t)/8, 0, (3*exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (exp(8*t) - exp(4*t) - 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8], [0, -I*t*exp(-2*t)/2 + exp(-2*I*t)/4 - exp(-2*t)/4, -I*t*exp(-2*t)/2 - exp(-2*I*t)/4 + exp(-2*t)/4, 0, t*exp(-2*t)/2, 0, 0, -t*exp(-2*t)/2, t*exp(-2*t)/2, 0, 0, -t*exp(-2*t)/2, 0, I*t*exp(-2*t)/2 + exp(-2*I*t)/4 + 3*exp(-2*t)/4, I*t*exp(-2*t)/2 - exp(-2*I*t)/4 + exp(-2*t)/4, 0], [0, -I*t*exp(-2*t)/2 - exp(-2*I*t)/4 + exp(-2*t)/4, -I*t*exp(-2*t)/2 + exp(-2*I*t)/4 - exp(-2*t)/4, 0, t*exp(-2*t)/2, 0, 0, -t*exp(-2*t)/2, t*exp(-2*t)/2, 0, 0, -t*exp(-2*t)/2, 0, I*t*exp(-2*t)/2 - exp(-2*I*t)/4 + exp(-2*t)/4, I*t*exp(-2*t)/2 + exp(-2*I*t)/4 + 3*exp(-2*t)/4, 0], [(3*exp(8*t) + exp(4*t) - 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (exp(8*t) - exp(4*t) + 2*exp(2*t*(3 - I)) - 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 1/8 - exp(-4*t)/8, 1/8 - exp(-4*t)/8, 0, 0, 1/8 - exp(-4*t)/8, 1/8 - exp(-4*t)/8, 0, (exp(8*t) - exp(4*t) - 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8, 0, 0, (3*exp(8*t) + exp(4*t) + 2*exp(2*t*(3 - I)) + 2*exp(2*t*(3 + I)))*exp(-8*t)/8]])'
            matrix = matrix.replace('*t', "*self.variable")
            matrix = matrix.replace(' t', " self.variable")
            matrix = matrix.replace('-t', "-self.variable")
            matrix = matrix.replace('(t', "(self.variable")
            governMat_X_XH = expand(eval(matrix), complex=True)
        else:
            f = open('./EXPM.txt', 'r')
            matrix=f.readline()
            matrix=matrix.replace('*t',"*self.variable")
            matrix=matrix.replace(' t', " self.variable")
            matrix=matrix.replace('-t', "-self.variable")
            matrix=matrix.replace('(t', "(self.variable")
            governMat_X_XH = expand(eval(matrix), complex=True)
            f.close()
        # print('governMat_X_XH:', governMat_X_XH)
        solution = self.V2L(governMat_X_XH * ini_state_vector)
        # print(latex(solution))
        return solution


def iso_satisfy(I, J, B, RRI, factor):
    sat_interval = []
    value_t = simplify(RRI.f.subs({RRI.t: RRI.invl[0]}))
    if len(RRI.solution) == 0:
        if (factor[1] == '>' and value_t > 0) or (factor[1] == '>=' and value_t >= 0) or (
                factor[1] == '<' and value_t < 0) or (factor[1] == '<=' and value_t <= 0):
            sat_interval.append([B[0] - J[1], B[1] - J[0]])
    else:
        minu_invl = IntervalMinusCloseSet([B], RRI.solution)
        for invl in minu_invl:
            value_t = RRI.f.subs({RRI.t: invl[0] + (invl[1] - invl[0]) / 2})
            if (factor[1] == '>' and value_t > 0) or (factor[1] == '>=' and value_t >= 0) or (
                    factor[1] == '<' and value_t < 0) or (factor[1] == '<=' and value_t <= 0):
                sat_interval.append([invl[0] - J[1], invl[1] - J[0]])
    if IntervalMinusCloseSet([I], sat_interval) is None:
        return True
    return False


def qctmc_rho_t(is_compute):
    start = time.time()
    dim = 4
    num_states = 6
    labels = ''
    Hadamard = sqrt(2) * Rational(1, 2) * Matrix([[1, 1], [1, -1]])
    # Hermitian_Operator = Matrix([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    Hermitian_Operator = Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    I_H = TensorProduct(Matrix(Identity(2)), Hadamard)
    H_I = TensorProduct(Hadamard, Matrix(Identity(2)))
    H_H = TensorProduct(Hadamard, Hadamard)
    X_H = TensorProduct((sqrt(2) * Rational(1, 2)) * Matrix([[1, -1], [1, 1]]),
                        (sqrt(2) * Rational(1, 2)) * Matrix([[1, -1], [1, 1]]))
    H_X = TensorProduct((sqrt(2) * Rational(1, 2)) * Matrix([[1, 1], [-1, 1]]),
                        (sqrt(2) * Rational(1, 2)) * Matrix([[1, 1], [-1, 1]]))
    CX = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    X = Matrix([[0, 1], [1, 0]])
    X_X = TensorProduct(X, X)
    # print(X_X)
    Linear_Operators = [X_H, H_X]
    Hermitian_Operator = X_X

    ket_0 = Matrix([[1, 0], [0, 0]])
    ket_1 = Matrix([[0, 0], [0, 1]])
    ket_0_0 = TensorProduct(ket_0, ket_0)
    ket_1_1 = TensorProduct(ket_1, ket_1)
    ket_0_1 = TensorProduct(ket_0, ket_1)
    ket_1_0 = TensorProduct(ket_1, ket_0)

    ket_add_0 = TensorProduct(Rational(1, 2) * Matrix([[1, 1], [1, 1]]), Matrix([[1, 0], [0, 0]]))
    ket_add_1 = TensorProduct(Rational(1, 2) * Matrix([[1, 1], [1, 1]]), Matrix([[0, 0], [0, 1]]))
    ket_sub_0 = TensorProduct(Rational(1, 2) * Matrix([[1, -1], [-1, 1]]), Matrix([[1, 0], [0, 0]]))
    ket_sub_1 = TensorProduct(Rational(1, 2) * Matrix([[1, -1], [-1, 1]]), Matrix([[0, 0], [0, 1]]))
    ini_state = Matrix([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    qctmc = QCTMC(dim, num_states, labels, Hermitian_Operator, Linear_Operators, ini_state=ini_state)
    # print('governingMatrix: ', qctmc.governingMatrix)
    # print('latex(governingMatrix): ', latex(qctmc.governingMatrix))
    rho_t = qctmc.Sol2LME(is_compute)
    # print("rho:", rho_t)
    # print("rho——latex:", latex(rho_t))
    # print(latex(trace(ket_add_0 * rho_t)))
    # print(latex(trace(ket_add_1 * rho_t)))
    # print(latex(trace(ket_sub_0 * rho_t)))
    # print(latex(trace(ket_sub_1 * rho_t)))
    #
    # print((trace(ket_0_0 * rho_t)))
    # print((trace(ket_0_1 * rho_t)))
    # print((trace(ket_1_0 * rho_t)))
    # print((trace(ket_1_1 * rho_t)))
    P = [ket_0_0, ket_0_1, ket_1_0, ket_1_1]
    qctmc_time = time.time() - start
    print(f'QCTMC_TIME: {time.time() - start}')
    return rho_t, P, qctmc_time


def isolate_solution(I, J, factor, rho_t, P, interval, qctmc_time):
    print(f'FACTOR: {factor}')
    start = time.time()
    f3_prime = str(Q_poly_sub(factor[0], P, rho_t).simplify())
    if not (len(re.findall(r'exp\(.\d*.t\)', f3_prime)) > 0 or len(re.findall(r'cos\(.\d*.t\)', f3_prime)) > 0):
        return 0, 0, False, []
    print(f'PHI: {f3_prime}')
    print(f'INTERVAL: {interval}')
    print('**************ISOLATE**********************')
    RRI = RealRootIsolate('t', f3_prime, interval, 10)
    RRI.RealRootIsolation(RRI, RRI.invl)
    iso_cache = (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    if iso_cache < RRI.cache:
        iso_cache = RRI.cache
    value = 0
    if len(RRI.solution) > 0:
        print(f'OK ISOLATE: {RRI.solution}')
    else:
        print(f'NO ROOT ISOLATE: {RRI.solution}')
    if iso_satisfy(I, J, interval, RRI, factor):
        value = 1
        print(f'ISOLATE SATISFIED')
    else:
        print(f'ISOLATE NOT SATISFIED')
    print(f'ISOLATE CACHE USED: %.4f MB' % (iso_cache))
    iso_time = time.time() - start + qctmc_time
    print(f'ISOLATE TIME: {iso_time}')
    print('************************************', iso_time, iso_cache, value, RRI.solution)
    return iso_time, iso_cache, value, RRI.solution


def conflict_solution(factor, rho_t, P, box_I, diamond_J, qctmc_time):
    start = time.time()
    # factor=['x2-x1**2', '==', '0']
    phi = factor[0]
    print(f'FACTOR: {factor}')

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
    Phi = {'projector': P, 'poly': phi, 'interval': interval}
    # print(f'Phi:{Phi}')
    rr = [box_I, diamond_J, Phi]
    Q_poly = Q_poly_sub(factor[0], P, rho_t).simplify()
    print(f'PHI: {str(Q_poly)}')
    print('***************SAMPLE-DRIVEN*********************')
    # print(f'I:{box_I}, J:{diamond_J}')
    if len(interval.args) == 1:
        phi_t = (Q_poly - interval.args[0]) * (Q_poly - interval.args[0])
    else:
        if interval.start == 99999999999999 or interval.start == -99999999999999:
            phi_t = (Q_poly - interval.end)
        elif interval.end == 99999999999999 or interval.end == -99999999999999:
            phi_t = (Q_poly - interval.start)
        else:
            phi_t = (Q_poly - interval.start) * (Q_poly - interval.end)
    phi_t=expand(phi_t)
    factors = FactorPolynomial(str(phi_t))
    T = ConflictDrivenSolving(rho_t, rr,phi_t,factors)

    print('************************************')
    cache = (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    conflict_cache = cache
    print(u'SAMPLE-DRIVEN CACHE USED: %.4f MB' % (cache))
    conflict_time = time.time() - start + qctmc_time
    print(f'SAMPLE-DRIVEN TIME: {time.time() - start + qctmc_time}')
    return round(conflict_time,2), round(conflict_cache,2), T


if __name__ == '__main__':

    start = time.time()

    rho_t, P, qctmc_time = qctmc_rho_t(0)
    # phi='-1/64-exp(-8*t)/64-7*exp(-4*t)/32-1/8*exp(-6*t)*cos(2*t)-3/8*exp(-2*t)*cos(2*t)-1/4*exp(-4*t)*cos(2*t)*cos(2*t)'
    # phi='-1/64-3/16*exp(-(2+2*I)*t)-3/16*exp(-(2-2*I)*t)-1/16*exp(-(4+4*I)*t)-1/16*exp(-(4-4*I)*t)-1/16*exp(-(6+2*I)*t)-1/16*exp(-(6-2*I)*t)-11/32*exp(-4*t)-1/64*exp(-8*t)'
    # phi='-(3*exp(4*t) + 4*exp(2*t)*cos(2*t) + 1)**2*exp(-8*t)/64 + 1/8 - exp(-4*t)/8'
    # phi = '2.015625 - 4.6875*exp(-2*t)*cos(2*t) + 0.78125*exp(-4*t) + 0.6875*exp(-6*t)*cos(2*t) + 0.203125*exp(-8*t)'
    t=symbols('t', real=True)
    # f=eval(phi)
    # i=0
    # while i<=3:
    #     value=f.subs({t: i})
    #     if value>-0.01:
    #         print(f'{i:.6f} {value*10-0.3:.6f}')
    #         pass
    #     else:
    #         # print(f'{i:.6f} {value-0.09:.6f}')
    #         pass
    #     i+=0.001
    #
    # P = [ket_0_0, ket_0_1, ket_1_0, ket_1_1]
    box_I = [0.0, 6.0]
    diamond_J =  [0.2, 1.2]
    cnf_list=[['683*x2*x4-575*x3*x4+183*x3**2-327*x1*x4-789*x1**2-68*x4**2+233*x1*x3+961*x1**2-638*x2*x3+141*x2**2+432*x1*x2 >= 0']]
    B = [box_I[0] + diamond_J[0], box_I[1] + diamond_J[1]]
    for cnf in cnf_list:
        for phi in cnf:
            factor=phi.split(' ')
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
            Phi = {'projector': P, 'poly': factor[0], 'interval': interval}
            rr = [box_I, diamond_J, Phi]
            Q_poly = Q_poly_sub(factor[0], P, rho_t).simplify()
            print(f'PHI: {str(Q_poly)}')
            print('***************SAMPLE-DRIVEN*********************')
            # print(f'I:{box_I}, J:{diamond_J}')
            if len(interval.args) == 1:
                phi_t = (Q_poly - interval.args[0]) * (Q_poly - interval.args[0])
            else:
                if interval.start == 99999999999999 or interval.start == -99999999999999:
                    phi_t = (Q_poly - interval.end)
                elif interval.end == 99999999999999 or interval.end == -99999999999999:
                    phi_t = (Q_poly - interval.start)
                else:
                    phi_t = (Q_poly - interval.start) * (Q_poly - interval.end)
            phi_t = expand(phi_t)
            factors = FactorPolynomial(str(phi_t))

            # iso=isolate_solution(box_I, diamond_J, factor, rho_t, P, B, qctmc_time)
            sample=ConflictDrivenSolving(rho_t, rr,phi_t,factors)
            iso=isolate_solution(box_I, diamond_J, factor, rho_t, P, B, qctmc_time)
            if sample!=iso[2]:
                print(f'sample: {sample}  iso:{iso}')
                exit(-1)
    # # B[ind]: [1.56093620410116, 1.56093620410116] delta: 1.05471187339390e-16
    # print(time.time() - start)
    # prepare for the QCTMC data
    # qctmc_rho_t(int(sys.argv[1]))
    pass
# 9*x2**2+10*x3**3+3*x4+1*x2*x3-3*x1**3-5*x3**3-9*x1*x4+10*x1*x2*x4-9*x2*x3*x4-6*x1*x3+6*x2*x4 > 0