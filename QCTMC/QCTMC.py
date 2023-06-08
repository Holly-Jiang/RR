import math
import profile
import random
import re
import time
import psutil
import os
from sympy.matrices import Matrix, Identity
from sympy import sqrt, I, Rational, shape, exp, symbols, cos, E, sin, latex, trace, Interval
from sympy.physics.quantum import TensorProduct
import numpy as np
import sys
from ConflictDrivenSolving import Q_poly_sub, ConflictDrivenSolving, IntervalMinusSet
from random_expr import random_expression
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

        self.variable = symbols('t')  # requires that the corresponding input variable is 't'

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
        ZeroVector = Matrix([np.zeros(dim)])
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
        sum = Matrix([np.zeros(self.dimension * self.dimension)]).T
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

    def Sol2LME(self):
        '''
        :return: density operator (in matrix form)
        '''
        ini_state_vector = self.L2V(self.ini_state)
        governMat_X_XH = Matrix(
            [[1 / 8 * (4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 0, 0,
              1 / 8 * (8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) - exp(
                  -4 * self.variable) + 1), 0, 1 / 8 - exp(-4 * self.variable) / 8, 1 / 8 - exp(-4 * self.variable) / 8,
              0, 0, 1 / 8 - exp(-4 * self.variable) / 8, 1 / 8 - exp(-4 * self.variable) / 8, 0, 1 / 8 * (
                      -8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) - exp(
                  -4 * self.variable) + 1), 0, 0,
              1 / 8 * (-4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3)], [
                 0, 1 / 4 * (exp(-2 * self.variable) * (2 * I * self.variable + 3) + exp(-2 * I * self.variable)),
                    1 / 4 * (exp(-2 * self.variable) * (2 * I * self.variable + 1) - exp(-2 * I * self.variable)), 0,
                    -1 / 2 * exp(-2 * self.variable) * self.variable, 0, 0,
                    1 / 2 * exp(-2 * self.variable) * self.variable, -1 / 2 * exp(-2 * self.variable) * self.variable,
                 0, 0, 1 / 2 * exp(-2 * self.variable) * self.variable, 0,
                    1 / 4 * (exp(-2 * self.variable) * (-2 * I * self.variable - 1) + exp(-2 * I * self.variable)),
                    1 / 4 * (exp(-2 * self.variable) * (1 - 2 * I * self.variable) - exp(-2 * I * self.variable)), 0], [
                 0, 1 / 4 * (exp(-2 * self.variable) * (2 * I * self.variable + 1) - exp(-2 * I * self.variable)),
                    1 / 4 * (exp(-2 * self.variable) * (2 * I * self.variable + 3) + exp(-2 * I * self.variable)), 0,
                    -1 / 2 * exp(-2 * self.variable) * self.variable, 0, 0,
                    1 / 2 * exp(-2 * self.variable) * self.variable, -1 / 2 * exp(-2 * self.variable) * self.variable,
                 0, 0, 1 / 2 * exp(-2 * self.variable) * self.variable, 0,
                    1 / 4 * (exp(-2 * self.variable) * (1 - 2 * I * self.variable) - exp(-2 * I * self.variable)),
                    1 / 4 * (exp(-2 * self.variable) * (-2 * I * self.variable - 1) + exp(-2 * I * self.variable)), 0],
             [
                 1 / 8 * (8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) - exp(
                     -4 * self.variable) + 1), 0, 0,
                 1 / 8 * (4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 0,
                 1 / 8 * (-1 + exp(-4 * self.variable)), 1 / 8 * (-1 + exp(-4 * self.variable)), 0, 0,
                 1 / 8 * (-1 + exp(-4 * self.variable)), 1 / 8 * (-1 + exp(-4 * self.variable)), 0,
                 1 / 8 * (-4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 0, 0,
                 1 / 8 * (-8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) - exp(
                     -4 * self.variable) + 1)], [
                 0, -1 / 2 * exp(-2 * self.variable) * self.variable, -1 / 2 * exp(-2 * self.variable) * self.variable,
                 0, 1 / 4 * exp(-2 * self.variable) * (-2 * I * self.variable + exp((2 + 2 * I) * self.variable) + 3),
                 0, 0, 1 / 4 * exp(-2 * self.variable) * (2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1),
                    -1 / 4 * exp(-2 * self.variable) * (2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1),
                 0, 0,
                    -1 / 4 * exp(-2 * self.variable) * (-2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1),
                 0, 1 / 2 * exp(-2 * self.variable) * self.variable, 1 / 2 * exp(-2 * self.variable) * self.variable,
                 0], [
                 1 / 8 - exp(-4 * self.variable) / 8, 0, 0, 1 / 8 * (-1 + exp(-4 * self.variable)), 0,
                 1 / 8 * (4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 1 / 8 * (
                         8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) + exp(
                     -4 * self.variable) - 1), 0, 0, 1 / 8 * (
                         -8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) + exp(
                     -4 * self.variable) - 1),
                 1 / 8 * (-4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 0,
                 1 / 8 * (-1 + exp(-4 * self.variable)), 0, 0, 1 / 8 - exp(-4 * self.variable) / 8], [
                 1 / 8 - exp(-4 * self.variable) / 8, 0, 0, 1 / 8 * (-1 + exp(-4 * self.variable)), 0, 1 / 8 * (
                        8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) + exp(
                    -4 * self.variable) - 1),
                 1 / 8 * (4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 0, 0,
                 1 / 8 * (-4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3),
                 1 / 8 * (-8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) + exp(
                     -4 * self.variable) - 1), 0, 1 / 8 * (-1 + exp(-4 * self.variable)), 0, 0,
                 1 / 8 - exp(-4 * self.variable) / 8], [
                 0, 1 / 2 * exp(-2 * self.variable) * self.variable, 1 / 2 * exp(-2 * self.variable) * self.variable, 0,
                    1 / 4 * exp(-2 * self.variable) * (2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1), 0,
                 0, 1 / 4 * exp(-2 * self.variable) * (-2 * I * self.variable + exp((2 + 2 * I) * self.variable) + 3),
                    -1 / 4 * exp(-2 * self.variable) * (-2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1),
                 0, 0,
                    -1 / 4 * exp(-2 * self.variable) * (2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1),
                 0, -1 / 2 * exp(-2 * self.variable) * self.variable, -1 / 2 * exp(-2 * self.variable) * self.variable,
                 0], [
                 0, -1 / 2 * exp(-2 * self.variable) * self.variable, -1 / 2 * exp(-2 * self.variable) * self.variable,
                 0, -1 / 4 * exp(-2 * self.variable) * (2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1),
                 0, 0,
                    -1 / 4 * exp(-2 * self.variable) * (-2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1),
                    1 / 4 * exp(-2 * self.variable) * (-2 * I * self.variable + exp((2 + 2 * I) * self.variable) + 3),
                 0, 0, 1 / 4 * exp(-2 * self.variable) * (2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1),
                 0, 1 / 2 * exp(-2 * self.variable) * self.variable, 1 / 2 * exp(-2 * self.variable) * self.variable,
                 0], [
                 1 / 8 - exp(-4 * self.variable) / 8, 0, 0, 1 / 8 * (-1 + exp(-4 * self.variable)), 0, 1 / 8 * (
                        -8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) + exp(
                    -4 * self.variable) - 1),
                 1 / 8 * (-4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 0, 0,
                 1 / 8 * (4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 1 / 8 * (
                         8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) + exp(
                     -4 * self.variable) - 1), 0, 1 / 8 * (-1 + exp(-4 * self.variable)), 0, 0,
                 1 / 8 - exp(-4 * self.variable) / 8], [
                 1 / 8 - exp(-4 * self.variable) / 8, 0, 0, 1 / 8 * (-1 + exp(-4 * self.variable)), 0,
                 1 / 8 * (-4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3),
                 1 / 8 * (-8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) + exp(
                     -4 * self.variable) - 1), 0, 0, 1 / 8 * (
                         8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) + exp(
                     -4 * self.variable) - 1),
                 1 / 8 * (4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 0,
                 1 / 8 * (-1 + exp(-4 * self.variable)), 0, 0, 1 / 8 - exp(-4 * self.variable) / 8], [
                 0, 1 / 2 * exp(-2 * self.variable) * self.variable, 1 / 2 * exp(-2 * self.variable) * self.variable, 0,
                    -1 / 4 * exp(-2 * self.variable) * (-2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1),
                 0, 0,
                    -1 / 4 * exp(-2 * self.variable) * (2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1),
                    1 / 4 * exp(-2 * self.variable) * (2 * I * self.variable + exp((2 + 2 * I) * self.variable) - 1), 0,
                 0, 1 / 4 * exp(-2 * self.variable) * (-2 * I * self.variable + exp((2 + 2 * I) * self.variable) + 3),
                 0, -1 / 2 * exp(-2 * self.variable) * self.variable, -1 / 2 * exp(-2 * self.variable) * self.variable,
                 0], [
                 1 / 8 * (-8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) - exp(
                     -4 * self.variable) + 1), 0, 0,
                 1 / 8 * (-4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 0,
                 1 / 8 * (-1 + exp(-4 * self.variable)), 1 / 8 * (-1 + exp(-4 * self.variable)), 0, 0,
                 1 / 8 * (-1 + exp(-4 * self.variable)), 1 / 8 * (-1 + exp(-4 * self.variable)), 0,
                 1 / 8 * (4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 0, 0,
                 1 / 8 * (8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) - exp(
                     -4 * self.variable) + 1)], [
                 0, 1 / 4 * (exp(-2 * self.variable) * (-2 * I * self.variable - 1) + exp(-2 * I * self.variable)),
                    1 / 4 * (exp(-2 * self.variable) * (1 - 2 * I * self.variable) - exp(-2 * I * self.variable)), 0,
                    1 / 2 * exp(-2 * self.variable) * self.variable, 0, 0,
                    -1 / 2 * exp(-2 * self.variable) * self.variable, 1 / 2 * exp(-2 * self.variable) * self.variable,
                 0, 0, -1 / 2 * exp(-2 * self.variable) * self.variable, 0,
                    1 / 4 * (exp(-2 * self.variable) * (2 * I * self.variable + 3) + exp(-2 * I * self.variable)),
                    1 / 4 * (exp(-2 * self.variable) * (2 * I * self.variable + 1) - exp(-2 * I * self.variable)), 0], [
                 0, 1 / 4 * (exp(-2 * self.variable) * (1 - 2 * I * self.variable) - exp(-2 * I * self.variable)),
                    1 / 4 * (exp(-2 * self.variable) * (-2 * I * self.variable - 1) + exp(-2 * I * self.variable)), 0,
                    1 / 2 * exp(-2 * self.variable) * self.variable, 0, 0,
                    -1 / 2 * exp(-2 * self.variable) * self.variable, 1 / 2 * exp(-2 * self.variable) * self.variable,
                 0, 0, -1 / 2 * exp(-2 * self.variable) * self.variable, 0,
                    1 / 4 * (exp(-2 * self.variable) * (2 * I * self.variable + 1) - exp(-2 * I * self.variable)),
                    1 / 4 * (exp(-2 * self.variable) * (2 * I * self.variable + 3) + exp(-2 * I * self.variable)), 0], [
                 1 / 8 * (-4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3), 0, 0,
                 1 / 8 * (-8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) - exp(
                     -4 * self.variable) + 1), 0, 1 / 8 - exp(-4 * self.variable) / 8,
                 1 / 8 - exp(-4 * self.variable) / 8, 0, 0, 1 / 8 - exp(-4 * self.variable) / 8,
                 1 / 8 - exp(-4 * self.variable) / 8, 0, 1 / 8 * (
                         8 * exp(-2 * self.variable) * I * cos(self.variable) * sin(self.variable) - exp(
                     -4 * self.variable) + 1), 0, 0,
                 1 / 8 * (4 * exp(-2 * self.variable) * cos(2 * self.variable) + exp(-4 * self.variable) + 3)]]
        )
        # governMat_0_IH=Matrix([[1/16*exp(-4*self.variable)*pow((1+3*exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2)],[1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*(3+10*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2)],[1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(3+10*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2)],[1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((3+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2)],[1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(3+10*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2)],[1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((1+3*exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable))],[1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((3+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2)],[1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(3+10*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable))],[1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(3+10*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2)],[1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((3+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2)],[1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((1+3*exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable))],[1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*(3+10*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable))],[1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((3+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2)],[1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*(3+10*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable))],[1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*(-3+2*exp(2*self.variable)+exp(4 *self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(3+10*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable))],[1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),1/16*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((-1+exp(2*self.variable)),2),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),-(1/16)*exp(-4*self.variable)*(-1-2*exp(2*self.variable)+3*exp(4*self.variable)),1/16*exp(-4*self.variable)*pow((1+3*exp(2*self.variable)),2)]])
        # governMat_X_XH = expMt(self.governingMatrix, self.variable)
        # print(latex(governMat_X_XH))
        # print(latex(governMat_X_XH * ini_state_vector))
        solution = self.V2L(governMat_X_XH * ini_state_vector)
        # print(latex(solution))
        return solution


def iso_satisfy(I, J, B, RRI,factor):
    sat_interval = []
    value_t = RRI.f.subs({RRI.t: RRI.invl[0]})
    if len(RRI.solution) == 0:
        if (factor[1] == '>' and value_t > 0) or (factor[1] == '>=' and value_t >= 0) or (
                factor[1] == '<' and value_t < 0) or (factor[1] == '<=' and value_t <= 0):
            sat_interval.append([B[0] - J[1], B[1] - J[0]])
    else:
        minu_invl = IntervalMinusSet([B], RRI.solution)
        for invl in minu_invl:
            value_t = RRI.f.subs({RRI.t: invl[0] + (invl[1] - invl[0]) / 2})
            if (factor[1] == '>' and value_t > 0) or (factor[1] == '>=' and value_t >= 0) or (
                    factor[1] == '<' and value_t < 0) or (factor[1] == '<=' and value_t <= 0):
                sat_interval.append([invl[0] - J[1], invl[1] - J[0]])
    if IntervalMinusSet([I], sat_interval) is None:
        return True
    return False


def qctmc_rho_t():
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
    rho_t = qctmc.Sol2LME()
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
    print(f'qctmc_time: {time.time() - start}')
    return rho_t, P, qctmc_time


def isolate_solution(I, J, factor, rho_t, P, interval, qctmc_time):
    iso_cache = 0
    iso_time = 0
    print(f'factor: {factor}')
    start = time.time()
    f3_prime = str(Q_poly_sub(factor[0], P, rho_t).simplify())
    if not (len(re.findall(r'exp\(.\d*.t\)', f3_prime)) > 0 or len(re.findall(r'cos\(.\d*.t\)', f3_prime)) > 0):
        return 0, 0, False, []
    print(f'phi: {f3_prime}')
    print(f'interval:{interval}')
    print('**************realrootisolate**********************')
    RRI = RealRootIsolate('t', f3_prime, interval, 2)
    RRI.RealRootIsolation(RRI, RRI.invl)
    iso_cache = (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    if iso_cache<RRI.cache:
        iso_cache=RRI.cache
    value = 0
    if len(RRI.solution) > 0:
        print(f'ok isolate: {RRI.solution}')
    else:
        print(f'no root isolate: {RRI.solution}')
    if iso_satisfy(I, J, interval, RRI, factor):
        value=1
        print(f'iso satisfy')
    else:
        print(f'iso not satisfy')
    print(f'iso 当前进程的内存使用：%.4f MB' % (iso_cache))
    iso_time = time.time() - start + qctmc_time
    print(f'realrootisolate time:{iso_time}')
    print('************************************', iso_time, iso_cache, value, RRI.solution)
    return iso_time, iso_cache, value, RRI.solution


def conflict_solution(factor, rho_t, P, box_I, diamond_J, qctmc_time):
    start = time.time()
    # factor=['x2-x1**2', '==', '0']
    phi = factor[0]
    print(f'factor: {factor}')

    interval = Interval(int(factor[2]), 9999, left_open=True, right_open=True)
    if factor[1].__eq__('>'):
        interval = Interval(int(factor[2]), 9999, left_open=True, right_open=True)
    elif factor[1].__eq__('>='):
        interval = Interval(int(factor[2]), 9999, left_open=False, right_open=True)
    elif factor[1].__eq__('<'):
        interval = Interval(-9999, int(factor[2]), left_open=True, right_open=True)
    elif factor[1].__eq__('<='):
        interval = Interval(-9999, int(factor[2]), left_open=True, right_open=False)
    elif factor[1].__eq__('=='):
        interval = Interval(int(factor[2]), int(factor[2]), left_open=False, right_open=False)
    Phi = {'projector': P, 'poly': phi, 'interval': interval}
    # print(f'Phi:{Phi}')
    rr = [box_I, diamond_J, Phi]

    print('***************ConflictDrivenSolving*********************')
    # print(f'I:{box_I}, J:{diamond_J}')
    T = ConflictDrivenSolving(rho_t, rr)

    print('************************************')
    cache = (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    conflict_cache = cache
    print(u'conflict 当前进程的内存使用：%.4f MB' % (cache))
    conflict_time = time.time() - start + qctmc_time
    print(f'conflictdriving time:{time.time() - start + qctmc_time}')
    return conflict_time, conflict_cache, T


if __name__ == '__main__':
    start = time.time()
    iso_time = 0
    conflict_time = 0
    instance = 5
    conflict_cache = 0
    iso_cache = 0
    rho_t, P, qctmc_time = qctmc_rho_t()
    count = 0
    root_count = random.randint(2, 3)
    # root_count = 2
    print(f'root_count: {root_count}')
    degree = int(sys.argv[1])
    height_from = int(sys.argv[2])
    height_to = int(sys.argv[3])
    Ileft = 1
    Iinter1 = 10
    Jinter1 = 5
    Jleft = 0.5
    while count < root_count:
        exp, cnf_exp, cnf_exp_list = random_expression(1, 1, 1, degree, height_from, height_to, 10)
        factor = exp[0].split(' ')
        Iinter = random.randint(1, Iinter1)
        Jinter = random.randint(1, Jinter1)
        # boxleft=random.randint(0, Ileft)
        # dialeft=random.randint(0, Jleft)
        boxleft = round(random.uniform(0, Ileft), 1)
        dialeft = round(random.uniform(0, Jleft), 1)
        I = [boxleft, boxleft + Iinter]
        J = [dialeft, dialeft + Jinter]
        inf_I = I[0]
        sup_I = I[1]
        inf_J = J[0]
        sup_J = J[1]
        B = [inf_I + inf_J, sup_I + sup_J]
        print(f'I: {I}, J: {J}')

        iso_time1, iso_cache1, value, solution = isolate_solution(factor, rho_t, P, B, qctmc_time)
        if not solution:
            count += 1
            iso_time += iso_time1
            iso_cache += iso_cache1
        else:
            continue
        conflict_time, conflict_cache, T = conflict_solution(factor, rho_t, P, I, J, qctmc_time, conflict_time,
                                                             conflict_cache)
        print('time:', time.time() - start)
    for i in range(5 - root_count):
        exp, cnf_exp, cnf_exp_list = random_expression(1, 1, 1, degree, height_from, height_to, 10)
        factor = exp[0].split(' ')
        Iinter = random.randint(1, Iinter1)
        Jinter = random.randint(1, Jinter1)
        boxleft = round(random.uniform(0, Ileft), 1)
        dialeft = round(random.uniform(0, Jleft), 1)
        I = [boxleft, boxleft + Iinter]
        J = [dialeft, dialeft + Jinter]
        inf_I = I[0]
        sup_I = I[1]
        inf_J = J[0]
        sup_J = J[1]
        print(f'I: {I}, J: {J}')
        B = [inf_I + inf_J, sup_I + sup_J]
        iso_time1, iso_cache1, value, solution = isolate_solution(factor, rho_t, P, B, qctmc_time)
        iso_time += iso_time1
        iso_cache += iso_cache1
        conflict_time, conflict_cache, T = conflict_solution(factor, rho_t, P, I, J, qctmc_time, conflict_time,
                                                             conflict_cache)
        print('time:', time.time() - start)

    print(f'average time｜ isolate: {iso_time / instance}, conflict: {conflict_time / instance}')
    print(f'average space｜ isolate: {iso_cache / instance}, conflict: {conflict_cache / instance}')
