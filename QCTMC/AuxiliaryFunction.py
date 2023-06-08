import time
from functools import wraps
from sympy import *
from sympy.calculus.util import *
from sympy.physics.quantum import TensorProduct
import numpy as np


def parse_STL_formula_test(self, formula):
    self.list_ind += 1
    print("parsing....")
    left_formula = None
    right_formula = None
    connect_symbol = None
    head_of_rightformula = 0

    if formula[0] != '(':
        print("here1")
        print("formula:", formula)
        # four cases:
        # 1. Phi
        # 2. ¬Phi | ¬(phi)
        # 3. Phi1 ∧ Phi2 | Phi1 ∧ (phi)
        # 4. Phi1 U[l,u] Phi2 | Phi1 U[l,u] (phi)
        if re.search(r"\(", formula) is None:
            if len(formula) == 1:
                parse_element = self.formula_type([formula])
                self.parse_STL_tree[self.list_ind] = parse_element
            elif formula[0] == '¬':
                parse_element = self.formula_type(['¬', formula[1]])
                self.parse_STL_tree[self.list_ind] = parse_element
            elif formula[1] == '∧':
                parse_element = self.formula_type(['∧', formula[0], formula[2]])
                self.parse_STL_tree[self.list_ind] = parse_element
            elif formula[1] == 'U':
                parse_element = self.formula_type([formula[1:-1], formula[0], formula[-1]])
                self.parse_STL_tree[self.list_ind] = parse_element
            else:
                parse_element = self.formula_type([formula])
                self.parse_STL_tree[self.list_ind] = parse_element
            # self.list_ind += 1
        else:
            if formula[0] == '¬':
                parse_element = self.formula_type(['¬', self.list_ind + 1])
                self.parse_STL_tree[self.list_ind] = parse_element
                right_formula = formula[2:-1]
            elif formula[1] == '∧':
                parse_element = self.formula_type(['∧', formula[0], self.list_ind + 1])
                self.parse_STL_tree[self.list_ind] = parse_element
                right_formula = formula[2:-1]
            elif formula[1] == 'U':
                utime = re.match(r"U\[\d+,\d+\]", formula[1:]).group()
                parse_element = self.formula_type([utime, formula[0], self.list_ind + 1])
                self.parse_STL_tree[self.list_ind] = parse_element
                right_formula = formula[re.search(r"\(", formula).span()[1]:-1]
            # self.list_ind += 1
            self.parse_STL_formula_test(right_formula)

    elif formula[0] == '(':
        print("formula:", formula)
        print("here2")
        match = 0
        right_bracket_position = 0
        for sym in formula:
            if sym == '(':
                match += 1
            elif sym == ')':
                match -= 1
            if match == 0:
                break
            right_bracket_position += 1
        left_formula = formula[1: right_bracket_position]

        reg = r"∧|U\[\d+,\d+\]"
        try:
            connect_symbol = re.match(reg, formula[right_bracket_position + 1:]).group()
        except:
            raise Exception('some syntax errors in your STL formula, please CHECK!')
        connect_symbol_position = re.match(reg, formula[right_bracket_position + 1:]).span()
        head_of_rightformula = right_bracket_position + connect_symbol_position[1] + 1

        cur_formula_num = self.list_ind
        if formula[head_of_rightformula] == '(':

            right_formula = formula[head_of_rightformula + 1: -1]
            print("left_formula:", left_formula, "right_formula:", right_formula)

            # self.list_ind += 1
            self.parse_STL_formula_test(left_formula)

            parse_element = self.formula_type([connect_symbol, cur_formula_num + 1, self.list_ind + 1])
            self.parse_STL_tree[cur_formula_num] = parse_element

            # self.list_ind += 1
            self.parse_STL_formula_test(right_formula)
        else:

            right_formula = formula[head_of_rightformula:]

            print("left_formula:", left_formula, "right_formula:", right_formula)
            self.parse_STL_formula_test(left_formula)

            parse_element = self.formula_type([connect_symbol, cur_formula_num + 1, right_formula])

            self.parse_STL_tree[cur_formula_num] = parse_element
    else:
        raise Exception('some syntax errors in your STL formula, please CHECK!')

class fancy_timed(object):
    def __init__(self, f):
        self.f = f
        self.active = False
        print('instance')

    def __call__(self, *args):
        if self.active:
            print('true')
            return self.f(*args)
        print('here')
        start = time.time()
        self.active = True
        res = self.f(*args)
        end = time.time()
        self.activate = False
        print(end - start)
        return  res, end - start

def timethis(func):
    @wraps(func)
    def wrapper(* args, ** kwargs):
        start = time.time()
        result = func(* args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result
    return wrapper


class test:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.sum = []

    @fancy_timed
    def test1(self, nn):

        if nn == 1:
            return 1
        else:
            self.sum.append(nn)
            return nn * self.test1(self, nn-1)

@fancy_timed
def recursion(n):
    print('test')
    if  n == 1:# 正确的返回添加（结束条件）
        return 1
    else:
        return  n * recursion(n-1)


@timethis
def recursion1(n):
    if  n == 1:# 正确的返回添加（结束条件）
        return 1
    else:
        return  n * recursion1(n-1)

# a = test(5,6)
# b = a.test1(a,3)
# print(b)
# print(a.sum)
# print(recursion(3))

v = symbols('x')
f = eval('-Rational(1,5)-sqrt(2)*Rational(1,2)*exp(-(2+sqrt(2))*Rational(1,2)*v)+'
         'sqrt(2)*Rational(1,2)*exp(-(2-sqrt(2))*Rational(1,2)*v)')
print(f)


# class tt:
#     def __init__(self, a, b):
#         self.locals()[a]=30
#         self.

# a = 'x'
# locals()[a] = 20
# print(x)
#
# a = Matrix([[1,0,0,0]])
# b = Matrix([[1],[0],[0],[0]])
# print(a.T)
# print(b*a)


def KetI(ind, dim):
    # return_type: vector in the form of matrix
    vec_state = [0] * dim
    vec_state[ind] = 1
    return Matrix(np.array(vec_state)[None, :]).T

def ZeroMatrix(dim):
    # return_type: Matrix
    ZeroVector = Matrix([np.zeros(dim)])
    ZeroMatrix = ZeroVector.T * ZeroVector
    return ZeroMatrix

def trace(mat):
    dim = shape(mat)[0]
    returnTrace = mat[0, 0]
    print(returnTrace)
    for i in range(1, dim):
        returnTrace += mat[i, i]
    return returnTrace

def L2V(LinearMat, dim):
    # LinerMat_type: Matrix
    # return_type: column vector in the form of Matrix
    dim_LM = shape(LinearMat)[0]
    if dim_LM != dim:
        raise Exception('The dimension of linear matrix is not matched, please check!')
    sum = Matrix([np.zeros(dim * dim)]).T
    for i in range(0, dim):
        ketI = KetI(i, dim)
        sum += Matrix(TensorProduct(ketI, ketI))
    vector = TensorProduct(LinearMat, Matrix(Identity(dim))) * sum
    return vector

def V2L(vector, dim):
    # vector_type: column vector in the form of Matrix
    # return_type: Matrix
    vector = vector.T  # row vector
    dim_vec = shape(vector)[1]
    if dim_vec != dim * dim:
        raise Exception('The dimension of vector is not matched, please check!')
    returnMatrix = ZeroMatrix(dim)
    print(returnMatrix)
    for i in range(0, dim):
        for j in range(0, dim):
            traceM = trace(TensorProduct(KetI(i, dim), KetI(j, dim)) * vector)

            returnMatrix[i, j] = traceM.simplify()
    return returnMatrix

def Sol2LME(ini_state, M, dim):
    ini_state_vector = L2V(ini_state, dim)
    print(ini_state_vector)
    governMat = M
    solution = V2L(exp(governMat) * ini_state_vector, dim)
    return solution

M = Matrix([[Rational(1,2),0,0,1],[0,3,0,0],[0,0,0,1],[1,0,0,1]])
t = symbols('t')
M = M * t
print(exp(M*t))
sol = Sol2LME(Matrix([[1,2],[2,3]]),M,2)
print(sol)