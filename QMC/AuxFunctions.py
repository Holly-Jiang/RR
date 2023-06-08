import numpy as np
from sympy.matrices import Matrix
from sympy import sqrt,shape, Trace
from sympy.physics.quantum import TensorProduct

class AuxFunctions:
    def __init__(self, QMC):
        self.QMC = QMC

    def states2ket(self, ind_state):
        # return_type: vector in the form of matrix
        vec_state = [0]* self.QMC.dimension
        vec_state[ind_state] = 1
        return Matrix(np.array(vec_state)[None,:]).T

    def KetI(self, ind, dim):
        # return_type: vector in the form of matrix
        vec_state = [0] * dim
        vec_state[ind] = 1
        return Matrix(np.array(vec_state)[None, :]).T

    def isZero(self, Mat):
        ZeroMatrix = self.ZeroMatrix(self.QMC.dimension)
        if Mat == ZeroMatrix:
            return True
        else:
            return False

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

    # Convert linear operator to vector in Hilbert space
    def linear2vector(self, LinearMat, dim):
        # LinerMat_type: Matrix
        # return_type: vector in the form of Matrix
        dim_LM = shape(LinearMat)[0]
        if dim_LM != dim:
            raise Exception('The dimension of linear matrix is not matched, please check!')
        sum = Matrix([np.zeros(dim * dim)]).T
        for i in range(0, dim):
            KetI = self.KetI(i, dim)
            sum += Matrix(np.kron(KetI, KetI))
        vector = Matrix(np.kron(LinearMat,np.identity(dim))) * sum
        return vector

    def V2L(self, vector, dim):
        # vector_type: column vector in the form of Matrix
        # return_type: Matrix
        vector = vector.T  # row vector
        dim_vec = shape(vector)[1]
        if dim_vec != dim * dim:
            raise Exception('The dimension of vector is not matched, please check!')
        returnMatrix = self.ZeroMatrix(dim)
        for i in range(0, dim):
            for j in range(0, dim):
                traceM = self.trace(TensorProduct(self.KetI(i, dim), self.KetI(j, dim)) * vector)
                returnMatrix[i, j] = traceM.simplify()  # traceM_type: symbol
        return returnMatrix

    def SuperOperator(self, super_operator, EE, dim):
        ZeroMatrix = self.ZeroMatrix(dim)
        for i in range(0, self.QMC.dimension):
            pass

    # Convert linear operator to vector in Classical-Quantum composite space
    def linear2vecCQ(self, LinearMat, dim):
        pass







