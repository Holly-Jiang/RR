from sympy.interactive.printing import init_printing
from sympy.matrices import Matrix, GramSchmidt
from sympy import sqrt

# init_printing(use_unicode=False, wrap_line=False)

L = [Matrix([sqrt(2),3,5]), Matrix([3,6,2]), Matrix([8,3,6])]
out1 = GramSchmidt(L)
for item in out1:
    print(item)

ID_dim = Matrix([[1,0,0],[0,1,0],[0,0,1]])
a = Matrix([[1,0,0],[0,1,0],[0,0,0]])
b = Matrix([[0,0,0],[0,0,0],[0,0,1]])
if a + b == ID_dim:
    print("yes:",ID_dim)