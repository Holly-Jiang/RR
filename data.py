from sympy import Rational, Matrix, exp, I, symbols, latex, trace, sqrt, Identity
from sympy.physics.quantum import TensorProduct

t = symbols('t', real=True)
r11=(3 * exp(8 * t) + exp(4 * t) + 2 * exp(2 * t * (3 - I)) + 2 * exp(2 * t * (3 + I))) * exp(-8 * t) / 8
r14=(exp(8 * t) - exp(4 * t) - 2 * exp(2 * t * (3 - I)) + 2 * exp(2 * t * (3 + I))) * exp(-8 * t) / 8
r41=(exp(8 * t) - exp(4 * t) + 2 * exp(2 * t * (3 - I)) - 2 * exp(2 * t * (3 + I))) * exp(-8 * t) / 8
r44=(3 * exp(8 * t) + exp(4 * t) - 2 * exp(2 * t * (3 - I)) - 2 * exp(2 * t * (3 + I))) * exp(-8 * t) / 8
r22=1 / 8 - exp(-4 * t) / 8
rho = Matrix([[(3 * exp(8 * t) + exp(4 * t) + 2 * exp(2 * t * (3 - I)) + 2 * exp(2 * t * (3 + I))) * exp(-8 * t) / 8, 0,
               0, (exp(8 * t) - exp(4 * t) - 2 * exp(2 * t * (3 - I)) + 2 * exp(2 * t * (3 + I))) * exp(-8 * t) / 8],
              [0, 1 / 8 - exp(-4 * t) / 8, 1 / 8 - exp(-4 * t) / 8, 0],
              [0, 1 / 8 - exp(-4 * t) / 8, 1 / 8 - exp(-4 * t) / 8, 0],
              [(exp(8 * t) - exp(4 * t) + 2 * exp(2 * t * (3 - I)) - 2 * exp(2 * t * (3 + I))) * exp(-8 * t) / 8, 0, 0,
               (3 * exp(8 * t) + exp(4 * t) - 2 * exp(2 * t * (3 - I)) - 2 * exp(2 * t * (3 + I))) * exp(-8 * t) / 8]])
ket_add_add = Rational(1, 2) * Matrix([[1, 1], [1, 1]])
ket_sub_sub = Rational(1, 2) * Matrix([[1, -1], [-1, 1]])
ket_add_sub = Rational(1, 2) * Matrix([[1, -1], [1, -1]])
ket_sub_add = Rational(1, 2) * Matrix([[1, 1], [-1, -1]])
ket_1_1 = Matrix([[0, 0], [0, 1]])
ket_0_0 = Matrix([[1, 0], [0, 0]])
ket_1_0 = Matrix([[0, 0], [1, 0]])
ket_0_1 = Matrix([[0, 1], [0, 0]])
ket_add_0_add_0 = TensorProduct(ket_add_add, ket_0_0)
ket_add_0_add_1 = TensorProduct(ket_add_add, ket_0_1)
ket_add_1_add_0 = TensorProduct(ket_add_add, ket_1_0)
ket_add_1_add_1 = TensorProduct(ket_add_add, ket_1_1)

ket_sub_0_sub_0 = TensorProduct(ket_sub_sub, ket_0_0)
ket_sub_0_sub_1 = TensorProduct(ket_sub_sub, ket_0_1)
ket_sub_1_sub_0 = TensorProduct(ket_sub_sub, ket_1_0)
ket_sub_1_sub_1 = TensorProduct(ket_sub_sub, ket_1_1)

ket_add_0_sub_0 = TensorProduct(ket_add_sub, ket_0_0)
ket_add_0_sub_1 = TensorProduct(ket_add_sub, ket_0_1)
ket_add_1_sub_0 = TensorProduct(ket_add_sub, ket_1_0)
ket_add_1_sub_1 = TensorProduct(ket_add_sub, ket_1_1)

ket_sub_0_add_0 = TensorProduct(ket_sub_add, ket_0_0)
ket_sub_0_add_1 = TensorProduct(ket_sub_add, ket_0_1)
ket_sub_1_add_0 = TensorProduct(ket_sub_add, ket_1_0)
ket_sub_1_add_1 = TensorProduct(ket_sub_add, ket_1_1)

ket_add_add_add_add=TensorProduct(ket_add_add,ket_add_add)
ket_sub_sub_sub_sub=TensorProduct(ket_sub_sub,ket_sub_sub)
ket_add_add_sub_sub=TensorProduct(ket_add_add,ket_sub_sub)
ket_sub_sub_add_add=TensorProduct(ket_sub_sub,ket_add_add)

a11=ket_sub_0_sub_0+ket_add_0_add_0+ket_sub_0_add_0+ket_add_0_sub_0
print(a11)
a14=ket_sub_0_add_1+ket_add_0_add_1-ket_sub_0_sub_1-ket_add_0_sub_1
print(a14)
a41=ket_add_1_sub_0-ket_sub_1_sub_0+ket_add_1_add_0-ket_sub_1_add_0
print(a41)
a44=ket_add_1_add_1+ket_sub_1_sub_1-ket_add_1_sub_1-ket_sub_1_add_1
print(a44)
a22=ket_add_1_add_1+ket_sub_1_sub_1+ket_add_1_sub_1+ket_sub_1_add_1
print(a22)
a23=ket_add_1_add_0+ket_sub_1_add_0-ket_add_1_sub_0-ket_sub_1_sub_0
print(a23)
a32=ket_add_0_sub_1+ket_add_0_add_1-ket_sub_0_sub_1-ket_sub_0_add_1
print(a32)
a33=ket_sub_0_sub_0+ket_add_0_add_0-ket_sub_0_add_0-ket_add_0_sub_0
print(a33)

trans_rho=Rational(1,2)*(r11*a11+r14*a14+r22*a22+r22*a23+r22*a32+r22*a33+r41*a41+r44*a44)
print(latex(trans_rho))
# print((ket_add_0_add_0, ket_add_0_add_1, ket_add_1_add_0, ket_add_1_add_1,
#        ket_sub_0_sub_0, ket_sub_0_sub_1, ket_sub_1_sub_0, ket_sub_1_sub_1,
#        ket_add_0_sub_0, ket_add_0_sub_1, ket_add_1_sub_0, ket_add_1_sub_1
#        , ket_sub_0_add_0, ket_sub_0_add_1, ket_sub_1_add_0, ket_sub_1_add_1))

# print((ket_add_0_add_0 + ket_add_0_add_1 + ket_add_1_add_0 + ket_add_1_add_1 +
#        ket_sub_0_sub_0 + ket_sub_0_sub_1 + ket_sub_1_sub_0 + ket_sub_1_sub_1 +
#        ket_add_0_sub_0 + ket_add_0_sub_1 + ket_add_1_sub_0 + ket_add_1_sub_1
#        + ket_sub_0_add_0 + ket_sub_0_add_1 + ket_sub_1_add_0 + ket_sub_1_add_1))


# print(latex(sqrt(2)*(Rational(1,2)*Matrix([[1,0,1,0],[0,1,0,1],[1,0,-1,0],[0,1,0,-1]]))*rho))
# trans_rho=(sqrt(2)*(Rational(1,2)*Matrix([[1,0,1,0],[0,1,0,1],[1,0,-1,0],[0,1,0,-1]]))*rho)
# print((ket_add_0_add_0 + ket_add_0_add_1 + ket_add_1_add_0 + ket_add_1_add_1 +
#        ket_sub_0_sub_0 + ket_sub_0_sub_1 + ket_sub_1_sub_0 + ket_sub_1_sub_1 +
#        ket_add_0_sub_0 + ket_add_0_sub_1 + ket_add_1_sub_0 + ket_add_1_sub_1
#        + ket_sub_0_add_0 + ket_sub_0_add_1 + ket_sub_1_add_0 + ket_sub_1_add_1) * trans_rho)
#
# print('---', latex((ket_add_0_add_0 + ket_add_0_add_1 + ket_add_1_add_0 + ket_add_1_add_1 +
#                     ket_sub_0_sub_0 + ket_sub_0_sub_1 + ket_sub_1_sub_0 + ket_sub_1_sub_1 +
#                     ket_add_0_sub_0 + ket_add_0_sub_1 + ket_add_1_sub_0 + ket_add_1_sub_1
#                     + ket_sub_0_add_0 + ket_sub_0_add_1 + ket_sub_1_add_0 + ket_sub_1_add_1) * trans_rho))
#
#
# print('--------------------')

# print((trace(ket_add_add_add_add * rho).simplify()))
# print( (trace(ket_sub_sub_sub_sub * rho)).simplify())
# print((trace(ket_add_add_sub_sub * rho)).simplify())
# print((trace(ket_sub_sub_add_add * rho)).simplify())

# print(trace(TensorProduct(ket_0_0,ket_0_0)*rho).simplify())
# print(trace(TensorProduct(ket_0_1,ket_0_1)*rho).simplify())
# print(trace(TensorProduct(ket_1_0,ket_1_0)*rho).simplify())
# print(trace(TensorProduct(ket_1_1,ket_1_1)*rho).simplify())


print('\\begin{itemize}\n \item $%s$\n' % latex(trace(ket_add_0_add_0 * rho)))
print('\item $%s$\n'  % latex(trace(ket_add_1_add_1 * rho)))
print('\item $%s$\n'  % latex(trace(ket_sub_0_sub_0 * rho)))
print('\item $%s$\n\\end{itemize}'  % latex(trace(ket_sub_1_sub_1 * rho)))
