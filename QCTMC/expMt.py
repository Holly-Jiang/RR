import time

from sympy import Dummy, eye, apart, Matrix, symbols, Wild, RootSum, exp, Lambda, gamma, together, randMatrix


def expMt(M, t=1):
    """Compute matrix exponential exp(M*t)"""

    assert M.is_square
    N = M.shape[0]
    s = Dummy('s')

    Ms = (s*eye(N) - M)
    Mres = Ms.adjugate() / Ms.det()

    def expMij(i, j):
        """Partial fraction expansion then inverse Laplace transform"""
        Mresij_pfe = apart(Mres[i, j], s, full=True)
        return ilt(Mresij_pfe, s, t)

    return Matrix(N, N, expMij)


def ilt(e, s, t):
    """Fast inverse Laplace transform of rational function including RootSum"""
    a, b, n = symbols('a, b, n', cls=Wild, exclude=[s])

    def _ilt(e):
        if not e.has(s):
            return e
        elif e.is_Add:
            return _ilt_add(e)
        elif e.is_Mul:
            return _ilt_mul(e)
        elif e.is_Pow:
            return _ilt_pow(e)
        elif isinstance(e, RootSum):
            return _ilt_rootsum(e)
        else:
            raise NotImplementedError

    def _ilt_add(e):
        return e.func(*map(_ilt, e.args))

    def _ilt_mul(e):
        coeff, expr = e.as_independent(s)
        if expr.is_Mul:
            raise NotImplementedError
        return coeff * _ilt(expr)

    def _ilt_pow(e):
        match = e.match((a*s + b)**n)
        if match is not None:
            nm, am, bm = match[n], match[a], match[b]
            if nm.is_Integer and nm < 0:
                if nm == 1:
                    return exp(-(bm/am)*t) / am
                else:
                    return t**(-nm-1)*exp(-(bm/am)*t)/(am**-nm*gamma(-nm))
        raise NotImplementedError

    def _ilt_rootsum(e):
        expr = e.fun.expr
        [variable] = e.fun.variables
        return RootSum(e.poly, Lambda(variable, together(_ilt(expr))))

    return _ilt(e)

# M=randMatrix(8,8,0,1)
# print(M)
# t=symbols('t')
# #time
# start=time.time()
# ok = expMt(M, t)
# print(f'time: {time.time()-start}\n {ok}')
# ok2=exp(M*t)
# print(f'time: {time.time()-start}\n {ok2}')