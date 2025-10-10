from sympy import symbols, sympify
from sympy.utilities.lambdify import lambdify

def safe_parse_function(func_str, n_vars):
    vars = symbols(' '.join([f'x{i+1}' for i in range(n_vars)]))
    expr = sympify(func_str)
    f = lambdify(vars, expr, modules='numpy')
    return lambda x: f(*x)

if __name__ == "__main__":
    f = safe_parse_function("x1**2 + (x2 - 3)**2", 2)
    result = f([1, 2])
    print(result)