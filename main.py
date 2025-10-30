from sympy import symbols, sympify
from sympy.utilities.lambdify import lambdify
import numpy as np


def safe_parse_function(func_str, n_vars):
    vars = symbols(' '.join([f'x{i+1}' for i in range(n_vars)]))
    expr = sympify(func_str)
    f = lambdify(vars, expr, modules='numpy')
    return lambda x: f(*x)

def hook_jeeves(func_str, x0, deltas=None, epsilon=1e-3, alpha=0.5, max_iter=1000):
    """
    Статус: зачтено
    """
    f = safe_parse_function(func_str, 2)
    x = np.array(x0, dtype=float)
    n = len(x)
    
    if deltas is None:
        deltas = np.full(n, 0.5)  # начальные шаги по умолчанию
    
    k = 0
    while k < max_iter:
        # --- Шаг 1: Исследующий поиск ---
        y = x.copy()
        improved = False
        
        for i in range(n):
            # Пробуем +delta
            y_plus = y.copy()
            y_plus[i] += deltas[i]
            if f(y_plus) < f(y):
                y = y_plus
                improved = True
            else:
                # Пробуем -delta
                y_minus = y.copy()
                y_minus[i] -= deltas[i]
                if f(y_minus) < f(y):
                    y = y_minus
                    improved = True
        
        # --- Шаг 2: Проверка результата ---
        if improved and f(y) < f(x):
            # --- Поиск по образцу ---
            x_new = y + (y - x)  # лямбда = 1 -> x_new = 2*y - x
            x = x_new
        else:
            # --- Уменьшение шага ---
            if np.all(deltas <= epsilon):
                print(f"Сходимость достигнута на итерации {k}")
                break
            deltas = alpha * deltas
            # x остаётся прежним
        
        k += 1
    
    return x, f(x)

def random_search(func_str, bounds, max_iter=1000):
    """
    Статус: не зачтено
    """
    n_vars = len(bounds)
    f = safe_parse_function(func_str, n_vars)
    
    best_x = None
    best_val = float('inf')
    
    for i in range(max_iter):
        x = np.array([np.random.uniform(low, high) for (low, high) in bounds])
        val = f(x)
        if val < best_val:
            print(f'Лучшее значение для х={x}: {val}, итерация №{i}')
            best_val = val
            best_x = x       
            
    
    return best_x, best_val

# # === Пример использования ===
# if __name__ == "__main__":
#     func = "(x1-2)^2"   # минимум в (0, 3)
#     print(func)
#     x0 = [1.0, 1.0]                # начальная точка
#     result, f_val = hook_jeeves(func, x0, epsilon=1e-4)
#     print(f"Найден минимум в: {result}")
#     print(f"Значение функции: {f_val:.6f}")

# === Случайный поиск (это все временно, потом будет нормальный консоль-запускатор)
if __name__ == "__main__":
    func = "(x1-2)**2 + (x2-3)**2"  # минимум: (2,3)
    bounds = [(-10, 10), (-10, 10)]
    result, f_val = random_search(func, bounds, max_iter=10000)
    print(f"Найден минимум в: {result}")
    print(f"Значение функции: {f_val:.6f}")