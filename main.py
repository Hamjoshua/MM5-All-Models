from sympy import symbols, sympify
from sympy.utilities.lambdify import lambdify
import numpy as np


def get_vars(n_vars):
    vars = symbols(' '.join([f'x{i+1}' for i in range(n_vars)]))
    return vars

def get_expr(func_str):
    expr = sympify(func_str)
    return expr

def safe_parse_function(func_str, n_vars):
    vars = get_vars(n_vars)
    expr = get_expr(func_str)
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
    Статистический метод нахождения минимума для функции: если значение выходит меньше
    предыдущего, оно берется в память. Чем больше итераций, тем больше точность достигается.

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

def gradient_descent(func_str, x0, alpha=0.1, epsilon=1e-6, max_iter=1000):
    """
    Мы спускаемся по вектору функции на шажок вниз, вопреки подъему градиента. 
    Если все норм, становимся на эту ступеньку и еще вниз шагаем.

    Статус: 
    """
    n_vars = len(x0)
    f = safe_parse_function(func_str, n_vars)
    vars = get_vars(n_vars)
    expr = get_expr(func_str)
    
    # Градиент для каждой х
    grad_expr = [expr.diff(var) for var in vars]
    grad_funcs = [lambdify(vars, g, modules='numpy') for g in grad_expr]
    grad_direction = None # заглушка направления градиента
    
    x = np.array(x0, dtype=float)
    
    for i in range(max_iter):
        grad_val = np.array([g(*x) for g in grad_funcs])

        # Уменьшение шага, если градиент сменил направление
        # Это нужно, чтобы метод не проскочил минимум
        if i == 0:
            grad_direction = grad_val > 0
        else:
            new_grad_direction = grad_val > 0
            if (new_grad_direction.all() != grad_direction.all()):
                print(f"(+-) Градиент сменил направление! alpha={alpha}->{alpha / 2}")
                alpha /= 2 
            grad_direction = new_grad_direction

        print(f"градиент {grad_val} при х={x}")
        x_new = x - alpha * grad_val
        
        print(f"норма разности х и х_new: {np.linalg.norm(x_new - x)}, меньше эпсилона? - {np.linalg.norm(x_new - x) < epsilon}")

        if np.linalg.norm(x_new - x) < epsilon:
            print(f"Всего итераций: {i}")
            break
        x = x_new
    
    return x, f(x)

# # === Пример использования ===
# if __name__ == "__main__":
#     func = "(x1-2)^2"   # минимум в (0, 3)
#     print(func)
#     x0 = [1.0, 1.0]                # начальная точка
#     result, f_val = hook_jeeves(func, x0, epsilon=1e-4)
#     print(f"Найден минимум в: {result}")
#     print(f"Значение функции: {f_val:.6f}")

# === Случайный поиск (это все временно, потом будет нормальный консоль-запускатор)
# if __name__ == "__main__":
#     func = "(x1)**2 + (x2)**2 + x1*x2"  # минимум: (2,3)
#     bounds = [(-10, 10), (-10, 10)]
#     result, f_val = random_search(func, bounds, max_iter=20000)
#     print(f"Найден минимум в: {result}")
#     print(f"Значение функции: {f_val:.6f}")

# === Градиентный спуск
if __name__ == "__main__":
    func = "(x1-2)**2 + (x2-3)**2"  # минимум в (2,3)
    x0 = [0.0, 0.0]
    result, f_val = gradient_descent(func, x0, alpha=0.6)
    print(f"Найден минимум в: {result}")
    print(f"Значение функции: {f_val:.6f}")