from sympy import symbols, sympify
from sympy.utilities.lambdify import lambdify
import numpy as np

class AbstractMethod:
    name = ""

    def __init__(self):
        self.name = "Имя метода"

    def execute(self, **kwargs):
        return self.do(**kwargs)

    def get_vars(self, n_vars):
        vars = symbols(' '.join([f'x{i+1}' for i in range(n_vars)]))
        return vars

    def get_expr(self, func_str):
        expr = sympify(func_str)
        return expr

    def safe_parse_function(self, func_str, n_vars):
        vars = self.get_vars(n_vars)
        expr = self.get_expr(func_str)
        f = lambdify(vars, expr, modules='numpy')
        return lambda x: f(*x)


class DichotomyMethod:
    def __init__(self):
        self.name = "Метод дихотомии"
    
    def minimize(self, func, a, b, epsilon=1e-6, delta=1e-6, max_iter=1000):
        """
        func - минимизируемая функция одной переменной
        a, b - начальный интервал поиска минимума
        epsilon - малое число для смещения точек в интервале
        delta - точность по длине интервала
        max_iter - максимальное число итераций
        
        Возвращает точку минимума на отрезке [a,b].
        """
        
        k = 0
        while (b - a) / 2.0 > delta and k < max_iter:
            y = (a + b - epsilon) / 2.0
            z = (a + b + epsilon) / 2.0
            f_y = func(y)
            f_z = func(z)
            
            if f_y <= f_z:
                b = z
            else:
                a = y
            
            k += 1
        
        x_min = (a + b) / 2.0
        return x_min


class GradientDescentMethod(AbstractMethod):
    def __init__(self):
        super().__init__()
        self.name = "Метод градиентного спуска"
    
    def do(self, func_str="", x0=[], alpha=0.1, epsilon=1e-6, max_iter=1000):
        """
        Мы спускаемся по вектору функции на шажок вниз, вопреки подъему градиента. 
        Если все норм, становимся на эту ступеньку и еще вниз шагаем.

        Статус: зачтено
        """
        n_vars = len(x0)
        f = self.safe_parse_function(func_str, n_vars)
        vars = self.get_vars(n_vars)
        expr = self.get_expr(func_str)
        
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


class RandomSearchMethod(AbstractMethod):
    def __init__(self):
        super().__init__()
        self.name = "Метод случайного поиска"
    
    def do(self, func_str="", bounds=[], max_iter=1000):
        """
        Статистический метод нахождения минимума для функции: если значение выходит меньше
        предыдущего, оно берется в память. Чем больше итераций, тем больше точность достигается.

        Статус: зачтено
        """
        n_vars = len(bounds)
        f = self.safe_parse_function(func_str, n_vars)
        
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


class HookJeevesMethod(AbstractMethod):
    def __init__(self):
        super().__init__()
        self.name = "Метод Хука Джився"
    
    def do(self, func_str="", x0=None, deltas=None, epsilon=1e-3, alpha=0.5, max_iter=1000):
        """
        Статус: зачтено
        """
        f = self.safe_parse_function(func_str, 2)
        x = np.array(x0, dtype=float)
        n = len(x)
        
        if deltas is None:
            deltas = np.full(n, 0.5)  # начальные шаги по умолчанию
        
        k = 0
        while k < max_iter:
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
            
            if improved and f(y) < f(x):
                x_new = y + (y - x)  # лямбда = 1 -> x_new = 2*y - x
                x = x_new
            else:
                if np.all(deltas <= epsilon):
                    print(f"Сходимость достигнута на итерации {k}")
                    break
                deltas = alpha * deltas
                # x остаётся прежним
            
            k += 1
        
        return x, f(x)