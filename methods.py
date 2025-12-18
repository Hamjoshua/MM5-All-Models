from sympy import symbols, sympify
from sympy.utilities.lambdify import lambdify
from scipy.linalg import LinAlgError
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


class DichotomyMethod(AbstractMethod):
    def __init__(self):
        self.name = "Метод дихотомии"
    
    def do(self, func_str="", a=0, b=1, epsilon=1e-6, delta=1e-6, max_iter=1000):
        """
        func - минимизируемая функция одной переменной
        a, b - начальный интервал поиска минимума
        epsilon - малое число для смещения точек в интервале
        delta - точность по длине интервала
        max_iter - максимальное число итераций
        
        Возвращает точку минимума на отрезке [a,b].
        """
        
        func = self.safe_parse_function(func_str=func_str, n_vars=1) # для функции 

        k = 0
        while (b - a) / 2.0 > delta and k < max_iter:
            y = (a + b - epsilon) / 2.0
            z = (a + b + epsilon) / 2.0
            f_y = func([y])
            f_z = func([z])
            
            if f_y <= f_z:
                b = z
            else:
                a = y
            
            k += 1
        
        x_min = (a + b) / 2.0
        return x_min, func([x_min])
    

class NewtonMethod(AbstractMethod):
    def __init__(self):
        self.name = "Квази-Ньютоновский метод"

    def do(self, func_str="", x0=None, epsilon1=1e-6, epsilon2=1e-6, max_iter=1000):
        """
        func_str      - строковое представление функции от n переменных
        x0            - начальное приближение (список или np.array)
        epsilon1      - точность по норме градиента: ||∇f(x)|| ≤ epsilon1
        epsilon2      - точность по изменению x и f(x)
        max_iter      - максимальное число итераций

        Возвращает: (x*, f(x*))
        """

        if x0 is None:
            x0 = [0.0]

        func = self.safe_parse_function(func_str=func_str, n_vars=len(x0))
        x = np.array(x0, dtype=float)
        k = 0

        while k < max_iter:
            # Шаг 3: вычислить градиент ∇f(x)
            try:
                grad = self._numerical_gradient(func, x)
            except Exception as e:
                raise RuntimeError(f"Ошибка вычисления градиента: {e}")

            # Шаг 4: критерий остановки по градиенту
            if np.linalg.norm(grad) <= epsilon1:
                return x.copy(), func(x)

            # Шаг 5: проверка числа итераций
            if k >= max_iter:
                break

            # Шаг 6–7: вычислить гессиан и его обратную матрицу
            try:
                hess = self._numerical_hessian(func, x)
                hess_inv = np.linalg.inv(hess)
                # Шаг 8: проверка положительной определённости через собственные значения
                if np.all(np.linalg.eigvals(hess) > 0):
                    d = -hess_inv @ grad  # Шаг 9
                else:
                    d = -grad  # Шаг 10: антиградиент
            except (LinAlgError, np.linalg.LinAlgError):
                d = -grad  # если гессиан вырожден или необратим

            # Шаг 10: выбор шага tk
            tk = 1.0
            x_new = x + tk * d
            f_curr = func(x)
            f_new = func(x_new)

            # Если не уменьшается — делаем простой градиентный шаг с backtracking
            if f_new >= f_curr:
                # Попытка найти tk, при котором f(x + tk*d) < f(x)
                found = False
                for _ in range(20):
                    tk *= 0.5
                    x_new = x + tk * d
                    f_new = func(x_new)
                    if f_new < f_curr:
                        found = True
                        break
                if not found:
                    d = -grad
                    tk = 1.0
                    x_new = x + tk * d
                    f_new = func(x_new)

            # Шаг 11: критерий остановки по x и f(x)
            if np.linalg.norm(x_new - x) <= epsilon2 and abs(f_new - f_curr) <= epsilon2:
                return x_new.copy(), f_new

            x = x_new
            k += 1

        return x.copy(), func(x)

    def _numerical_gradient(self, func, x, h=1e-8):
        """Численный градиент функции func в точке x."""
        n = len(x)
        grad = np.zeros(n)
        for i in range(n):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] -= h
            x2[i] += h
            grad[i] = (func(x2) - func(x1)) / (2 * h)
        return grad

    def _numerical_hessian(self, func, x, h=1e-5):
        """Численный гессиан функции func в точке x."""
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    x_pp = x.copy()
                    x_mm = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_pp[i] += h
                    x_mm[i] -= h
                    hess[i, j] = (func(x_pp) - 2 * func(x) + func(x_mm)) / (h ** 2)
                else:
                    x_pp = x.copy()
                    x_mm = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_pp[i] += h
                    x_pp[j] += h
                    x_mm[i] -= h
                    x_mm[j] -= h
                    x_pm[i] += h
                    x_pm[j] -= h
                    x_mp[i] -= h
                    x_mp[j] += h
                    hess[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h ** 2)
        return hess



class HalfDivisionMethod(AbstractMethod):
    def __init__(self):
        self.name = "Метод деления пополам"

    def do(self, func_str="", a=0, b=1, delta=1e-6, max_iter=100):
        """
        func - минимизируемая функция одной переменной
        a, b - начальный интервал поиска минимума
        epsilon - малое число для проверки разницы значений
        delta - точность по длине интервала
        max_iter - максимальное число итераций

        Возвращает точку минимума на отрезке [a,b].
        """
        
        func = self.safe_parse_function(func_str=func_str, n_vars=1) # для функции 

        k = 0
        while (b - a) > delta and k < max_iter:
            x_mid = (a + b) / 2.0
            L = b - a
            y = a + L / 4.0
            z = b - L / 4.0

            f_mid = func([x_mid])
            f_y = func([y])
            f_z = func([z])

            if f_y < f_mid:
                b = x_mid
                x_mid = y  # Новая средняя точка
            else:
                if f_z < f_mid:
                    a = x_mid
                    b = b
                    x_mid = z  # Новая средняя точка
                else:
                    a = y
                    b = z
                    # x_mid сохраняется

            k += 1

        x_mid = (a + b) / 2.0
        return x_mid, func([x_mid])


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