from methods import *
from console_interface import ConsoleLauncher

if __name__ == "__main__":
    # launcher = ConsoleLauncher(HookJeevesMethod(), RandomSearchMethod())

    # launcher.launch()
    
    method1 = NewtonMethod()
    method2 = NewtonRaphsonMethod()
    # Тест на функции Розенброка (2D)
    func_rosenbrock = "100*(x2 - x1**2)**2 + (1 - x1)**2"
    x0 = [0.0, 0.5]  # начальная точка
    result = method1.do(func_str=func_rosenbrock, x0=x0)
    print(f"-- Метод Ньютона-- \nНайденный минимум: x={result[0]}, f(x)={result[1]:.6f}")

    result = method2.do(func_str=func_rosenbrock, x0=x0)
    print(f"-- Метод Ньютона-Рафсона -- \nНайденный минимум: x={result[0]}, f(x)={result[1]:.6f}")
    # Ожидаемый вывод: x~[1.0, 1.0], f(x)~0.0
    
    
    
    # func = "x1**3-2*x1-5"
    # x0 = [-2, 3]    
    # result, f_val = method.do(func_str=func, x0=x0)
    # print(f"Найден минимум в: {result}")
    # print(f"Значение функции: {f_val:.6f}")