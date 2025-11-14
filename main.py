from methods import *
from console_interface import ConsoleLauncher

if __name__ == "__main__":
    # launcher = ConsoleLauncher(HookJeevesMethod(), RandomSearchMethod())

    # launcher.launch()
    
    method = DichotomyMethod()
    func = "x1**3-2*x1-5"
    a, b = -2, 3
    result, f_val = method.do(func_str=func, a=a, b=b)
    print(f"Найден минимум в: {result}")
    print(f"Значение функции: {f_val:.6f}")