from methods import *
from console_interface import ConsoleLauncher

if __name__ == "__main__":
    launcher = ConsoleLauncher(HookJeevesMethod(), RandomSearchMethod())

    launcher.launch()
    
    # method = HookJeevesMethod()
    # func = "(x1-2)**2 + (x2-3)**2"
    # x0 = [0.0, 0.0]
    # result, f_val = method.execute(func_str=func, x0=x0, alpha=0.6)
    # print(f"Найден минимум в: {result}")
    # print(f"Значение функции: {f_val:.6f}")