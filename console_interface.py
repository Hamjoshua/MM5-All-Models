from methods import AbstractMethod
import inspect

class ConsoleLauncher:
    methods: list[AbstractMethod] = []

    def __init__(self, *args):
        self.methods = args

    def launch(self):
        print("Введите цифру для запуска метода")
        [print(f"{i} - {m.name}") for i, m in enumerate(self.methods)]

        line = ""
        while(True):
            line = input()
            if(not line.isdigit()):
                print("Это не цифра")
                continue

            digit = int(line)

            if(digit not in range(0, len(self.methods) + 1)):
                print("Цифра вне массива методов")
                continue

            method = self.methods[digit]
            real_method = getattr(method, 'do', method.execute)
            func_info = inspect.signature(real_method)
            print(func_info.parameters.keys())
            break