import importlib
reader = importlib.import_module('.Reader', package='src.projects.problem_id')

reader.read_problem()


class MyNumbers:

    def __iter__(self):
        self.a = 1


    def __next__(self):
        if self.a <= 20:
            x = self.a
            self.a += 1
            return x
        else:
            raise StopIteration

numbers = MyNumbers()
print(str(numbers.next()))