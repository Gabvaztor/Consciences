import importlib
reader = importlib.import_module('.Reader', package='src.projects.problem_id')

reader.read_problem()