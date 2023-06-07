import os
import sys

project_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(project_path, 'models')
datasets_path = os.path.join(project_path, 'data')
evaluators_path = os.path.join(project_path, 'evaluators')
hook_path = os.path.join(project_path, 'hook')
sys.path.append(models_path)
sys.path.append(evaluators_path)
sys.path.append(datasets_path)
sys.path.append(hook_path)
