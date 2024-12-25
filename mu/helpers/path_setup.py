import os 


file_path = __file__

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

logs_dir = os.path.join(work_dir, 'logs')
models_dir = os.path.join(work_dir, 'models')
data_dir = os.path.join(work_dir, 'data')
output_dir = os.path.join(work_dir, 'outputs')