import os


base_dir = os.path.dirname(
    os.path.dirname(__file__)
)

data_dir = os.path.join(base_dir, 'data')
eigenvalues_dir = os.path.join(base_dir, 'eigenvalues')
weights_dir = os.path.join(base_dir, 'weights')

for dir in [data_dir, eigenvalues_dir, weights_dir]:
    if not os.path.isdir(dir):
        os.mkdir(dir)