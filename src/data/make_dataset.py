import pandas as pd
import numpy as np

from src.utils import generate_dataset

np.random.seed(42)
x, x_test, y, y_test = generate_dataset(n=500, test_size=.2)

pd.DataFrame({'x':x.flatten(), 'y':y.flatten()}).to_csv('data/train.csv', index=False)
pd.DataFrame({'x_test':x.flatten(), 'y_test':y.flatten()}).to_csv('data/test.csv', index=False)