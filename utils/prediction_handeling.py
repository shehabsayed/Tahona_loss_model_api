import numpy as np

def handle_negative_predictions(y):

    y = np.array(y, dtype=int)
    y[y < 0] = 0
    return y