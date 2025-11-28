import numpy as np

class SetCoverGenerator:

    def __init__(self, n_rows=50, n_cols=100, density=0.4):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.density = density

    def generate(self):
        A = np.random.choice(
            [0, 1],
            size=(self.n_rows, self.n_cols),
            p=[1 - self.density, self.density]
        )

        for i in range(self.n_rows):
            if A[i].sum() < 2:
                cols = np.random.choice(self.n_cols, size=2, replace=False)
                A[i, cols] = 1

        for j in range(self.n_cols):
            if A[:, j].sum() < 2:
                rows = np.random.choice(self.n_rows, size=2, replace=False)
                A[rows, j] = 1

        c = np.ones(self.n_cols, dtype=float)
        b = np.ones(self.n_rows, dtype=float)

        return {'A': A, 'c': c, 'b': b, 'type': 'cover'}

