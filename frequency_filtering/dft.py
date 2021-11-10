# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

import numpy as np
from numpy import round, zeros, set_printoptions
import math

class Dft:
    def __init__(self):
        pass

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        r = len(matrix)
        c = len(matrix[0])
        forward = np.zeros((r, c), dtype=np.complex)
        for a in range(r):
            for b in range(c):
                value = 0
                for i in range(r):
                    for j in range(c):
                        incos = (((2 * np.pi) / r) * (a * i + b * j))
                        insin = (((2 * np.pi) / c) * (a * i + b * j))
                        coshalf = matrix[i][j] * (math.cos(incos))
                        sinhalf = -1 * (matrix[i][j] * (math.sin(insin)))
                        value += complex(coshalf, sinhalf)
                forward[a][b] = value

        return forward


    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        You can implement the inverse transform formula with or without the normalizing factor.
        Both formulas are accepted.
        takes as input:
        matrix: a 2d matrix (DFT) usually complex
        returns a complex matrix representing the inverse fourier transform"""
        r = len(matrix)
        c = len(matrix[0])
        inverse = np.zeros((r, c), dtype=np.complex)
        for i in range(r):
            for j in range(c):
                value = 0
                for a in range(r):
                    for b in range(c):
                        incos = (((2 * np.pi) / r) * (a * i + b * j))
                        insin = (((2 * np.pi) / c) * (a * i + b * j))
                        coshalf = matrix[a][b] * (math.cos(incos))
                        sinhalf = -1 * (matrix[a][b] * (math.sin(insin)))
                        value += complex(coshalf, sinhalf)
                inverse[i][j] = value
        return inverse

    def magnitude(self, matrix):
        """Computes the magnitude of the input matrix (iDFT)
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the complex matrix"""
        r, c = matrix.shape

        magnitude_matrix = np.zeros(matrix.shape, dtype='int64')
        for i in range(r):
            for j in range(c):
                magnitude = math.sqrt((math.pow(matrix[i][j].real, 2) + math.pow(matrix[i][j].imag, 2)))
                magnitude_matrix[i][j] = magnitude

        return magnitude_matrix
