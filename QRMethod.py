from typing import List, Any
import numpy as np
from Eigen.QRDecomposition import QRdecomp
from Eigen.MatrixCalculator import Matrix

class QRmethod:

    def process(self, mat: List[List[float]], residual=None):
        # return a list of eigenvalues (not ordered), and column lists of eigenvectors

        if residual == None: residual = 0.00001
        ma = Matrix()
        row = range(len(mat))

        check = True
        qrd = QRdecomp()
        next_a = mat[:]
        eigenValues = []
        eigenVectors = [[1.0 if i == j else 0.0 for i in row] for j in row]

        while check:
            check = False
            q, r = qrd.householder(next_a)
            next_a = ma.product(r, q)
            eigenVectors = ma.product(eigenVectors, q)

            for i in row:
                for j in range(i):
                    if abs(next_a[i][j]) > residual: check = True

        for i in range(len(next_a)): eigenValues.append(next_a[i][i])

        #np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
        #print(eigenVectors)
        #print(next_a)
        #print(np.linalg.inv(eigenVectors))
        #print(np.dot(eigenVectors, np.linalg.inv(eigenVectors)))
        #print(np.dot(np.dot(eigenVectors, next_a), np.linalg.inv(eigenVectors)))

        return eigenValues, eigenVectors

    def ordering(self, eigenValues: List[float], eigenVectors: List[List[float]]):
        # return ascending ordered eigenvalues and row lists of eigenvectors

        row = range(len(eigenVectors))
        col = range(len(eigenVectors[0]))

        orderedValues = []
        orderedVectors = [[0.0 for i in col] for j in row]

        qrd = QRdecomp()
        orderedValues = qrd.ordering(eigenValues)
        for i in col:
            tmpIndex = eigenValues.index(orderedValues[i])
            for j in row: orderedVectors[j][i] = eigenVectors[j][tmpIndex]
        return orderedValues, orderedVectors


"""
test = QRmethod()

#matrix = [[1,3,0,0],[3,2,1,0],[0,1,3,4],[0,0,4,1]]

#matrix = [[5,4,3,2,1],[4,4,3,2,1],[3,3,3,2,1],[2,2,2,2,1],[1,1,1,1,1]]

#matrix = [[1,1/2,1/3,1/4,1/5],[1/2,1/3,1/4,1/5,1/6],[1/3,1/4,1/5,1/6,1/7],[1/4,1/5,1/6,1/7,1/8],[1/5,1/6,1/7,1/8,1/9]]

#matrix = [[52,30,49,28],[30,50,8,44],[49,8,46,16],[28,44,16,22]]

matrix = [[1, -2, 0, 5],[0,7,1,5],[0,4,4,0],[0,0,0,2]]

q, r = test.process(matrix)
print(np.array(q))
print(np.array(r))

q, r = test.ordering(q, r)
print(np.array(q))
print(np.array(r))

e, v = np.linalg.eig(matrix)
print(e)
print(v)
"""

"""
vv = []
print(np.array(matrix))
for i in range(len(v)): vv.append(v[i][0])
print(vv)
print(np.dot(matrix, vv))
"""
