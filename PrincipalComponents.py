from typing import List, Any
import numpy as np
from Eigen.QRMethod import QRmethod
from Eigen.MatrixCalculator import Matrix
from matplotlib import pyplot as plt

import sklearn as sk
from sklearn.decomposition import PCA as skp

class PCA:

    mean = []
    centered = []
    eigVal: List[float]
    eigVec: List[List[float]]
    numOfComp = 0   # number of principal components
    transformed = []


    def preProcess(self, mat: List[List[float]], residual=None):
        # operation of centering and calculation of covariance, eigenvalue, and eigenvector

        ma = Matrix()
        qr = QRmethod()

        self.centered, self.mean = ma.centering(mat, True)
        #self.centered = ma.standardizing(mat, True)
        cov = ma.product(ma.transpose(self.centered), self.centered)

        n = len(mat) - 1
        row = range(len(mat))
        col = range(len(mat[0]))
        for i in col:
            for j in col: cov[i][j] /= n

        self.eigVal, self.eigVec = qr.process(cov)
        self.eigVal, self.eigVec = qr.ordering(self.eigVal, self.eigVec)

        return self.eigVal, self.eigVec

    def postProcess(self, mat: List[List[float]], components: int):
        # operate post process

        ma = Matrix()
        if isinstance(components, int): self.numOfComp = components
        elif components > 0 and components < 1:
            check = 0.0
            for i in self.eigVal: check += i
            check *= components
            while check > 0:
                check -= self.eigVal[self.numOfComp]
                self.numOfComp += 1
        else: print("components value is wrong: not int nor 0 to 1")

        self.transformed = ma.product(ma.transpose(self.eigVec)[:self.numOfComp], ma.transpose(self.centered))
        self.transformed = ma.transpose(self.transformed)

        return self.transformed

    def inverseProcess(self):
        # return inverse-transformed data by the reduced dimensions

        col = range(len(self.eigVec[0]))
        row = range(len(self.transformed))
        ma = Matrix()
        tm = []

        for i in col:
            tm.append([])
            for j in range(self.numOfComp): tm[i].append(self.eigVec[i][j])

        inverse = ma.transpose(ma.product(tm, ma.transpose(self.transformed)))
        for i in col:
            for j in row: inverse[j][i] += self.mean[i]

        return inverse

"""
test = PCA()
ma = Matrix()

#matrix = [[52,30,49,28],[30,50,8,44],[49,8,46,16],[28,44,16,22]]
matrix = [[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],[2.3,2.7],[2,1.6],[1,1.1],[1.5,1.6],[1.1,0.9]]

e, v = test.preProcess(matrix)

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
print(np.array(e))
print(np.array(v))

nm = test.postProcess(matrix, 1)
print(np.array(nm))

inv = test.inverseProcess()
print(np.array(inv))

x = np.array(range(len(e))) + 1
plt.plot(x,e,'ro-', linewidth=1.0, color="blue")
#plt.show()
"""

"""
print("[validation]")
sta = sk.preprocessing.StandardScaler()
scaled = sta.fit_transform(matrix)
comp = skp(n_components= 1)
result = comp.fit_transform(scaled)
print(comp.singular_values_)
print(comp.components_)

print(result)

rt = comp.inverse_transform(result)
print(rt)
rtt = sta.inverse_transform(rt)
print(rtt)
"""

