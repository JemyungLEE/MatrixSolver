from typing import List, Any
import math
import numpy as np
from Eigen.QRMethod import QRmethod
from Eigen.MatrixCalculator import Matrix

class SVD:

    def simple(self, mat: List[List[float]]):
        # calculate singular value decomposition by simple algorithm of MM* and M*M

        row = range(len(mat))
        col = range(len(mat[0]))

        u = []
        s = []  # diagonal matrix of singular values
        v = []
        e = []  # list of eigenvalues

        ma = Matrix()
        qrd = QRmethod()

        mm = ma.product(ma.transpose(mat), mat)
        e, v = qrd.process(mm)
        e, v = qrd.ordering(e, v)
        s = [[0.0 for i in col] for j in row]
        for i in row: s[i][i] = math.sqrt(e[i])

        av = ma.product(mat, v)
        u = [[0.0 for i in row] for j in row]
        for i in row:
            for j in row: u[j][i] = av[j][i] / s[i][i]

        """
        mm = ma.product(mat, ma.transpose(mat))
        e, u = qrd.process(mm)
        s = [[0.0 for c in col] for r in row]
        for i in row: s[i][i] = math.sqrt(e[i])

        mm = ma.product(ma.transpose(mat), mat)
        e, v = qrd.process(mm)
        """

        return u, s, v


    def inverse(self, mat: List[List[float]]):
        # calculate inverse matrix by SVD

        u, s, v = self.simple(mat)

        ma = Matrix()

        for i in range(len(s)): s[i][i] = 1.0/s[i][i]

        return ma.product(ma.product(v, s),ma.transpose(u))


test = SVD()

#matrix = [[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,2,0,0,0]]

#matrix =  [[3,2,2],[2,3,-2]]

matrix = [[2,0,8,6,0],[1,6,0,1,7],[5,0,7,4,0],[7,0,8,5,0],[0,10,0,0,7]]

u, s, v = test.simple(matrix)
ua = np.array(u)
sa = np.array(s)
va = np.array(v).transpose()
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
#for i in range(len(ua[0])):
#    if ua[0,i]<0: ua[:,i] = -ua[:,i]
#for i in range(len(va)):
#    if va[i,0]<0: va[i,:] = -va[i,:]
#print(ua)
#print(sa)
#print(va)
#val = np.dot(np.dot(ua,sa),va)
#print(val)
#print(np.allclose(matrix, val))

inv = test.inverse(matrix)

print(np.array(inv))

invc = np.linalg.inv(matrix)
print(invc)
print(np.allclose(inv, invc))

"""
print("[numpy results]")
ut, st, vt = np.linalg.svd(matrix)
#for i in range(len(ut[0])):
#    if ut[0,i]<0: ut[:,i] = -ut[:,i]
#for i in range(len(vt)):
#    if vt[i,0]<0: vt[i,:] = -vt[i,:]
print(ut)
print(np.diag(st))
print(vt)
valt = np.dot(np.dot(ut,np.diag(st)),vt)
print(valt)
print(np.allclose(matrix, valt))
print(np.allclose(ua, ut))
print(np.allclose(sa, np.diag(st)))
print(np.allclose(va, vt))
"""