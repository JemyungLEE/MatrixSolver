from typing import List, Any
import numpy as np
import math
from Eigen.MatrixCalculator import Matrix

class QRdecomp:

    a = []
    q = []
    r = []

    def transpose(self, mat: List[List[float]]):
        tmp_mat: List[List[float]] = []

        for i in range(len(mat[0])):
            tmp_mat.append([])
            for j in range(len(mat)):
                tmp_mat[i].append(mat[j][i])

        return tmp_mat

    def ordering(self, vector: List[float]):
        tmp_vector: List[float, int] = vector[:]
        for i in range(len(tmp_vector)):
            for j in range(i, len(tmp_vector)):
                if tmp_vector[i] < tmp_vector[j]:
                    tmp_vector[i], tmp_vector[j] = tmp_vector[j], tmp_vector[i]
        return tmp_vector

    def gramschmidt(self, mat: List[List[float]]):
        # Compute the QR decomposition using the Gram-Schmidt process

        row = range(len(mat))
        col = range(len(mat[0]))
        self.a = mat[:]
        mat_a = self.transpose(mat)

        q = []
        r = [[0.0 for i in col] for j in col]
        u = [[0.0 for i in row] for j in col]
        e = [[0.0 for i in row] for j in col]

        for i in col:
            tmpU = mat_a[i][:]

            if i > 0:
               for j in range(i):
                   tmp = 0.0
                   for k in range(len(mat_a[i])):
                        tmp += mat_a[i][k] * e[j][k]
                   for k in range(len(mat_a[i])):
                        tmpU[k] -= tmp * e[j][k]

            #calculate e_n
            normU = 0.0
            tmpE = [0.0 for j in row]
            for j in tmpU:
                normU += pow(j, 2)
            normU = math.sqrt(normU)
            for j in row:
                tmpE[j] = tmpU[j] / normU

            u[i] = tmpU[:]
            e[i] = tmpE[:]

        q = self.transpose(e)

        for i in col:
            for j in range(i, len(mat[i])):
                for k in row:
                    r[i][j] += mat_a[j][k] * e[i][k]

        self.q = q[:]
        self.r = r[:]

        return q, r

    def householder(self, mat: List[List[float]]):
        # Compute the QR decomposition using the Householder reflections

        ma = Matrix()

        if len(mat) < len(mat[0]): return False
        col = range(len(mat[0]))

        a = []
        r = []
        q = []
        hm = []     # Householder matrix list

        for i in col[:-1]:
            alpha = 0
            u = []
            v = []
            qn = []
            qa = []

            if len(a) == 0: a = mat[:]

            row = range(len(a))
            check = False
            for j in row[1:]:
                if a[j][0] != 0.0: check = True

            if check:
                for j in row:
                    u.append(a[j][0])
                    alpha += math.pow(u[j],2)
                alpha = math.sqrt(alpha)
                u[0] -= alpha
                norm_u = 0
                for j in row: norm_u += math.pow(u[j],2)
                norm_u = math.sqrt(norm_u)
                for j in row: v.append(u[j]/norm_u)
                for j in row:
                    qn.append([])
                    for k in row:
                        if j == k: qn[j].append(1.0)
                        else: qn[j].append(0.0)
                        qn[j][k] -= 2.0 * v[j] * v[k]
            else: qn = [[1.0 if j == k else 0.0 for j in row] for k in row]
            qa = ma.product(qn, a)

            a = []
            for j in row[1:]: a.append(qa[j][1:])
            hm.append(qn[:])

        row = range(len(mat))

        for i in col[1:-1]:
            qk = [[0.0 for j in col] for k in row]
            for j in range(i): qk[j][j] = 1.0
            for j in range(len(hm[i])):
                for k in range(len(hm[i][0])):
                    qk[j+i][k+i] = hm[i][j][k]
            hm[i] = qk[:]

        # calculate R
        r = [[1.0 if j == k else 0.0 for j in col] for k in col]
        for i in reversed(hm): r = ma.product(r, i)
        r = ma.product(r, mat)
        self.r = r[:]

        # calculate Q
        q = [[1.0 if j == k else 0.0 for j in col] for k in col]
        for i in hm: q = ma.product(q, ma.transpose(i))
        self.q = q[:]

        return q, r
"""
test = QRdecomp()

matrix = [[1,3,0,0],[3,2,1,0],[0,1,3,4],[0,0,4,1]]
#matrix = [[12,-51,4],[6,167,-68],[-4,24,-41]]

test.decompHouseholder(matrix)
test.decomposition(matrix)

print(test.q)
print(test.r)
print(np.dot(test.q, test.r))
"""

