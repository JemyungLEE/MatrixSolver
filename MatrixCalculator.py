from typing import List, Any
import numpy as np
import math
import os

class Matrix:

    row = 0
    column = 0
    mat: List[List[float]] = []
    meanList = []
    stdList = []

    def product(self, matA: List[List[float]], matB: List[List[float]]):
        # calculate outer product between matA and matB
        row = len(matA)
        col = len(matB[0])

        col_A = len(matA[0])
        row_B = len(matB)

        if col_A != row_B: return False
        else: product = [[0.0 for i in range(col)] for j in range(row)]

        for i in range(row):
            for j in range(col):
                tmp = 0.0
                for k in range(col_A):
                    tmp += matA[i][k] * matB[k][j]
                product[i][j] = tmp
        return product

    def inner(self, matA: List[List[float]], matB: List[List[float]]):
        # calculate Frobenius inner product between matA and matB
        row = len(matA)
        col = len(matA[0])
        if row != len(matB) and col != len(matB[0]): return False
        else: product = 0

        for i in range(row):
            for j in range(col):
                product += matA[i][j] * matB[i][j]
        return product

    def transpose(self, mat: List[List[float]]):
        tran: List[List[float]] = []

        for i in range(len(mat[0])):
            tran.append([])
            for j in range(len(mat)):
                tran[i].append(mat[j][i])

        return tran

    def mean(self, mat: List[List[float]], column = True):
        # return mean values of the columns (default)
        # if 'column = False' then mean values of rows

        row = range(len(mat))
        col = range(len(mat[0]))

        if(column):
            n = len(mat)
            self.meanList = [0.0 for i in col]
            for i in col:
                for j in row: self.meanList[i] += mat[j][i]
                self.meanList[i] /= n
        else:
            n = len(mat[0])
            self.meanList = [0.0 for i in row]
            for i in row:
                for j in col: self.meanList[i] += mat[i][j]
                self.meanList[i] /= n

        return self.meanList

    def std(self, mat: List[List[float]], column = True):
        # return standard deviation of the columns (default)
        # if 'column = False' then std of rows

        row = range(len(mat))
        col = range(len(mat[0]))
        self.mean(mat, column)

        if(column):
            n = len(mat) - 1
            self.stdList = [0.0 for i in col]
            for i in col:
                for j in row: self.stdList[i] += math.pow(mat[j][i] - self.meanList[i], 2)
                self.stdList[i] = math.sqrt(self.stdList[i] / n)
        else:
            n = len(mat[0]) - 1
            self.stdList = [0.0 for i in row]
            for i in row:
                for j in col: self.stdList[i] += math.pow(mat[i][j] - self.meanList[i], 2)
                self.stdList[i] = math.sqrt(self.stdList[i] / n)

        return self.stdList

    def centering(self, mat: List[List[float]], column = True):
        # centering the matrix by columns (default)
        # if 'column = False' then centering by rows

        row = range(len(mat))
        col = range(len(mat[0]))
        self.mean(mat, column)
        centered = [[0.0 for i in col] for j in row]

        if(column):
            for i in col:
                for j in row: centered[j][i] = mat[j][i] - self.meanList[i]
        else:
            for i in row:
                for j in col: centered[i][j] = mat[i][j] - self.meanList[i]

        return centered, self.meanList

    def standardizing(self, mat: List[List[float]], column = True):
        # return the results of standardization (or Z-score normalization) by columns (default)
        # if 'column = False' then standardization by rows

        row = range(len(mat))
        col = range(len(mat[0]))
        self.mean(mat, column)
        self.std(mat, column)
        standardized = [[0.0 for i in col] for j in row]

        if (column):
            for i in col:
                for j in row: standardized[j][i] = (mat[j][i] - self.meanList[i]) / self.stdList[i]
        else:
            for i in row:
                for j in col: standardized[i][j] = (mat[i][j] - self.meanList[i]) / self.stdList[i]

        return standardized

    def minMaxScaling(self, mat: List[List[float]], column = True):
        # return the 0 - 1 scaling results by columns (default)
        # if 'column = False' then standardization by rows

        row = range(len(mat))
        col = range(len(mat[0]))
        scaled = [[0.0 for i in col] for j in row]

        if (column):
            for i in col:
                tmp = []
                for j in row: tmp.append(mat[j][i])
                tmpMin = min(tmp)
                tmpScaled = max(tmp) - tmpMin
                for j in row: scaled[j][i] = (mat[j][i]-tmpMin)/tmpScaled
        else:
            for i in row:
                tmp = []
                for j in col: tmp.append(mat[i][j])
                tmpMin = min(tmp)
                tmpScaled = max(tmp) - tmpMin
                for j in col: scaled[i][j] = (mat[i][j] - tmpMin) / tmpScaled

        return scaled

    def normalization(self, mat, threeDimension = True, column = True):
        # adjust the length of columns (default, column = True) or rows (column = False) to be 1.0

        row = range(len(mat))

        if(threeDimension):
            col = range(len(mat[0]))
            normalized = [[0.0 for i in col] for j in row]
            if(column):
                for i in col:
                    tmp = 0
                    for j in row: tmp += math.pow(mat[j][i], 2)
                    tmp = math.sqrt(tmp)
                    for j in row: normalized[j][i] = mat[j][i] / tmp
            else:
                for i in row:
                    tmp = 0
                    for j in col: tmp += math.pow(mat[i][j], 2)
                    tmp = math.sqrt(tmp)
                    for j in col: normalized[i][j] = mat[i][j] / tmp
        else:
            normalized = [0.0 for i in row]
            tmp = 0
            for i in row: tmp += math.pow(mat[i], 2)
            tmp = math.sqrt(tmp)
            for i in row: normalized[i] = mat[i] / tmp

        return normalized

    def determinant(self, mat: List[List[float]]):
        # calculate determinant of mat
        row = len(mat)

        if row != len(mat[0]) and row < 2: return False
        elif row == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        elif row > 2:
            det = 0
            for i in range(row):
                s = [[0.0 for i in range(row - 1)] for i in range(row - 1)]
                for j in range(1, row):
                    for k in range(row):
                        if k < i: s[j-1][k] = mat[j][k]
                        elif k > i: s[j-1][k-1] = mat[j][k]
                det += pow(-1, i) * mat[0][i] * self.determinant(s)
            return det
        else: return False

    def readMatrix(self, inputFile):
        # read matrix from file
        mat = []

        try:
            f = open(inputFile, 'r')
            title = f.readline().split()
            check = True
            for s in title:
                if not s.isdecimal(): check = False
            if check:
                f.seek(0)
                for s in f: mat.append([float(e) for e in s.split()])
            else:
                for s in f: mat.append([float(e) for e in s.split()[1:]])
        except:
            print("file reading error.")
        finally:
            f.close()

        return mat

    def findAllTextFiles(self, filePath):
        # read all matrix files in the 'filePath' directory

        fileList = []
        for (path, dir, files) in os.walk(filePath):
            for fileName in files:
                if fileName.endswith(".txt"): fileList.append(os.path.join(path,fileName))
        return fileList

"""
#test = Matrix()

matPath = "/Users/Jemyung/Desktop/test/"
matFile = "/Users/Jemyung/Desktop/test/matrix_example.txt"

mat = test.readMatrix(matFile)

print(np.array(mat))

print('\n'.join(test.readAllMatrices(matPath)))

mat_A = [[1,0,0,0,0,0,0,0,0,0],[-0.0096,1.0192,-0.0096,0,0,0,0,0,0,0],[0,-0.0096,1.0192,-0.0096,0,0,0,0,0,0],[0,0,-0.0096,1.0192,-0.0096,0,0,0,0,0],[0,0,0,-0.0096,1.0192,-0.0096,0,0,0,0],[0,0,0,0,-0.0096,1.0192,-0.0096,0,0,0],[0,0,0,0,0,-0.0096,1.0192,-0.0096,0,0],[0,0,0,0,0,0,-0.0096,1.0192,-0.0096,0],[0,0,0,0,0,0,0,-0.0096,1.0192,-0.0096],[0,0,0,0,0,0,0,0,0,1]]

print(test.determinant(mat_A))

ma = np.array(mat_A)
print(np.linalg.det(mat_A))
"""