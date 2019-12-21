from typing import List, Any
import numpy as np
import math
from Eigen.MatrixCalculator import Matrix

class stat:

    r = []
    p = []
    t = []

    def corr(self, mat: List[List[float]], column = True):
        # calculate Pearson's correlation coefficient

        ma = Matrix()
        samples = []

        if column: samples = mat[:]
        else: samples = ma.transpose(mat)

        n = len(mat)
        row = range(len(mat))
        col = range(len(mat[0]))
        self.r = [[0.0 for i in len(mat[0])] for j in len(mat[0])]

        # calculate mean
        mean = [0.0 for i in len(mat[0])]
        for i in col:
            for j in row: mean[i] += mat[i][j]
            mean[i] /= n

        # calculate r
        for i in col:
            nxs = n * math.pow(mean[i], 2)
            sum_xs = 0.0
            for k in row: sum_xs += math.pow(samples[i][k])
            for j in col[i:]:
                nxy = n * mean[i] * mean[j]
                nys = n * math.pow(mean[j], 2)
                sum_xy = 0.0
                sum_ys = 0.0
                for k in row:
                    sum_xy += samples[i][k] * samples[j][k]
                    sum_ys += pow(samples[j][k])

                self.r[i][j] = (sum_xy - nxy) / math.sqrt(sum_xs - nxs) / math.sqrt(sum_ys - nys)
                self.r[j][i] = self.r[i][j]

        return self.r

    def ttest_oneSample(self, mat: List[float], u: float):
        # operate one-sample t-test
        # u is the population mean of the null hypothesis

        n = len(mat)

        mean = 0.0
        for i in range(n): mean += mat[i]
        mean /= n

        std = 0.0
        for i in range(n): std += math.pow(mat[i] - mean ,2)
        std = math.sqrt(std/(n-1))

        t = (mean-u)/(std/math.sqrt(n))
        df = n - 1

        return t, df

    def ttest_pairedSample(self, matA: List[float], matB: List[float], u = 0):
        # operate paired(dependent)-sample t-test
        # u is zero when we want to test whether the average of the difference is significant

        if len(matA) != len(matB): return None

        n = len(matA)

        dif = []
        for i in range(n): dif.append(matA[i] - matB[i])

        t = self.ttest_onesample(dif, u)

        return t

    def ttest_independentSample(self, matA: List[float], matB: List[float]):
        # operate independent two-sample t-test
        # it is assumed that the two distributions have the same variance

        na = len(matA)
        nb = len(matB)

        meanA = 0.0
        meanB = 0.0
        for i in range(na): meanA += matA[i]
        for i in range(nb): meanB += matB[i]
        meanA /= na
        meanB /= nb

        varA = 0.0
        varB = 0.0
        for i in range(na): varA += pow(matA[i] - meanA, 2)
        for i in range(nb): varB += pow(matB[i] - meanB, 2)
        varA /= na - 1
        varB /= nb - 1

        sp = math.sqrt(((na-1)*varA+(nb-1)*varB)/(na+nb-2)) # pooled standard deviation
        t = (meanA - meanB)/sp/math.sqrt(1/na+1/nb)
        df = na + nb - 2

        return t, df

    def ttest_welch(self, matA: List[float], matB: List[float]):
        # operate Welch's t-test
        # it is Not assumed that the two distributions have the same variance

        na = len(matA)
        nb = len(matB)

        meanA = 0.0
        meanB = 0.0
        for i in range(na): meanA += matA[i]
        for i in range(nb): meanB += matB[i]
        meanA /= na
        meanB /= nb

        varA = 0.0
        varB = 0.0
        for i in range(na): varA += pow(matA[i] - meanA, 2)
        for i in range(nb): varB += pow(matB[i] - meanB, 2)
        varA /= na - 1
        varB /= nb - 1

        sd = math.sqrt(varA/na + varB/nb)
        t = (meanA - meanB) / sd
        df = pow(varA/na+varB/nb ,2)/(pow(varA/na,2)/(na-1)+pow(varB/nb,2)/(nb-1))

        return t, df

    def ftest(self, matA: List[float], matB: List[float]):
        # test the alternative hypothesis that the two variances are not equal
        # the data should come from normal distribution

        na = len(matA)
        nb = len(matB)

        meanA = 0.0
        meanB = 0.0
        for i in range(na): meanA += matA[i]
        for i in range(nb): meanB += matB[i]
        meanA /= na
        meanB /= nb

        varA = 0.0
        varB = 0.0
        for i in range(na): varA += pow(matA[i] - meanA, 2)
        for i in range(nb): varB += pow(matB[i] - meanB, 2)
        varA /= na - 1
        varB /= nb - 1

        f = varA/varB

        return f

    def bartletttest(self, mat: List[List[float]], column = False):
        # the null hypothesis is all k population variances are equal
        # the alternative hypothesis is at least two are different
        # the data should come from normal distribution

        ma = Matrix()

        if not column: samples = mat[:]
        else: samples = ma.transpose(mat)

        k = len(samples)
        row = range(k)

        n = 0
        ni = []
        vi = []
        for i in row:
            ni.append(len(samples[i]))
            vi[i] = ni[i] - 1
            n += ni[i]

        mean = [0.0 for i in row]
        var = [0.0 for i in row]
        sp = 0.0    # pooled estimate for the variance

        for i in row:
            for j in range(ni[i]): mean[i] += samples[i][j]
            mean[i] /= ni[i]
            for j in range(ni[i]): var[i] += pow(samples[i][j] - mean[i], 2)
            var[i] /= vi[i]
            sp += vi[i] * var[i]
        sp /= n - k

        tmpA = 0.0
        tmpB = 0.0
        for i in row:
            tmpA += vi[i] * math.log(var[i])
            tmpB += 1.0 / vi[i]

        # calculate Bartlett's test statistic
        kais = ((n-k)*math.log(sp) - tmpA) / (1.0+1/3.0/(k-1.0)*(tmpB-1.0/(n-k)))

        df = k - 1

        return kais, df

    def center(self, sample: List[float], method ='mean'):
        # return the mean, median, or 10% trimmed mean of the list

        n = len(sample)

        center = 0.0
        if center == 'mean':
            for i in range(n): center += sample[i]
            center /= n
        elif center == 'median':
            tmpsorted = sorted(sample)
            if n % 2 == 1: center = tmpsorted[int(n / 2)]
            else: center = 0.5 * (tmpsorted[int(n / 2 - 1)] - tmpsorted[int(n / 2)])
        elif center == 'trimmed':
            tmpsorted = sorted(sample)[int(n * 0.1):int(n * 0.9)]
            for i in range(len(tmpsorted)): center += tmpsorted[i]
            center /= len(tmpsorted)
        else: return("choose center among 'median', 'mean', and 'trimmed'")

        return center

    def levenet_test(self, mat: List[List[float]], center = 'mean', column = False):
        # test the alternative hypothesis that the variances are not equal
        # the data should be continuous, but not necessarily normal distributions
        # center = ‘median’ : recommended for skewed (non-normal) distributions
        # center = ‘mean’ : recommended for symmetric, moderate-tailed distributions
        # center = ‘trimmed’ : recommended for heavy-tailed distributions

        # the test is significant against F(alpha, dff, dfs), of F-distribution

        ma = Matrix()

        if not column: samples = mat[:]
        else: samples = ma.transpose(mat)

        k = len(samples)
        row = range(k)

        n = 0
        ni = []
        for i in row:
            ni.append(len(samples[i]))
            n += ni[i]

        centerlist = [0.0 for i in row]
        for i in row:
            tmpsamples = samples[i][:]
            centerlist[i] = self.center(tmpsamples, center)

        z = 0.0
        zi = [0.0 for i in row]
        zij = [[0.0 for j in ni[i]] for i in row]
        for i in row:
            for j in range(ni[i]):
                zij[i][j] = abs(samples[i][j] - centerlist[i])
                zi[i] += zij[i][j]
            z += zi[i]
            zi[i] /= ni[i]
        z /= n

        tmpa = 0.0
        tmpb = 0.0
        for i in row:
            tmpa += ni[i] * pow(zi[i]-z,2)
            for j in range(ni[i]): tmpb += pow(zij[i][j]-zi[i],2)

        w = (n - k)/(k - 1.0) * tmpa / tmpb
        dff = k - 1.0
        dfs = n - k

        return w, dff, dfs

    def chitest(self, mat: List[List[float]], column = False):
        # calulcate Chi-square test statistics and degree of freedom

        ma = Matrix()

        if column: samples = mat[:]
        else: samples = ma.transpose(mat)

        row = range(len(samples))
        col = range(len(samples[0]))

        total = 0.0
        sumrow = [0.0 for i in row]
        sumcol = [0.0 for j in col]

        for i in row:
            for j in col:
                sumrow[i] += samples[i][j]
                sumcol[j] += samples[i][j]
                total += samples[i][j]

        chi = 0.0
        e = [[0.0 for j in col] for i in row]   # expected values
        cont = [[0.0 for j in col] for i in row]    #contribution

        for i in row:
            for j in col:
                e[i][j] = sumcol[j] * sumrow[i] / total
                cont[i][j] = pow(samples[i][j]-e[i][j], 2)/e[i][j]
                chi += cont[i][j]

        df = (len(samples)-1) * (len(samples[0]))

        return chi, df
