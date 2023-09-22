import numpy as np
import cvxpy as cvx
from sklearn.cluster import KMeans
from scipy.sparse import identity

class SparseSubspaceClustering:
    '''
    Class implementing SSC algorithm
    :optim_prog: - optimization program to be solved ({'L1Perfect','L1Noise','Lasso','L1ED'})
    :lambda: - 
    :affine: - affine constraint to be added ({True, False})
    :k: - number of subspaces/clusters
    '''
    def __init__(self, optim_prog="Lasso", lmbda=0.001, affine=True, k=None):
        self.optim_prog = optim_prog
        self.lmbda = lmbda
        self.affine = affine
        self.k = k
        self.coef_matrix = None
        self.coef_matrix_outliered = None
        self.adj_matrix = None
        self.labels = None

    def _spectral_clustering(self):
        # This is direct port of JHU vision lab code. Could probably use sklearn SpectralClustering.
        self.adj_matrix = self.adj_matrix.astype(float)
        N, _ = self.adj_matrix.shape
        MAXiter = 1000  # Maximum number of iterations for KMeans
        REPlic = 20  # Number of replications for KMeans

        DN = np.diag(np.divide(1, np.sqrt(np.sum(self.adj_matrix, axis=0) + np.finfo(float).eps)))
        LapN = identity(N).toarray().astype(float) - np.matmul(np.matmul(DN, self.adj_matrix), DN)
        _, _, vN = np.linalg.svd(LapN)
        vN = vN.T
        kerN = vN[:, N - self.k:N]
        normN = np.sqrt(np.sum(np.square(kerN), axis=1))
        kerNS = np.divide(kerN, normN.reshape(len(normN), 1) + np.finfo(float).eps)
        km = KMeans(n_clusters=self.k, n_init=REPlic, max_iter=MAXiter).fit(kerNS)
        return km.labels_

    def _build_adjacency(self, K):
        self.coef_matrix_outliered = self.coef_matrix_outliered.astype(float)
        CKSym = None
        N, _ = self.coef_matrix_outliered.shape
        CAbs = np.absolute(self.coef_matrix_outliered).astype(float)
        for i in range(0, N):
            c = CAbs[:, i]
            PInd = np.flip(np.argsort(c), 0)
            CAbs[:, i] = CAbs[:, i] / float(np.absolute(c[PInd[0]]))
        CSym = np.add(CAbs, CAbs.T).astype(float)
        if K != 0:
            Ind = np.flip(np.argsort(CSym, axis=0), 0)
            CK = np.zeros([N, N]).astype(float)
            for i in range(0, N):
                for j in range(0, K):
                    CK[Ind[j, i], i] = CSym[Ind[j, i], i] / float(np.absolute(CSym[Ind[0, i], i]))
            CKSym = np.add(CK, CK.T)
        else:
            CKSym = CSym
        return CKSym

    def _outlier_detection(self):
        _, N = self.coef_matrix.shape
        OutlierIndx = list()
        FailCnt = 0
        Fail = False
        for i in range(0, N):
            c = self.coef_matrix[:, i]
            if np.sum(np.isnan(c)) >= 1:
                OutlierIndx.append(i)
                FailCnt += 1
        CMatC = self.coef_matrix.astype(float)
        CMatC[OutlierIndx, :] = np.nan
        CMatC[:, OutlierIndx] = np.nan
        OutlierIndx = OutlierIndx
        if FailCnt > (N - self.k):
            CMatC = np.nan
            Fail = True
        return CMatC, OutlierIndx, Fail

    def _sparse_coefficients(self, X):
        D, N = X.shape
        CMat = np.zeros([N, N])
        for i in range(0, N):
            y = X[:, i]
            y = y.reshape((-1,1)) 
            if i == 0:
                Y = X[:, i + 1:]
            elif i > 0 and i < N - 1:
                Y = np.concatenate((X[:, 0:i], X[:, i + 1:N]), axis=1)
            else:
                Y = X[:, 0:N - 1]

            if self.affine:
                if self.optim_prog == 'Lasso':
                    c = cvx.Variable((N - 1, 1))
                    obj = cvx.Minimize(cvx.norm(c, 1) + self.lmbda * cvx.norm(Y @ c - y))
                    constraint = [cvx.sum(c) == 1]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
                elif self.optim_prog == 'L1Perfect':
                    c = cvx.Variable((N - 1, 1))
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [Y @ c == y, cvx.sum(c) == 1]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
                elif self.optim_prog == 'L1Noise':
                    c = cvx.Variable((N - 1, 1))
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [(Y @ c - y) <= self.lmbda, cvx.sum(c) == 1]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
                elif self.optim_prog == 'L1ED':
                    c = cvx.Variable((N - 1 + D, 1))
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [np.concatenate((Y, np.identity(D)), axis=1)
                                @ c == y, cvx.sum(c[0:N - 1]) == 1]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
            else:
                if self.optim_prog == 'Lasso':
                    c = cvx.Variable((N - 1, 1))
                    obj = cvx.Minimize(cvx.norm(c, 1) + self.lmbda * cvx.norm(Y @ c - y))
                    prob = cvx.Problem(obj)
                    prob.solve()
                elif self.optim_prog == 'L1Perfect':
                    c = cvx.Variable((N - 1, 1))
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [Y @ c == y]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
                elif self.optim_prog == 'L1Noise':
                    c = cvx.Variable((N - 1, 1))
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [(Y @ c - y) <= self.lmbda]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()
                elif self.optim_prog == 'L1ED':
                    c = cvx.Variable((N - 1 + D, 1))
                    obj = cvx.Minimize(cvx.norm(c, 1))
                    constraint = [np.concatenate((Y, np.identity(D)), axis=1) @ c == y]
                    prob = cvx.Problem(obj, constraint)
                    prob.solve()

            if i == 0:
                CMat[0, 0] = 0
                CMat[1:N, 0:1] = c.value[0: N - 1]
            elif i > 0 and i < N - 1:
                CMat[0:i, i:i+1] = c.value[0:i]
                CMat[i, i] = 0
                CMat[i + 1:N, i:i+1] = c.value[i:N - 1]
            else:
                CMat[0:N - 1, N - 1:N] = c.value[0:N - 1]
                CMat[N - 1, N - 1] = 0

        eps = np.finfo(float).eps
        CMat[np.abs(CMat) < eps] = 0
        return CMat


    def fit(self, X):
        # Add data projection (as in original implementation) - not necessary
        self.coef_matrix = self._sparse_coefficients(X.T)
        self.coef_matrix_outliered, OutlierIndx, Fail = self._outlier_detection()

        if Fail == False:
            self.adj_matrix = self._build_adjacency(K=0)
            self.labels = self._spectral_clustering()
            #Grps = BestMap(sc, Grps)
            #Missrate = float(np.sum(sc != Grps)) / sc.size
            #print("Misclassification rate: {:.4f} %".format(Missrate * 100))
        else:
            print("Something failed")
        # output - segmentation
        pass