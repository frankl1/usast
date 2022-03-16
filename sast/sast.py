# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.pipeline import Pipeline

from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.transformations.panel.rocket import Rocket

# from matrixprofile.algorithms.mass2 import mass2 as MASS

from numba import njit, prange, float32 as T, int32, objmode, int32, optional

from .mass2 import *

import gc

@njit(T[:](T[:]), fastmath=False)
def znormalize_array(arr):
    m = np.mean(arr)
    s = np.std(arr)
    
    # s[s == 0] = 1 # avoid division by zero if any
    
    return ((arr - m) / (s + 1e-10)).astype(T)
    # return arr

@njit([T[:,:](T[:,:])], fastmath=False)
def znormalize_matrix(mat):
    n_rows, n_cols = mat.shape
    m = np.zeros((n_rows, 1))
    s = np.zeros((n_rows, 1))
    
    for i in prange(n_rows):
        m[i] = np.mean(mat[i])
        s[i] = np.std(mat[i])
        
    # avoid division by zero if any by adding 1e-8 to the standard deviation
    
    return ((mat - m) / (s + 1e-10)).astype(T)

@njit(T[:](T[:], T[:], optional(T)), fastmath=False)
def apply_kernel(ts, arr, r):

    # dist_profile = None
    # with objmode(dist_profile='complex128[:]'):
    #     dist_profile = MASS(ts, arr)

    dist_profile = mass2(ts, arr)

    d_best = np.min(dist_profile)

    # Count the number of best matches,
    # This are subsequences which are at a distance of at most [r] from [arr]
    
    if r is None:
        return np.array([d_best,], dtype=T)

    neighbors = dist_profile[dist_profile <= (r + d_best)]
    count = len(neighbors)

    return np.array([d_best, count], dtype=T)

@njit(T[:,:](T[:,:], T[:, :], T), parallel = True, fastmath=False)  
def apply_kernels(X, kernels, r=-1):
    n_kernels = kernels.shape[0]
    n_X = X.shape[0]

    n_cols = n_kernels 

    if r >= 0:
        n_cols = n_kernels*2

    with objmode():
        gc.collect()

    out = np.zeros((n_X, n_cols), dtype=T)

    for i in prange(n_kernels):
        k = kernels[i]
        # k = k[~np.isnan(k)]
        for t in range(n_X):
            ts = X[t]

            if r < 0:
                out[t, i] = apply_kernel(ts, k, r)[0]
            else:
                out[t, i], out[t, n_kernels + i] = apply_kernel(ts, k, r)
            
    return out

@njit(int32[:](T[:, :], T), fastmath=False)
def duplicate_indices(X, r):
    """This function returns an array of indices which can be used as slice to remove duplicate rows from X
    
    Args:
    X: matrix 
    r: float

    Output:
    a 1d array containing the indices to remove duplicates from X
    """

    out_idx = [0] # this list will contains the indices of rows to return from T

    n = X.shape[0]

    # znormalize each row so they are on the same scale
    X_normalized = znormalize_matrix(X)

    for i in prange(1, n):
        j = 0
        while j < len(out_idx) and np.linalg.norm(X_normalized[out_idx[j]] - X_normalized[i]) > r:
            j += 1
    
        if j == len(out_idx):
            out_idx.append(i)

    return np.array(out_idx, dtype=np.int32)

def remove_duplicate(X, r):
    """This function removes duplicate rows from the matrix X

    Args:
    X: matrix of float
    r: float

    Ouputs:
    a copy of X in which duplicate rows have been removed
    """
    
    idx = duplicate_indices(X, r)
    return X[idx]



class SAST(BaseEstimator, ClassifierMixin):
    
    def __init__(self, cand_length_list, shp_step = 1, nb_inst_per_class = 1, use_count=True,random_state = None, classifier = None, radius=0.0, drop_duplicates = True):
        super(SAST, self).__init__()
        self.cand_length_list = cand_length_list
        self.shp_step = shp_step
        self.nb_inst_per_class = nb_inst_per_class
        self.kernels_dict = {} # dictionary of kernels, the keys are kernel sizes and the values are arrays of kernels of the corresponding size
        self.kernels_generators_ = {}
        self.random_state = np.random.RandomState(random_state) if not isinstance(random_state, np.random.RandomState) else random_state
        self.radius = radius
        self.classifier = classifier
        self.is_initialized = False # True if the subsequences have been extracted, otherwise it is false.
        self.drop_duplicates = drop_duplicates # whether to drop duplicates subsequences
        self.use_count = use_count # Whether to use counting features

        self.n_kernels = 0 # Number of kernels before removing duplicates
        self.n_kernels_no_duplicates = 0 # Number of kernels after removing duplicates

    def get_params(self, deep=True):
        return {
            'cand_length_list': self.cand_length_list,
            'shp_step': self.shp_step,
            'nb_inst_per_class': self.nb_inst_per_class,
            'classifier': self.classifier,
            'radius': self.radius,
            'drop_duplicates': self.drop_duplicates
        }

    def init_sast(self, X, y):

        X, y = check_X_y(X, y) # check the shape of the data

        self.cand_length_list = np.array(sorted(self.cand_length_list))

        assert self.cand_length_list.ndim == 1, 'Invalid shapelet length list: required list or tuple, or a 1d numpy array'

        if self.classifier is None:
            self.classifier = RandomForestClassifier(min_impurity_decrease=0.05, max_features=None) 

        classes = np.unique(y)
        self.num_classes = classes.shape[0]
        
        candidates_ts = []
        for c in classes:
            X_c = X[y==c]
            
            cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int) # convert to int because if self.nb_inst_per_class is float, the result of np.min() will be float
            choosen = self.random_state.permutation(X_c.shape[0])[:cnt]
            candidates_ts.append(X_c[choosen])
            self.kernels_generators_[c] = X_c[choosen]
            
        candidates_ts = np.concatenate(candidates_ts, axis=0)
        
        self.cand_length_list = self.cand_length_list[self.cand_length_list <= X.shape[1]]

        n, m = candidates_ts.shape

        self.kernels_dict = {i: np.zeros((len(range(0, m - i + 1, self.shp_step)) * n, i), dtype=np.float32) for i in self.cand_length_list}
        
        for shp_length in self.cand_length_list:
            for k, i in enumerate(range(0, m - shp_length + 1, self.shp_step)):
                s = k * n
                
                self.n_kernels += n

                self.kernels_dict[shp_length][s:s+n] = candidates_ts[:, i:i+shp_length]  


            if self.drop_duplicates:
                self.kernels_dict[shp_length] = remove_duplicate(self.kernels_dict[shp_length], self.radius)

            self.n_kernels_no_duplicates += self.kernels_dict[shp_length].shape[0]

        self.is_initialized = True

    def transform(self, X):
        
        assert self.is_initialized, 'You must initialize the model by calling init_sast(X, y) before using it' # make sure SAST is initialized

        X = check_array(X) # validate the shape of X

        res = np.concatenate([apply_kernels(X, k, self.radius if self.use_count else -1.0) for k in self.kernels_dict.values()], axis=1)
        
        return res 

    def fit(self, X, y):
        
        X, y = check_X_y(X, y) # check the shape of the data

        X_transformed = self.transform(X) # subsequence transform of X

        self.classifier.fit(X_transformed, y) # fit the classifier

        return self

    def predict(self, X):
        
        check_is_fitted(self) # make sure the classifier is fitted

        X_transformed = self.transform(X) # subsequence transform of X

        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):

        check_is_fitted(self) # make sure the classifier is fitted
        
        X_transformed = self.transform(X) # subsequence transform of X

        if isinstance(self.classifier, LinearClassifierMixin):
            return self.classifier._predict_proba_lr(X_transformed)
        return self.classifier.predict_proba(X_transformed)
    
class SASTEnsemble(BaseEstimator, ClassifierMixin):
    
    def __init__(self, cand_length_list, shp_step = 1, nb_inst_per_class = 1, random_state = None, classifier = None, weights = None, n_jobs = None):
        super(SASTEnsemble, self).__init__()
        self.cand_length_list = cand_length_list
        self.shp_step = shp_step
        self.nb_inst_per_class = nb_inst_per_class
        self.classifier = classifier
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.saste = None

        self.weights = weights

        assert isinstance(self.classifier, BaseEstimator)

        self.init_ensemble()

    def init_ensemble(self):
        estimators = []
        for i, candidate_lengths in enumerate(self.cand_length_list):
            clf = clone(self.classifier)
            sast = SAST(cand_length_list=candidate_lengths,
                          nb_inst_per_class=self.nb_inst_per_class, 
                          random_state=self.random_state, 
                          shp_step = self.shp_step,
                          classifier=clf)
            estimators.append((f'sast{i}', sast))
            

        self.saste = VotingClassifier(estimators=estimators, voting='soft', n_jobs=self.n_jobs, weights = self.weights)

    def fit(self, X, y):
        self.saste.fit(X, y)
        return self

    def predict(self, X):
        return self.saste.predict(X)

    def predict_proba(self, X):
        return self.saste.predict_proba(X)

class RocketClassifier:
    def __init__(self, num_kernels=10000, normalise=True, random_state=None, clf=None, lr_clf=True):
        rocket = Rocket(num_kernels=num_kernels, normalise=normalise, random_state=random_state)
        clf = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10)) if clf is None else clf
        self.model = Pipeline(steps=[('rocket', rocket), ('clf', clf)])
        self.lr_clf = lr_clf # False if the classifier has the method predict_proba, otherwise False
        
    def fit(self, X, y):
        self.model.fit(from_2d_array_to_nested(X), y)
        
    def predict(self, X):
        return self.model.predict(from_2d_array_to_nested(X))
    
    def predict_proba(self, X):
        X_df = from_2d_array_to_nested(X)
        if not self.lr_clf:
            return self.model.predict_proba(X_df)
        X_transformed = self.model['rocket'].transform(X_df)
        return self.model['clf']._predict_proba_lr(X_transformed)

if __name__ == "__main__":
    a = np.arange(10, dtype=T).reshape((2, 5))
    y = np.array([0, 1])
    print('input=\n', a)
    print('y=\n', y)
    
    ## SAST
    sast = SAST(cand_length_list=np.arange(2, 5), nb_inst_per_class=2, classifier=RidgeClassifierCV())
    
    sast.fit(a, y)
    
    print('kernel:\n', sast.kernels_)

    print('Proba:', sast.predict_proba(a))

    print('score:', sast.score(a, y))

    ## SASTEnsemble
    saste = SASTEnsemble(cand_length_list=[np.arange(2, 4), np.arange(4, 6)], nb_inst_per_class=2, classifier=RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    
    saste.fit(a, y)

    print('SASTEnsemble Proba:', sast.predict_proba(a))

    print('SASTEnsemble score:', sast.score(a, y))


