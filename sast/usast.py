import numpy as np
import numba as nb

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from numba import njit, prange, float64 as T, int32, int32, optional

UED = 0
HBD = 1
JSD = 2
DUST = 3

@njit(nb.types.UniTuple(T[:], 2)(T[:], T[:]), fastmath=False)
def znormalize_uncertain_array(arr, arr_err):
    """Z-normalize the uncertain time series given as input
    
    Inputs:
    arr: a 1-d array representing the best estimate time series
    arr_err: a 1-d array representing the uncertainty on the best estimates
    
    Outputs:
    a 2-d array of shape (2, len(arr)) such that the first row is the normalized version of arr, and the second row is the uncertainty on the first row values
    """

    m = np.mean(arr)
    s = np.std(arr)
    
    arr_norm = ((arr - m) / (s + 1e-10)) # 1e-10 to avoid zero-division

    arr_norm_err = arr_err / (arr + 1e-10) # get error in percentage (relative error)
    arr_norm_err = np.abs(arr_norm * arr_norm_err) # go back to absolute error and make sure it is positive
    
    return arr_norm.astype(T), arr_norm_err.astype(T)
    # return arr

@njit(nb.types.UniTuple(T[:,:], 2)(T[:,:], T[:,:]), fastmath=False)
def znormalize_uncertain_matrix(mat, mat_err):
    """Z-normalize each row in the input matrix
    
    Inputs:
    mat: a 2-d array representing the best estimate time series. Each row is a time series
    mat_err: a 2-d array representing the uncertainty on the best estimates. 
    
    Outputs:
    a tuple (mat_norm, mat_norm_err) such that the first element is the normalized version of mat, and the second row is the uncertainty on the first element values
    """
    n_rows, _ = mat.shape

    mat_norm = np.zeros_like(mat)
    mat_norm_err = np.zeros_like(mat_err)
    
    for i in prange(n_rows):
        out = znormalize_uncertain_array(mat[i], mat_err[i])
        mat_norm[i] = out[0]
        mat_norm_err[i] = out[1]
    
    return mat_norm.astype(T), mat_norm_err.astype(T)

@njit(T(T[:], T[:], T[:], T[:]), fastmath=False)
def hellinger_distance(x1, x1_err, x2, x2_err):
    mu1, sigma1 = x1, x1_err
    mu2, sigma2 = x2, x2_err
    sum_sigma_squared = np.square(sigma1) + np.square(sigma2)
    res = np.exp(-np.square(mu1 - mu2)/(4 * sum_sigma_squared))
    res *= np.sqrt(2*sigma1*sigma2/sum_sigma_squared)

    res = np.linalg.norm(1 - res)
    
    return res

@njit(T(T[:], T[:], T[:], T[:]), fastmath=False)
def jensen_shannon_distance(x1, x1_err, x2, x2_err):
    mu1, sigma1 = x1, x1_err
    mu2, sigma2 = x2, x2_err
    mu3 = (mu1 + mu2) / 2
    sigma3_squared = (np.square(sigma1) + np.square(sigma2)) / 4
    sigma3 = np.sqrt(sigma3_squared)
    
    D_kl_13 = np.log2(sigma3/sigma1 + 1e-10) + (np.square(sigma1) + np.square(mu1 - mu3)) / (2 * sigma3_squared) - 0.5
    D_kl_23 = np.log2(sigma3/sigma2 + 1e-10) + (np.square(sigma2) + np.square(mu2 - mu3)) / (2 * sigma3_squared) - 0.5
    
    res = (np.linalg.norm(D_kl_13) + np.linalg.norm(D_kl_23)) / 2

    return res

@njit(nb.types.UniTuple(T, 2)(T[:], T[:], T[:], T[:]), fastmath=False)
def uncertain_euclidean_distance(x1, x1_err, x2, x2_err):
    dif = x1 - x2
    dist = np.sum(dif**2)
    dist_err = 2 * np.sum(np.abs(dif)*(x1_err+x2_err))
    return dist, dist_err

@njit(T(T[:], T[:], T[:], T[:]), fastmath=False)
def dust_normal(x1, x1_err, x2, x2_err):
    err_std = np.where(x1_err > x2_err, x1_err, x2_err)
    err_std[err_std==0] = 0.4238 # an approximation solution of 2x(1 + x^2) = 1
    res = np.linalg.norm(np.abs(x1 - x2) / (2 * err_std * ( 1 + err_std**2)))
    return res

@njit(nb.types.UniTuple(T, 3)(T[:], T[:], T[:], T[:], optional(T), optional(int32)), fastmath=False)
def apply_kernel(ts, ts_err, arr, arr_err, r, d=0):
    """This function computes the similarity between arr and the best matching sub-sequence in ts.

    Args:
    ts : a 1d array of type T
    ts_err : a 1d array of type T. This is the uncertainty of the values in ts
    arr: a 1d array of type T. This array should not be longer than the first argument
    arr_err: a 1d array of type T. This is the uncertainty of the values in arr
    r : a float. This is the maximum distance to considere two subsequences as identical. For the moment, considere only the best estimate, the uncertainty 
    on this maximum value could be considered in future if found valuable.
    d : and int, default=0. the distance to use. 0: UED, 1: HBD, 2: JSD, 3: DUST

    Outputs:
    a tuple of three T elements such that the first element is the best estimate of the similarity between arr and the best match, the second element is the 
    uncertainty on the best estimate and the third and last element is the number of matches
    """
    

    m = len(ts)
    l = len(arr)
    best_dist = np.inf
    best_dist_err = np.inf 
    count = 0

    arr_norm, arr_norm_err = znormalize_uncertain_array(arr, arr_err)
    
    for i in range(m - l + 1):
        ts_norm, ts_norm_err = znormalize_uncertain_array(ts[i:i+l], ts_err[i:i+l])
        dist, dist_err = np.inf, 0.0

        if d == HBD:
            dist = hellinger_distance(ts_norm, ts_norm_err, arr_norm, arr_norm_err)
        elif d == JSD:
            dist = jensen_shannon_distance(ts_norm, ts_norm_err, arr_norm, arr_norm_err)
        elif d == DUST:
            dist = dust_normal(ts_norm, ts_norm_err, arr_norm, arr_norm_err)
        else:
            dist, dist_err = uncertain_euclidean_distance(ts_norm, ts_norm_err, arr_norm, arr_norm_err)

        """We use the simple ordering strategy to compare the uncertain similarities.

        Mbouopda, M. F., & Nguifo, E. M. (2020, November). Uncertain Time Series Classification With Shapelet Transform. In 2020 International Conference on Data Mining Workshops (ICDMW) (pp. 259-266). IEEE.
        """
        if (dist < best_dist) or ((dist == best_dist) and (dist_err < best_dist_err)):
            best_dist = dist
            best_dist_err = dist_err

        if dist <= r:
            count += 1

    return T(best_dist), T(best_dist_err), T(count)

@njit(nb.types.UniTuple(T[:, :], 2)(T[:,:], T[:,:], T[:,:], T[:, :], T, int32), parallel=True, fastmath=False)  
def apply_kernels(X, X_err, kernels, kernels_err, r=-1.0, d=0):
    """This function applies the subsequences transformation on the input (X, X_err) using subsequences (kernels, kernels_err)

    Args:
    X: a 2d array of type T. Each row is a time series
    X_err: a 2d array of type T. This is the uncertainty of the values in X
    kernels: a 2d array of type T. Each row is a subsequence that will be considered for the transformation
    kernels_err: a 2d array of type T. Each row is the uncertainty of the same row in kernels
    r: a float. This is the maximum distance to considere two subsequences as identical. For the moment, considere only the best estimate, the uncertainty 
    on this maximum value could be considered in future if found valuable. If negative, then similar subsequences are not identified.
    d (int, default=0): the distance to use. 0: UED, 1: HBD, 2: JSD, 3:DUST

    Outputs:
    a tuple of two matrices, such that the first one is the best estimate of the transformation and the second one is uncertainty on that estimate
    """

    n_kernels = kernels.shape[0]
    n_X = X.shape[0]

    n_cols = n_kernels 

    if r >= 0:
        n_cols = n_kernels*2

    out = np.zeros((n_X, n_cols), dtype=T)

    out_err = np.zeros((n_X, n_kernels), dtype=T)

    for i in prange(n_kernels):
        k = kernels[i]
        k_err = kernels_err[i]
        for t in range(n_X):
            ts = X[t]
            ts_err = X_err[t]

            if r < 0:
                out[t, i], out_err[t, i] = apply_kernel(ts, ts_err, k, k_err, r, d)[:2]
            else:
                out[t, i], out_err[t, i], out[t, n_kernels + i] = apply_kernel(ts, ts_err, k, k_err, r, d)
            
    return out, out_err

@njit(int32[:](T[:, :], T[:, :], T), fastmath=False)
def duplicate_indices(X, X_err, r):
    """This function returns an array of indices which can be used as slice to remove duplicate rows from X. For the moment, only the best estimate are used to find dublicates but uncertainty could be added in the future if found valuable
    
    Args:
    X: matrix. Each row is a subsequence
    X_err: matrix. Each row is the uncertainties of the values of the same rows in X
    r: float

    Output:
    a 1d array containing the indices to remove duplicates from X
    """

    out_idx = [0] # this list will contains the indices of rows to return from T

    n = X.shape[0]

    # znormalize each row so they are on the same scale
    X_normalized, X_normalized_err = znormalize_uncertain_matrix(X, X_err)

    for i in prange(1, n):
        j = 0
        while j < len(out_idx) and np.linalg.norm(X_normalized[out_idx[j]] - X_normalized[i]) > r:
            j += 1
    
        if j == len(out_idx):
            out_idx.append(i)

    return np.array(out_idx, dtype=np.int32)

def remove_duplicate(X, X_err, r):
    """This function removes duplicate rows from the matrix X

    Args:
    X: matrix of float
    X_err: matrix of float
    r: float

    Ouputs:
    a copy of X and X_err in which duplicate rows have been removed
    """
    
    idx = duplicate_indices(X, X_err, r)
    return X[idx], X_err[idx]



class USAST(BaseEstimator, ClassifierMixin):
    
    def __init__(self, cand_length_list, shp_step = 1, nb_inst_per_class = 1, use_count=True, random_state = None, classifier = None, radius=0.0, drop_duplicates = True, scale_before_fit = True, distance='UED'):
        super(USAST, self).__init__()

        assert distance in ('UED', 'JSD', 'HBD', 'DUST'), 'Distance should be one of `UED`, `JSD`, `HBD`, or `DUST`'

        self.distance = UED
        if distance == 'HBD':
            self.distance = HBD
        elif distance == 'JSD':
            self.distance = JSD
        elif distance == 'DUST':
            self.distance = DUST

        self.cand_length_list = cand_length_list
        self.shp_step = shp_step
        self.nb_inst_per_class = nb_inst_per_class
        
        self.kernels_dict = {} # dictionary of kernels, the keys are kernel sizes and the values are arrays of kernels of the corresponding size
        self.kernels_err_dict = {} # uncertainty on kernels
        
        self.kernels_generators_ = {} # time series used to generate kernels
        self.kernels_generators_err_ = {} # uncertainties of times series used to generate kernels

        self.random_state = np.random.RandomState(random_state) if not isinstance(random_state, np.random.RandomState) else random_state
        self.radius = radius
        self.classifier = classifier
        self.is_initialized = False # True if the subsequences have been extracted, otherwise it is false.
        self.drop_duplicates = drop_duplicates # whether to drop duplicates subsequences
        self.use_count = use_count # Whether to use counting features

        self.n_kernels = 0 # Number of kernels before removing duplicates
        self.n_kernels_no_duplicates = 0 # Number of kernels after removing duplicates

        self.scale_before_fit = scale_before_fit # Wheter to scale the transformed data before fitting the classifier

    def get_params(self, deep=True):
        return {
            'cand_length_list': self.cand_length_list,
            'shp_step': self.shp_step,
            'nb_inst_per_class': self.nb_inst_per_class,
            'classifier': self.classifier,
            'radius': self.radius,
            'drop_duplicates': self.drop_duplicates,
            'scale_before_fit': self.scale_before_fit
        }

    def init_usast(self, X, X_err, y):
        """This function initializes the USAST model. The initialization consists of randomly selecting reference uncertain time series, generating subsequences and instantiate a scaler if needed

        Inputs:
        X: a matrix containing the best estimates of the uncertain time series. Each row is a time series
        X_err: a matrix containing the uncertainties. Each row is the uncertainty of the same row in X
        y: a 1d array containing the class label of each uncertain time series

        Outputs:
        None
        """
        X, y = check_X_y(X, y) # check the shape of the data
        X_err = check_array(X_err) # check the shape of the data

        self.cand_length_list = np.array(sorted(self.cand_length_list))

        assert self.cand_length_list.ndim == 1, 'Invalid shapelet length list: required list or tuple, or a 1d numpy array'

        if self.classifier is None:
            self.classifier = RandomForestClassifier(min_impurity_decrease=0.05, max_features=None) 

        classes = np.unique(y)
        self.num_classes = classes.shape[0]
        
        candidates_ts = []
        candidates_ts_err = []
        for c in classes:
            X_c = X[y==c]
            X_c_err = X_err[y==c]
            
            cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int) # convert to int because if self.nb_inst_per_class is float, the result of np.min() will be float
            choosen = self.random_state.permutation(X_c.shape[0])[:cnt]

            candidates_ts.append(X_c[choosen])
            candidates_ts_err.append(X_c_err[choosen])

            self.kernels_generators_[c] = X_c[choosen]
            self.kernels_generators_err_[c] = X_c_err[choosen]
            
        candidates_ts = np.concatenate(candidates_ts, axis=0)
        candidates_ts_err = np.concatenate(candidates_ts_err, axis=0)
        
        self.cand_length_list = self.cand_length_list[self.cand_length_list <= X.shape[1]]

        n, m = candidates_ts.shape

        self.kernels_dict = {i: np.zeros((len(range(0, m - i + 1, self.shp_step)) * n, i), dtype=np.float64) for i in self.cand_length_list}
        self.kernels_err_dict = {i: np.zeros((len(range(0, m - i + 1, self.shp_step)) * n, i), dtype=np.float64) for i in self.cand_length_list}

        for shp_length in self.cand_length_list:
            for k, i in enumerate(range(0, m - shp_length + 1, self.shp_step)):
                s = k * n
                
                self.n_kernels += n

                self.kernels_dict[shp_length][s:s+n] = candidates_ts[:, i:i+shp_length]
                self.kernels_err_dict[shp_length][s:s+n] = candidates_ts_err[:, i:i+shp_length]  


            if self.drop_duplicates:
                self.kernels_dict[shp_length], self.kernels_err_dict[shp_length] = remove_duplicate(self.kernels_dict[shp_length], self.kernels_err_dict[shp_length], self.radius)

            self.n_kernels_no_duplicates += self.kernels_dict[shp_length].shape[0]

        self.is_initialized = True

        if self.scale_before_fit:
            self.scaler = StandardScaler()

    def transform(self, X, X_err):
        
        assert self.is_initialized, 'You must initialize the model by calling init_sast(X, y) before using it' # make sure SAST is initialized

        X = check_array(X) # validate the shape of X
        X_err = check_array(X_err) # validate the shape of X_err

        res, count, res_err = [], [], []

        for k in self.kernels_dict.keys():
            X_transformed, X_transformed_err = apply_kernels(X, X_err, self.kernels_dict[k], self.kernels_err_dict[k], self.radius if self.use_count else -1.0, self.distance)

            count_start = X_transformed.shape[1]
            if self.use_count:
                count_start = count_start // 2
                count.append(X_transformed[:, count_start:])

            res.append(X_transformed[:, :count_start])
            res_err.append(X_transformed_err)
        
        res = np.concatenate(res, axis=1)
        res_err = np.concatenate(res_err, axis=1)
        
        if self.use_count:
            count = np.concatenate(count, axis=1)
            
            assert np.all(count>=0), 'Negative count not allowed'

            res = np.concatenate([res, count], axis=1)

        return res, res_err

    def fit(self, X, X_err, y):
        
        X, y = check_X_y(X, y) # check the shape of the data
        X_err = check_array(X_err) # check the shape of the data

        X_transformed, X_transformed_err = self.transform(X, X_err) # subsequence transform of X

        count_start = X_transformed.shape[1]
        if self.use_count:
            count_start //= 2 

        if self.scale_before_fit:
            relative_err = X_transformed_err / (X_transformed[:, :count_start] + 1e-10) # add 1e-10 to avoid zero division
            X_transformed = self.scaler.fit_transform(X_transformed) # Scale the best guesses
            X_transformed_err = np.abs(X_transformed[:, :count_start] * relative_err) # Absolute error on the scaled best guesses

        X_transformed_concat = np.concatenate([X_transformed, X_transformed_err], axis=1)

        self.classifier.fit(X_transformed_concat, y) # fit the classifier

        return self

    def predict(self, X, X_err):
        
        check_is_fitted(self) # make sure the classifier is fitted

        X_transformed, X_transformed_err = self.transform(X, X_err) # subsequence transform of X

        count_start = X_transformed.shape[1]
        if self.use_count:
            count_start //= 2 

        if self.scale_before_fit:
            relative_err = X_transformed_err / (X_transformed[:, :count_start] + 1e-10) # add 1e-10 to avoid zero division
            X_transformed = self.scaler.transform(X_transformed) # Scale the best guesses
            X_transformed_err = np.abs(X_transformed[:, :count_start] * relative_err) # Absolute error on the scaled best guesses

        X_transformed_concat = np.concatenate([X_transformed, X_transformed_err], axis=1)

        return self.classifier.predict(X_transformed_concat)

    def score(self, X, X_err, y):
        preds = self.predict(X, X_err)
        return accuracy_score(y, preds)

    def predict_proba(self, X, X_err):

        check_is_fitted(self) # make sure the classifier is fitted
        
        X_transformed, X_transformed_err = self.transform(X, X_err) # subsequence transform of X

        count_start = X_transformed.shape[1]
        if self.use_count:
            count_start //= 2 

        if self.scale_before_fit:
            relative_err = X_transformed_err / (X_transformed[:, :count_start] + 1e-10) # add 1e-10 to avoid zero division
            X_transformed = self.scaler.transform(X_transformed) # Scale the best guesses
            X_transformed_err = np.abs(X_transformed[:, :count_start] * relative_err) # Absolute error on the scaled best guesses

        X_transformed_concat = np.concatenate([X_transformed, X_transformed_err], axis=1)

        if isinstance(self.classifier, LinearClassifierMixin):
            return self.classifier._predict_proba_lr(X_transformed_concat)
        return self.classifier.predict_proba(X_transformed_concat)
    