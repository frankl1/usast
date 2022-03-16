import time 
import os 
import numpy as np
import joblib
import argparse

import multiprocessing

n_jobs = multiprocessing.cpu_count()

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from sast.utils import *
from sast.usast import *
from sast.ugnb import UGaussianNB

from sklearn.utils import shuffle
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import xgboost as xgb

from sklearn.metrics import precision_recall_fscore_support, log_loss

parser = argparse.ArgumentParser()

parser.add_argument('dataset_folder', help='The folder containing the datasets')
parser.add_argument('dataset_name', help='The name of the dataset, the part before the _[dimension]')
parser.add_argument('output_file', help='The file in which the results will be written')
parser.add_argument('-b', '--bands', help='String of dimensions to consider', default='giruyz')
parser.add_argument('-r', '--radius', help='The neighborhood radius in which subsequences are considered similar', type=float, default=0.0)
parser.add_argument('-c', '--use_count', help='Whether to count the number of times a subsequence appears in a time series', action='store_true')
parser.add_argument('-d', '--drop_duplicate', help='Whether to drop duplicate subsequences from reference time series', action='store_true')
parser.add_argument('-run', '--nb_run_per_dataset', help='The number of times each dataset must be run', type=int, default=3)
parser.add_argument('-inst','--nb_inst_per_class', help='Maximum number of references times series to use per class', type=int, default=1)
parser.add_argument('-m', '--min_shp_length', help='The minimum subsequence length to consider', type=int, default=3)
parser.add_argument('-ls', '--shp_length_step', help='The subsequence length stride to consider', type=int, default=1)
parser.add_argument('-M', '--max_shp_length', help='The maximum subsequence length to consider', type=int, default=100)
parser.add_argument('-s', '--shp_step', help='Stride value for shapelet extraction', type=int, default=1)
parser.add_argument('-k', '--keep_classes', help='List of classes to classify. The classes not in the list will be ignored', type=int, default=None, action='append')

args = parser.parse_args()

dataset_folder = args.dataset_folder

dataset_name = args.dataset_name

output_file = args.output_file

radius = args.radius

nb_run_per_dataset = args.nb_run_per_dataset

nb_inst_per_class = args.nb_inst_per_class

shp_step = args.shp_step

min_shp_length = args.min_shp_length

max_shp_length = args.max_shp_length

shp_length_step = args.shp_length_step

use_count = args.use_count

drop_duplicate = args.drop_duplicate

keep_classes = args.keep_classes

bands = args.bands

print("Running experiment with the following parameters:\n", args)

def read_data(folder, dataset, keep_classes=None):
	train_ds, train_ds_err, test_ds, test_ds_err = load_uncertain_dataset(dataset_folder, dataset)
	
	X_train, X_train_err, y_train = format_uncertain_dataset(train_ds, train_ds_err, shuffle=False)
	X_test, X_test_err, y_test = format_uncertain_dataset(test_ds, test_ds_err, shuffle=False)

	if keep_classes is not None:
		keep_train = [c in keep_classes for c in y_train]
		keep_test = [c in keep_classes for c in y_test]

		y_train = y_train[keep_train]
		X_train = X_train[keep_train]
		X_train_err = X_train_err[keep_train]

		y_test = y_test[keep_test]
		X_test = X_test[keep_test]
		X_test_err = X_test_err[keep_test]

	return X_train, X_train_err, y_train, X_test, X_test_err, y_test

def build_sast(min_shp_length, max_shp_length):
	clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
	sast = USAST(cand_length_list=np.arange(min_shp_length, max_shp_length+1, shp_length_step),
			                         nb_inst_per_class=nb_inst_per_class, shp_step=shp_step,
			                         random_state=None, classifier=clf, radius=radius, use_count=use_count, drop_duplicates=drop_duplicate, scale_before_fit=True)
	return sast

def compute_metrics(y_test, y_preds, y_pred_probas):
	P, R, F1, _ = precision_recall_fscore_support(y_test, y_preds, average='weighted')
	loss = log_loss(y_test, y_pred_probas)
	return P, R, F1, loss

keep_classes = None
class2index = None

if keep_classes:
	class2index = {c:i for i, c in enumerate(keep_classes)}

start_time = time.time()

with open(output_file, "w", buffering=1) as f:
	f.write('Model,Runno,Precision,Recall,F1score,LogLoss,Time,NbKernelNoDuplicates,NbKernelDuplicates\n')

	for runo in range(nb_run_per_dataset):
		models = {}
		X_train_transformed = None
		X_train_transformed_count = None
		X_train_transformed_err = None 
		X_test_transformed = None
		X_test_transformed_count = None
		X_test_transformed_err = None
		y_train = None
		y_test = None
		start_time_runo = time.time()

		for b in bands:
			print('#########> Reading band:', b)
			X_train, X_train_err, y_train, X_test, X_test_err, y_test = read_data(dataset_folder, f'{dataset_name}_{b}', keep_classes)

			labels = np.unique(y_train)

			if keep_classes:
				for c, i in class2index.items():
					y_train[y_train==c] = i 
					y_test[y_test==c] = i

			if X_train_transformed is None: # Display these infos only once
				print('\tdataset loaded')
				print('\tLabels:', labels)
				print('\tTrain shape:', X_train.shape, y_train.shape)
				print('\tTest shape:', X_test.shape, y_test.shape)

			max_shp_length = min(X_train.shape[1], max_shp_length)

			models[b] = build_sast(min_shp_length, max_shp_length)
			models[b].init_usast(X_train, X_train_err, y_train)

			print('sast initialized for band', b, 'nb kernels no duplicates =', models[b].n_kernels_no_duplicates, 'nb kernels =', models[b].n_kernels)

			X_train_t, X_train_t_err = models[b].transform(X_train, X_train_err)

			count_start = X_train_t.shape[1]
			if use_count:
				count_start //= 2

			relative_err = X_train_t_err 
			
			if use_count:
				relative_err /= (X_train_t[:, :count_start] + 1e-10) # Add 1e-10 to avoid zero division
			else:
				relative_err /= (X_train_t + 1e-10) # Add 1e-10 to avoid zero division

			X_train_t = models[b].scaler.fit_transform(X_train_t)

			if use_count:
				X_train_t_err = np.abs(X_train_t[:, :count_start] * relative_err)
				X_train_t_count = X_train_t[:, -count_start:]
				X_train_t = X_train_t[:, :count_start]
			else:
				X_train_t_err = np.abs(X_train_t * relative_err)

			assert np.all(X_train_t_err >= 0), "NO NEGATIVE ERROR ALLOWED"

			print('train transformed for band', b)

			X_test_t, X_test_t_err = models[b].transform(X_test, X_test_err)
			
			count_start = X_test_t.shape[1]
			if use_count:
				count_start //= 2 

			relative_err = X_test_t_err 
			
			if use_count:
				relative_err /= (X_test_t[:, :count_start] + 1e-10) # Add 1e-10 to avoid zero division
			else:
				relative_err /= (X_test_t + 1e-10) # Add 1e-10 to avoid zero division

			X_test_t = models[b].scaler.transform(X_test_t)

			if use_count:
				X_test_t_err = np.abs(X_test_t[:, :count_start] * relative_err)
				X_test_t_count = X_test_t[:, -count_start:]
				X_test_t = X_test_t[:, :count_start]
			else:
				X_test_t_err =  np.abs(X_test_t * relative_err)

			assert np.all(X_test_t_err >= 0), "NO NEGATIVE ERROR ALLOWED"

			print('test transformed for band', b)

			if X_train_transformed is not None:
				X_train_transformed = np.concatenate([X_train_transformed, X_train_t], axis=1)
				X_train_transformed_err = np.concatenate([X_train_transformed_err, X_train_t_err], axis=1)

				X_test_transformed = np.concatenate([X_test_transformed, X_test_t], axis=1)
				X_test_transformed_err = np.concatenate([X_test_transformed_err, X_test_t_err], axis=1)

				if use_count:
					X_train_transformed_count = np.concatenate([X_train_transformed_count, X_train_t_count], axis=1)
					X_test_transformed_count = np.concatenate([X_test_transformed_count, X_test_t_count], axis=1)
			else:
				X_train_transformed = X_train_t
				X_train_transformed_err = X_train_t_err

				X_test_transformed = X_test_t
				X_test_transformed_err = X_test_t_err

				if use_count:
					X_train_transformed_count = X_train_t_count
					X_test_transformed_count = X_test_t_count

				
			print('concatenation done for band', b)

		print("Datasets loaded and SAST transformers initialized")
		X_train_concat = None
		X_test_concat = None
		
		if use_count:
			X_train_concat = np.concatenate([X_train_transformed, X_train_transformed_count, X_train_transformed_err], axis=1)
			X_test_concat = np.concatenate([X_test_transformed, X_test_transformed_count, X_test_transformed_err], axis=1)
		else:
			X_train_concat = np.concatenate([X_train_transformed, X_train_transformed_err], axis=1)
			X_test_concat = np.concatenate([X_test_transformed, X_test_transformed_err], axis=1)

		X_train_concat, y_train = shuffle(X_train_concat, y_train)

		# Using Uncertain Gaussian Naive Bayes classifier
		X_utrain = np.zeros((X_train_transformed.shape[0], X_train_concat.shape[1] - X_train_transformed_err.shape[1], 2))
		X_utrain[:, :, 0] = X_train_concat[:, :-X_train_transformed_err.shape[1]] # take the best guesses and the counts
		X_utrain[:, :X_train_transformed.shape[1], 1] = X_train_concat[:, -X_train_transformed_err.shape[1]:] # take the uncertainties of best guesses

		assert np.all(X_utrain[:,:X_train_transformed.shape[1], 1] >= 0), "Error cannot be be negative"

		X_utest = np.zeros((X_test_transformed.shape[0], X_test_concat.shape[1] - X_test_transformed_err.shape[1], 2))
		X_utest[:, :, 0] = X_test_concat[:, :-X_test_transformed_err.shape[1]]
		X_utest[:, :X_test_transformed.shape[1], 1] = X_test_concat[:, -X_test_transformed_err.shape[1]:]

		assert np.all(X_utest[:, :X_test_transformed.shape[1], 1] >= 0), "Error cannot be negative" 

		ugnb_clf = UGaussianNB()
		ugnb_clf.fit(X_utrain, y_train)

		ugnb_probabilities = ugnb_clf.predict_proba(X_utest)
		ugnb_predictions = ugnb_clf.predict(X_utest)

		P_ugnb, R_ugnb, F1_ugnb, loss_ugnb = compute_metrics(y_test, ugnb_predictions, ugnb_probabilities)

		# Using Ridge classifier with LOO-CV
		ridge_clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
		ridge_clf.fit(X_train_concat, y_train)

		ridge_probabilities = ridge_clf._predict_proba_lr(X_test_concat)
		ridge_predictions = ridge_clf.predict(X_test_concat)

		P_ridge, R_ridge, F1_ridge, loss_ridge = compute_metrics(y_test, ridge_predictions, ridge_probabilities)

		# Using Random Forest classifier
		rf_clf = RandomForestClassifier(n_jobs=-1, criterion='entropy')
		rf_clf.fit(X_train_concat, y_train)

		rf_probabilities = rf_clf.predict_proba(X_test_concat)
		rf_predictions = rf_clf.predict(X_test_concat)

		P_rf, R_rf, F1_rf, loss_rf = compute_metrics(y_test, rf_predictions, rf_probabilities)

		# Using Gradient Boosting classifier
		# gb_clf = GradientBoostingClassifier()
		# gb_clf.fit(X_train_transformed, y_train)

		# gb_probabilities = gb_clf.predict_proba(X_test_transformed)
		# gb_predictions = gb_clf.predict(X_test_transformed)

		# P_gb, R_gb, F1_gb, loss_gb = compute_metrics(y_test, gb_predictions, gb_probabilities)

		# Using XGBoost classifier
		param_dist = {'objective':'binary:logistic', 'n_estimators': rf_clf.n_estimators, 'n_jobs':n_jobs}
		xgb_clf = xgb.XGBClassifier(**param_dist)
		xgb_clf.fit(X_train_concat, y_train)

		xgb_probabilities = xgb_clf.predict_proba(X_test_concat)
		xgb_predictions = xgb_clf.predict(X_test_concat)

		P_xgb, R_xgb, F1_xgb, loss_xgb = compute_metrics(y_test, xgb_predictions, xgb_probabilities)

		finish_time_runo = time.time() - start_time_runo

		f.write(f"Ridge,{runo},{P_ridge},{R_ridge},{F1_ridge},{loss_ridge},{finish_time_runo},{models[b].n_kernels_no_duplicates},{models[b].n_kernels}\n")
		f.write(f"RF,{runo},{P_rf},{R_rf},{F1_rf},{loss_rf},{finish_time_runo},{models[b].n_kernels_no_duplicates},{models[b].n_kernels}\n")
		f.write(f"XGBoost,{runo},{P_xgb},{R_xgb},{F1_xgb},{loss_xgb},{finish_time_runo},{models[b].n_kernels_no_duplicates},{models[b].n_kernels}\n")
		f.write(f"UGNB,{runo},{P_ugnb},{R_ugnb},{F1_ugnb},{loss_ugnb},{finish_time_runo},{models[b].n_kernels_no_duplicates},{models[b].n_kernels}\n")
		# f.write(f"\tGBoost,{runo},{P_gb},{R_gb},{F1_gb},{loss_gb},{finish_time_runo},{models[b].n_kernels_no_duplicates},{models[b].n_kernels}\n")

		for b in bands:
			joblib.dump(models[b], f'models/u_{dataset_name}_sast_{b}_c{use_count}_d{drop_duplicate}_s{shp_step}_runno{runo}.joblib')
		joblib.dump(ridge_clf, f'models/ridge_u_{dataset_name}_c{use_count}_d{drop_duplicate}_s{shp_step}_runno{runo}.joblib')
		joblib.dump(rf_clf, f'models/rf_u_{dataset_name}_c{use_count}_d{drop_duplicate}_s{shp_step}_runno{runo}.joblib')
		joblib.dump(xgb_clf, f'models/xgb_u_{dataset_name}_c{use_count}_d{drop_duplicate}_s{shp_step}_runno{runo}.joblib')
		joblib.dump(ugnb_clf, f'models/ugnb_u_{dataset_name}_c{use_count}_d{drop_duplicate}_s{shp_step}_runno{runo}.joblib')
		# joblib.dump(gb_clf, f'models/gb_u_{dataset_name}_c{use_count}_d{drop_duplicate}_s{shp_step}_runno{runo}.joblib')

finish_time = time.time() - start_time

print("Took:", finish_time , "sec")

# python scripts/experiment_plasticc_u.py /home/mimbouop/Codes/UST-Python/dataset/plasticc_5d_dataset plassticc_5d plasticc_test.txt -r 4 -d -M 6 -inst 2
