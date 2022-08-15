# Explainable Classification of Uncertain Astronomical Time Series

This repository contains the source code and results of our work on the effective classification of uncertain astronomical time series with explainability in mind. We propose the Uncertain Scalable and Accurate Subsequence Transform (uSAST), derived from both the Uncertain Shapelet Transform and the Scalable and Accurate Subsequence Transform to accurately classify the PLAsTiCC astronomical uncertain time series dataset without using techniques such as data augmentation and data oversampling despite the fact that dataset contains 14 unbalance classes. Furthermore, uSAST explainability allows astrophysicists to understand the method and the predictions. 

## Repository's content
- [experiment_plasticc_u.py](experiment_plasticc_u.py): Script used to run SAST and uSAST experiment.
- [result-plasticc-paper.ipynb](result-plasticc-paper.ipynb)`: Jupyter notebook used to summarised the results and to generate confusion matrices.
- [explain-plasticc.ipynb](explain-plasticc.ipynb): Jupyter notebook used to perform explainability.
- [plasticc_train_meta.csv](plasticc_train_meta.csv): Dataset's metadata.
- [plasticc_train_test_split.csv](plasticc_train_test_split.csv): Train and test split of the dataset.
- [plasticc-to-arff.ipynb](plasticc-to-arff.ipynb): Script used to convert the dataset from arff to csv.
- `sast/`: Folder containing SAST+ and uSAST source code.
- `XEM/`: Folder containing source code for XEM experiment.
- `results/`: Folder containing raw results of the experiment.
- `images/`: Folder containing confusion matrices and most important features images

## Dependencies
- matrixprofile==1.1.10
- scikit-learn==1.0.2
- sktime==0.5.3
- xgboost==1.5.1
- seaborn=0.11.0
- pandas=1.1.0
- numba=0.50.1
- numpy=1.19.2
- pyarrow=9.0.0
- pyyaml=6.0
- lcensemble=0.2.7


## Usage
```python
import numpy as np
from sast.usast import *
import xgboost as xgb

param_dist = {'objective':'binary:logistic'}
xgb_clf = xgb.XGBClassifier(**param_dist)
usast_xgb = USAST(cand_length_list=np.arange(min_shp_length, max_shp_length+1),
		          nb_inst_per_class=nb_inst_per_class, 
		          random_state=None, classifier=xgb_clf)

usast_xgb.fit(X_train, X_train_err, y_train)

prediction = usast_xgb.predict(X_test, X_test_err)
```

## Reproduce the experiment

To reproduce the experiment, use the script [experiment_plasticc_u.py](experiment_plasticc_u.py) as follows:
````
experiment_plasticc_u.py [-h] [-b BANDS] [-r RADIUS] [-c] [-d]
                                [-run NB_RUN_PER_DATASET]
                                [-inst NB_INST_PER_CLASS] [-m MIN_SHP_LENGTH]
                                [-ls SHP_LENGTH_STEP] [-M MAX_SHP_LENGTH]
                                [-s SHP_STEP] [-k KEEP_CLASSES]
                                dataset_folder dataset_name output_file

positional arguments:
  dataset_folder        The folder containing the datasets
  dataset_name          The name of the dataset, the part before the
                        _[dimension]
  output_file           The file in which the results will be written

optional arguments:
  -h, --help            show this help message and exit
  -b BANDS, --bands BANDS
                        String of dimensions to consider
  -r RADIUS, --radius RADIUS
                        The neighborhood radius in which subsequences are
                        considered similar
  -c, --use_count       Whether to count the number of times a subsequence
                        appears in a time series
  -d, --drop_duplicate  Whether to drop duplicate subsequences from reference
                        time series
  -run NB_RUN_PER_DATASET, --nb_run_per_dataset NB_RUN_PER_DATASET
                        The number of times each dataset must be run
  -inst NB_INST_PER_CLASS, --nb_inst_per_class NB_INST_PER_CLASS
                        Maximum number of references times series to use per class
  -m MIN_SHP_LENGTH, --min_shp_length MIN_SHP_LENGTH
                        The minimum subsequence length to consider
  -ls SHP_LENGTH_STEP, --shp_length_step SHP_LENGTH_STEP
                        The subsequence length stride to consider
  -M MAX_SHP_LENGTH, --max_shp_length MAX_SHP_LENGTH
                        The maximum subsequence length to consider
  -s SHP_STEP, --shp_step SHP_STEP
                        Stride value for shapelet extraction
  -k KEEP_CLASSES, --keep_classes KEEP_CLASSES
                        List of classes to classify. The classes not in the list will be ignore
````






