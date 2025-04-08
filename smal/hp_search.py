import copy
import os
import pandas as pd
import joblib
from rdkit import Chem
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn import metrics
from scipy import stats
import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut, KFold, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def else_none(
    fun,
):
    """
    A useful wrapper that attempts to call the
    given function <fun>, but returns None
    in case any error is encountered.
    """
 
    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except:
            return None
 
    return wrapper



def spearmanr(y_true,y_pred):
    return stats.spearmanr(y_true,y_pred)[0]

def regression_metrics():
    return [
        metrics.r2_score,
        spearmanr,
        metrics.mean_absolute_error,
        metrics.mean_squared_error,
    ]





def hp_search_helper(model,df_train,df_val):
    """
    Simple hyperparameter search method that provides sensible defaults 
    for common model types.
    """
    df = pd.concat([df_train,df_val])
    test_fold = np.array([-1]*len(df_train) + [0]*len(df_val))
    PARAM_GRID = {
    'SVC': {'model__C': [0.1, 1, 10, 100], 'model__kernel': ['rbf'], 'model__class_weight': ['balanced'], 'model__gamma': ['scale', 'auto', 1, 0.001, 0.01, 0.1]},
    'RandomForestClassifier': {'n_estimators': [400,700,1000], 'class_weight': ['balanced'], 'min_samples_leaf': [2,3]},
    # TODO:
    #'gbc': {'model__n_estimators': [400,700,1000], 'model__min_samples_leaf': [2,3], 'model__loss': ['log_loss', 'exponential']},
    'SVR': {'model__C': [0.1, 1, 10, 100], 'model__kernel': ['rbf'],  'model__gamma': ['scale', 'auto', 1, 0.001, 0.01, 0.1]},
    'RandomForestRegressor': {'model__n_estimators': [400,700,1000], 'model__max_depth':[30, 50], 'model__n_jobs':[32], 'model__random_state': [123]},#'model__min_samples_leaf': [2,3] },
    # TODO:
    #'gbr': {'model__n_estimators': [400,700,1000], 'model__min_samples_leaf': [2,3],},
    }

    SCORING = {
        'SVC': 'f1',
        'RandomForestClasisifer': 'f1',
        'RandomForestRegressor': "r2",
        'SVR': "r2",
    }

    param_grid = PARAM_GRID[model.__class__.__name__]
    ps = PredefinedSplit(test_fold=test_fold)
    pipe = GridSearchCV(Pipeline([('scaler', StandardScaler()), ('model', model.__class__())]), param_grid=param_grid, refit=True, cv=ps, scoring=SCORING[model.__class__.__name__])
    pipe.fit(np.vstack(df['X']),df['y'])
    return pipe.best_estimator_

    

def run_benchmark(model,feat_fun,hp_search=None,splt_col="split_5"):
    """
    Runs the benchmark on the given model utilizing
    the given featurizer feat_fun and doing
    the spliting on the providing splt_col.

    Here, hp_search is a function that takes a model, the train
    dataframe and the val dataframe and returns a model with
    the optimal hyperparameters.
    features in the dataframe are called X,
    the target is called y.

    Uses the GNN template splitting scheme:
    Train Train Train Val Test
    Test Train Train Train Val 
    Val Test Train Train Train  
    Train Val Test Train Train 
    Train Train Val Test Train  
    
    """
    scores = []
    df = pd.read_csv(CVI_DATASET)
    if str(os.environ.get("CVI_DOWNSAMPLE",None)) in ["1","t","T","true","True"]:
        df = df.sample(5000,random_state=123)
    df["mol"] = df["smiles"].apply(else_none(Chem.MolFromSmiles))
    try:
        df["X"] = list(feat_fun(df["mol"].tolist()))
    except:
        df["X"] = df["mol"].apply(feat_fun)
    df["y"] = df["y"].apply(np.log10)
    df_pred = df[["y"]].copy(deep=True)
    df_pred.loc[:, "prediction_y"] = np.nan

    original_model = model
    n_cvs = df[splt_col].nunique()
    for spl_test in tqdm.tqdm(range(n_cvs)):
        model = copy.deepcopy(original_model)
        spl_val = (spl_test - 1) % n_cvs
        df_test = df[df[splt_col] == spl_test]
        df_val = df[df[splt_col] == spl_val]
        df_train = df[~df[splt_col].isin([spl_test,spl_val])]
        ind_test = df[df[splt_col] == spl_test].index

        assert len(df_test)
        assert len(df_val)
        assert len(df_train)
        assert len(df) == sum(map(len,(df_test,df_val,df_train)))

        if hp_search is not None:
            model = hp_search(model,df_train,df_val)

        df_train_val = pd.concat([df_train,df_val])

        # final retrain including validation set to make optimal
        # use of the data available
        model.fit(np.vstack(df_train_val["X"]),df_train_val["y"])
        df_pred.loc[ind_test, "prediction_y"] = model.predict(np.vstack(df_test["X"]))
        for splt_type,df_splt in [("train",df_train),("test",df_test)]:
            this_scores = {"split":splt_type,"fold":spl_test}
            for metric in regression_metrics():
                y_pred = model.predict(np.vstack(df_splt["X"]))
                y_true = df_splt["y"]
                metric_val = metric(y_true,y_pred)
                this_scores[metric.__name__] = metric_val
            scores.append(this_scores)
    scores = pd.DataFrame(scores)
    try:
        scores["model_name"] = original_model.__class__.__name__
    except:
        scores["model_name"] = original_model.__name__

    scores["feature_name"] = feat_fun.__name__
    return scores, df_pred






