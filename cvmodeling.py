import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# import seaborn as sns
# sns.set_style("whitegrid")
# from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold, RandomizedSearchCV
# from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report, make_scorer, auc
# from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from baseline import *

class CVModeling(BaselineModeling):
    def __init__(self, estimator, X_train, y_train, X_test, y_test, scoring='f1', cv_type='stratifiedkfold', n_splits=5, param_grid=None):
        super().__init__(estimator, X_train, y_train, X_test, y_test)
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.param_grid = param_grid
        self.scoring = scoring
        self.SEED = 1234
        self.SHUFFLE = True

    @staticmethod
    def pr_auc(y_true, y_preds):
        precision, recall, _ = precision_recall_curve(y_true, y_preds)
        return auc(recall, precision)

    @staticmethod
    def custom_scorer(scoring):
        if scoring == 'precision':
            return make_scorer(precision_score, average='binary')
        elif scoring == 'recall':
            return make_scorer(recall_score, average='binary')
        elif scoring == 'f1':
            return make_scorer(f1_score, average='binary')
        elif scoring == 'roc_auc':
            return 'roc_auc'
        elif scoring == 'pr_auc':
            return make_scorer(CVModeling.pr_auc, needs_proba=True)
        else:
            raise "Invalid scoring metric"
    
    def perform_cv(self):
        scorer = CVModeling.custom_scorer(self.scoring)
        if self.cv_type == 'kfold':
            cv = KFold(n_splits=self.n_splits, shuffle=self.SHUFFLE, random_state=self.SEED)
            results = cross_val_score(self.estimator, self.X_train, self.y_train, cv=cv, scoring=scorer, return_train_score=True)
            print("=="*20, "K-Fold CV Train score:", np.mean(results['train_score']), "=="*20)
            print("=="*20, "K-Fold CV Test score:", np.mean(results['test_score']), "=="*20)
            return np.mean(results['train_score']), np.mean(results['test_score'])
        elif self.cv_type == 'stratifiedkfold':
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.SHUFFLE, random_state=self.SEED)
            results = cross_validate(self.estimator, self.X_train, self.y_train, cv=cv, scoring=scorer, return_train_score=True)
            print("=="*20, "Stratified K-Fold CV Train score:", np.mean(results['train_score']), "=="*20)
            print("=="*20, "Stratified K-Fold CV Test score:", np.mean(results['test_score']), "=="*20)
            return np.mean(results['train_score']), np.mean(results['test_score'])
        elif self.cv_type == 'gridsearchcv':
            if self.param_grid is None:
                raise ValueError("param_grid must be provided for GridSearchCV")
            grid_search = GridSearchCV(self.estimator, self.param_grid, cv=self.n_splits, scoring=scorer)
            grid_search.fit(self.X_train, self.y_train)
            print("=="*20, "Grid Search CV best cv results: ", grid_search.best_params_, grid_search.best_score_, "=="*20)
            print("=="*20, "Grid Search CV best estimator: ", grid_search.best_estimator_, "=="*20)
            print("=="*20, "Grid Search CV best params: ", grid_search.best_params_, "=="*20)
            print("=="*20, "Grid Search CV best score: ", grid_search.best_score_, "=="*20)
            return grid_search.cv_results_
        else:
            raise ValueError(f"Unknown cv_type: {self.cv_type}. Valid types are 'kfold', 'stratifiedkfold', 'gridsearchcv'")