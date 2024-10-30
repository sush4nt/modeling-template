import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score, \
                            f1_score, accuracy_score, confusion_matrix, classification_report, make_scorer, auc \
                            , mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from baseline_modeling import BaselineModeling, BaselineClfModeling, BaselineRegModeling

class ClassificationCV(BaselineClfModeling):
    def __init__(
        self, 
        estimator, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        scoring='f1', 
        cv_type='stratifiedkfold', 
        n_splits=5, 
        param_grid=None):
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
            return make_scorer(ClassificationCV.pr_auc, needs_proba=True)
        else:
            raise "Invalid scoring metric"
    
    def perform_cv(self):
        scorer = ClassificationCV.custom_scorer(self.scoring)
        if self.cv_type == 'kfold':
            cv = KFold(n_splits=self.n_splits, shuffle=self.SHUFFLE, random_state=self.SEED)
            results = cross_validate(self.estimator, self.X_train, self.y_train, cv=cv, scoring=scorer, return_train_score=True)
            print("=="*20, "K-Fold CV Train score:", "=="*20, "\n", np.mean(results['train_score']))
            print("=="*20, "K-Fold CV Test score:", "=="*20, "\n", np.mean(results['test_score']))
            return np.mean(results['train_score']), np.mean(results['test_score'])
        elif self.cv_type == 'stratifiedkfold':
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.SHUFFLE, random_state=self.SEED)
            results = cross_validate(self.estimator, self.X_train, self.y_train, cv=cv, scoring=scorer, return_train_score=True)
            print("=="*20, "Stratified K-Fold CV Train score:", "=="*20, "\n", np.mean(results['train_score']))
            print("=="*20, "Stratified K-Fold CV Test score:", "=="*20, "\n", np.mean(results['test_score']))
            return np.mean(results['train_score']), np.mean(results['test_score'])
        elif self.cv_type == 'gridsearchcv':
            if self.param_grid is None:
                raise ValueError("param_grid must be provided for GridSearchCV")
            grid_search = GridSearchCV(self.estimator, self.param_grid, cv=self.n_splits, scoring=scorer)
            grid_search.fit(self.X_train, self.y_train)
            print("=="*20, "Grid Search CV best cv results: ", "=="*20, "\n", grid_search.best_params_, grid_search.best_score_)
            print("=="*20, "Grid Search CV best estimator: ", "=="*20, "\n", grid_search.best_estimator_)
            print("=="*20, "Grid Search CV best params: ", "=="*20, "\n", grid_search.best_params_)
            print("=="*20, "Grid Search CV best score: ", "=="*20, "\n", grid_search.best_score_)
            return grid_search.cv_results_
        else:
            raise ValueError(f"Unknown cv_type: {self.cv_type}. Valid types are 'kfold', 'stratifiedkfold', 'gridsearchcv'")
        
class RegressionCV(BaselineRegModeling):
    def __init__(
        self, 
        estimator, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        scoring='r2', 
        cv_type='kfold', 
        n_splits=5, 
        param_grid=None):
        super().__init__(estimator, X_train, y_train, X_test, y_test)
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.param_grid = param_grid
        self.scoring = scoring
        self.SEED = 1234
        self.SHUFFLE = True

    @staticmethod
    def custom_scorer(scoring):
        if scoring == 'neg_mean_squared_error':
            return make_scorer(mean_squared_error, greater_is_better=False)
        elif scoring == 'neg_mean_absolute_error':
            return make_scorer(mean_absolute_error, greater_is_better=False)
        elif scoring == 'r2':
            return make_scorer(r2_score)
        elif scoring == 'neg_mean_absolute_percentage_error':
            return make_scorer(mean_absolute_percentage_error, greater_is_better=False)
        else:
            raise ValueError("Invalid scoring metric")

    def perform_cv(self):
        scorer = RegressionCV.custom_scorer(self.scoring)

        if self.cv_type == 'kfold':
            cv = KFold(n_splits=self.n_splits, shuffle=self.SHUFFLE, random_state=self.SEED)
            results = cross_validate(self.estimator, self.X_train, self.y_train, cv=cv, scoring=scorer, return_train_score=True)
            if self.scoring=='r2':
                train_score = np.mean(results['train_score'])
                test_score = np.mean(results['test_score'])
            else:
                train_score = np.mean(-1*results['train_score'])
                test_score = np.mean(-1*results['test_score'])
            print("=="*20, "K-Fold CV Train score:", "=="*20, "\n", train_score)
            print("=="*20, "K-Fold CV Test score:", "=="*20, "\n", test_score)
            return train_score, test_score

        elif self.cv_type == 'gridsearchcv':
            if self.param_grid is None:
                raise ValueError("param_grid must be provided for GridSearchCV")
            grid_search = GridSearchCV(self.estimator, self.param_grid, cv=self.n_splits, scoring=scorer)
            grid_search.fit(self.X_train, self.y_train)
            print("=="*20, "Grid Search CV best cv results: ", "=="*20, "\n", grid_search.best_params_, grid_search.best_score_)
            print("=="*20, "Grid Search CV best estimator: ", "=="*20, "\n", grid_search.best_estimator_)
            print("=="*20, "Grid Search CV best params: ", "=="*20, "\n", grid_search.best_params_)
            print("=="*20, "Grid Search CV best score: ", "=="*20, "\n", grid_search.best_score_)
            return grid_search.cv_results_

        else:
            raise ValueError(f"Unknown cv_type: {self.cv_type}. Valid types are 'kfold', 'gridsearchcv'")