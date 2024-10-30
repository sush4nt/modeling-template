import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from textwrap import dedent
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report, make_scorer, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score, cross_validate

class BaselineModeling:
    def __init__(self, estimator, X_train, y_train, X_test, y_test):
        self.estimator = estimator
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    @staticmethod
    def roundoff(num, precision=3):
        return np.round(num, precision)

    def log_metrics(self, metrics_dict):
        return {**metrics_dict, 'Model name': self.estimator.__class__.__name__}

class BaselineClfModeling(BaselineModeling):
    def fit(self):
        self.estimator.fit(self.X_train, self.y_train)

        self.preds = self.estimator.predict(self.X_test)
        self.pred_probs = self.estimator.predict_proba(self.X_test)

    @staticmethod
    def _calculate_pr_auc(precision, recall):
        """Calculate the area under the Precision-Recall curve."""
        sorted_indices = sorted(range(len(recall)), key=lambda i: recall[i])
        sorted_recall = [recall[i] for i in sorted_indices]
        sorted_precision = [precision[i] for i in sorted_indices]
    
        area = 0
        for i in range(1, len(sorted_recall)):
            area += 0.5 * (sorted_recall[i] - sorted_recall[i-1]) * (sorted_precision[i] + sorted_precision[i-1])
    
        return area

    def _plot_roc(self):
        fpr, tpr, thresholds = roc_curve(self.y_test, self.estimator.predict_proba(self.X_test)[:,1])
        plt.figure(figsize=(5,3))
        plt.plot(fpr, tpr, label='AU-ROC = %0.2f)' % self.roc_auc)
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AU-ROC')
        plt.legend(loc="lower right")
        plt.show()

    def _plot_pr(self):
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, self.estimator.predict_proba(self.X_test)[:,1])
        print("Area under P-R Curve: ", self.pr_auc)
        plt.figure(figsize=(5,3))
        plt.plot(recalls, precisions, marker='.', label='AU-PR')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('AU-PR')
        plt.legend()
        plt.show()

        threshold_boundary = thresholds.shape[0]
        plt.figure(figsize=(5,3))
        plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
        plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')
        start,end=plt.xlim()
        plt.xticks(np.round(np.arange(start,end,0.1),2))
        plt.xlabel('Threshold Value')
        plt.ylabel('Precision and Recall Value')
        plt.title('Thresholds vs PR')
        plt.legend()
        plt.show()

    def _plot_residuals(self):
        residuals = self.y_test - self.pred_probs[:, 1]
        plt.figure(figsize=(5,3))
        plt.hist(residuals, alpha=0.75)
        plt.ylabel("Residuals")
        plt.title("Residuals distribution")
        plt.show()
        
    def evaluate(self, verbose=True):
        try:
            self.train_accuracy = BaselineClfModeling.roundoff(accuracy_score(self.y_train, self.estimator.predict(self.X_train)))
            self.test_accuracy = BaselineClfModeling.roundoff(accuracy_score(self.y_test, self.preds))
            self.train_f1 = BaselineClfModeling.roundoff(f1_score(self.y_train, self.estimator.predict(self.X_train)))
            self.test_f1 = BaselineClfModeling.roundoff(f1_score(self.y_test, self.preds))
            self.precision = BaselineClfModeling.roundoff(precision_score(self.y_test, self.preds))
            self.recall = BaselineClfModeling.roundoff(recall_score(self.y_test, self.preds))
            self.roc_auc = BaselineClfModeling.roundoff(roc_auc_score(self.y_test, self.preds))
            self.c_matrix = confusion_matrix(self.y_test, self.preds)
            self.classify_report = classification_report(self.y_test, self.preds)

            precisions, recalls, thresholds = precision_recall_curve(self.y_test, self.estimator.predict_proba(self.X_test)[:,1])
            threshold_boundary = thresholds.shape[0]
            pr_df = pd.DataFrame({'precision': precisions[0:threshold_boundary],
                                'recall':recalls[0:threshold_boundary],
                                'threshold':thresholds
                                })
            pr_df['diff'] = np.abs(pr_df['precision']-pr_df['recall'])
            self.threshold_tuned = pr_df[pr_df['diff']==pr_df['diff'].min()]['threshold'].iloc[0]
            self.pr_auc = BaselineClfModeling._calculate_pr_auc(precisions, recalls)

            if verbose:
                print(dedent(f"""
                ====================================================================================================
                                                        {self.estimator.__class__.__name__}
                ====================================================================================================
                ======================================== Model Performance Metrics ========================================
                Train accuracy:           {self.train_accuracy:.2f}
                Test accuracy:            {self.test_accuracy:.2f}
                Train F1 Score:           {self.train_f1:.3f}
                Test F1 Score:            {self.test_f1:.3f}
                ====================================================================================================

                ======================================== Confusion Matrix ========================================
                {self.c_matrix}
                ====================================================================================================

                ======================================== Additional Metrics ========================================
                Precision Score:          {self.precision:.3f}
                Recall Score:             {self.recall:.3f}
                ROC AUC Score:            {self.roc_auc:.3f}
                ====================================================================================================

                ======================================== Classification Report ========================================
                {self.classify_report}
                ====================================================================================================

                ======================================== Optimal Threshold ========================================
                {self.threshold_tuned}
                ====================================================================================================
                """))
            
            self.results = self.log_metrics()
        except Exception as err:
            print("Error: ", err)
            print("Please checking if model.fit happened correctly")

    def display_plots(self):
        try:
            print("\n" + "="*40 + " Receiver Operating Characteristis (ROC) " + "="*40)
            self._plot_roc()
            print("\n" + "="*40 + " Precision-Recall (PR) Curve " + "="*40)
            self._plot_pr()
            print("\n" + "="*40 + " Residuals " + "="*40)
            self._plot_residuals()
        except Exception as err:
                print("Error: ", err)

    def log_metrics(self):
        metrics = {
                    'Model name': self.estimator.__class__.__name__,
                    'Train accuracy': self.train_accuracy,
                    'Test accuracy': self.test_accuracy,
                    "Train F1 Score" : self.train_f1,
                    "Test F1 Score" : self.test_f1,
                    "Precision" : self.precision,
                    "Recall" : self.recall,
                    "Confusion matrix" : self.c_matrix,
                    "Classification report" : self.classify_report,
                    "ROC AUC Score" : self.roc_auc,
                    "PR AUC Score" : self.pr_auc,
                    "Optimal threshold" : self.threshold_tuned
        }
        return metrics

class BaselineRegModeling(BaselineModeling):
    def fit(self):
        self.estimator.fit(self.X_train, self.y_train)
        self.preds = self.estimator.predict(self.X_test)

    def _plot_residuals(self):
        residuals = self.y_test - self.preds
        plt.figure(figsize=(7, 5))
        plt.scatter(self.preds, residuals, alpha=0.75)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted values")
        plt.show()

    def _plot_predictions(self):
        plt.figure(figsize=(7, 5))
        plt.scatter(self.y_test, self.preds, alpha=0.75)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.xlabel("Actual values")
        plt.ylabel("Predicted values")
        plt.title("Actual vs Predicted values")
        plt.show()
        
    def evaluate(self, verbose=True):
        try:
            self.train_rmse = BaselineRegModeling.roundoff(np.sqrt(mean_squared_error(self.y_train, self.estimator.predict(self.X_train))))
            self.test_rmse = BaselineRegModeling.roundoff(np.sqrt(mean_squared_error(self.y_test, self.preds)))
            self.train_mae = BaselineRegModeling.roundoff(mean_absolute_error(self.y_train, self.estimator.predict(self.X_train)))
            self.test_mae = BaselineRegModeling.roundoff(mean_absolute_error(self.y_test, self.preds))
            self.train_r2 = BaselineRegModeling.roundoff(r2_score(self.y_train, self.estimator.predict(self.X_train)))
            self.test_r2 = BaselineRegModeling.roundoff(r2_score(self.y_test, self.preds))
            self.train_mape = BaselineRegModeling.roundoff(mean_absolute_percentage_error(self.y_train, self.estimator.predict(self.X_train)))
            self.test_mape = BaselineRegModeling.roundoff(mean_absolute_percentage_error(self.y_test, self.preds))
            
            if verbose:
                print(dedent(f"""
                ====================================================================================================
                                                    {self.estimator.__class__.__name__}
                ====================================================================================================
                ======================================== Model Performance Metrics ========================================
                Train RMSE:              {self.train_rmse:.3f}
                Test RMSE:               {self.test_rmse:.3f}
                Train MAE:               {self.train_mae:.3f}
                Test MAE:                {self.test_mae:.3f}
                Train R^2 Score:         {self.train_r2:.3f}
                Test R^2 Score:          {self.test_r2:.3f}
                Train MAPE:              {self.train_mape:.3f}
                Test MAPE:               {self.test_mape:.3f}
                ====================================================================================================
                """))
            self.results = self.log_metrics()
        except Exception as err:
            print("Error: ", err)
            print("Please check if model.fit happened correctly")

    def display_plots(self):
        try:
            print("\n" + "="*40 + " Residuals " + "="*40)
            self._plot_residuals()
            print("\n" + "="*40 + " Predictions v/s Actual values " + "="*40)
            self._plot_predictions()
        except Exception as err:
            print("Error: ", err)

    def log_metrics(self):
        metrics = {
            'Model name': self.estimator.__class__.__name__,
            'Train RMSE': self.train_rmse,
            'Test RMSE': self.test_rmse,
            'Train MAE': self.train_mae,
            'Test MAE': self.test_mae,
            'Train R^2 Score': self.train_r2,
            'Test R^2 Score': self.test_r2,
            'Train MAPE': self.train_mape,
            'Test MAPE': self.test_mape
        }
        return metrics