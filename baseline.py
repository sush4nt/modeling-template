import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report, make_scorer, auc
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score, cross_validate

class BaselineModeling(object):
    def __init__(self, estimator, X_train, y_train, X_test, y_test):
        self.estimator = estimator
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        print("=="*30, self.estimator, "=="*30)

    @staticmethod
    def roundoff(num, precision=3):
        return np.round(num, precision)
    
    def fit(self):
        self.estimator.fit(self.X_train, self.y_train)

        self.preds = self.estimator.predict(self.X_test)
        self.pred_probs = self.estimator.predict_proba(self.X_test)
        print("----->   ", "finished training the model")
        
    def evaluate(self):
        try:
            self.train_accuracy = BaselineModeling.roundoff(accuracy_score(self.y_train, self.estimator.predict(self.X_train)))
            self.test_accuracy = BaselineModeling.roundoff(accuracy_score(self.y_test, self.preds))
            self.train_f1 = BaselineModeling.roundoff(f1_score(self.y_train, self.estimator.predict(self.X_train)))
            self.test_f1 = BaselineModeling.roundoff(f1_score(self.y_test, self.preds))
            self.precision = BaselineModeling.roundoff(precision_score(self.y_test, self.preds))
            self.recall = BaselineModeling.roundoff(recall_score(self.y_test, self.preds))
            self.roc_auc = BaselineModeling.roundoff(roc_auc_score(self.y_test, self.preds))
            self.c_matrix = confusion_matrix(self.y_test, self.preds)
            self.classify_report = classification_report(self.y_test, self.preds)
            
            print("=="*20, "Train accuracy:", self.train_accuracy, "=="*20)
            print("=="*20, "Test accuracy:", self.test_accuracy, "=="*20)
            print("=="*20, "Train F1 Score:", self.train_f1, "=="*20)
            print("=="*20, "Test F1 Score:", self.test_f1, "=="*20)
            print("=="*20, "Confusion matrix: ", "=="*30, "\n", self.c_matrix)
            print("=="*20, "Precision Score:", self.precision, "=="*20)
            print("=="*20, "Recall Score:", self.recall, "=="*20)
            print("=="*20, "ROC AUC Score:", self.roc_auc, "=="*20)
            self._plot_roc()
            print("=="*20, "Classification report: ", "=="*20, "\n", self.classify_report)
            self._plot_pr()
            
            self.results = self.log_metrics()
            print("=="*20, "metrics logged", "=="*20)
        except Exception as err:
            print("Error: ", err)
            print("Please checking if model.fit happened correctly")

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
        self.pr_auc = BaselineModeling._calculate_pr_auc(precisions, recalls)
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

        pr_df = pd.DataFrame({'precision': precisions[0:threshold_boundary],
                              'recall':recalls[0:threshold_boundary],
                              'threshold':thresholds
                             })
        pr_df['diff'] = np.abs(pr_df['precision']-pr_df['recall'])
        self.threshold_tuned = pr_df[pr_df['diff']==pr_df['diff'].min()]['threshold'].iloc[0]
        print("----->   ", "Optimal threshold: ", self.threshold_tuned)

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