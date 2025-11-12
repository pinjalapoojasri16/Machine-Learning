import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')
import logging
from log_file import setup_logging
logger = setup_logging('main')
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
class HEART:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info("The Data loaded Successfully")
            logger.info(f"We have: {self.df.shape[0]} Rows and {self.df.shape[1]} Columns")
            logger.info(f"Missing values:\n{self.df.isnull().sum()}")
            self.x = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=42)
            logger.info(f'training sample {len(self.x_train)},{len(self.y_train)}  testing sample   {len(self.x_test)},{len(self.y_test)}')
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")
    def knn_algo(self):
        try:
            self.reg_knn=KNeighborsClassifier(n_neighbors = 2)
            self.reg_knn.fit(self.x_train,self.y_train)
            logger.info(f'==============KNN algorithm================')
            logger.info(f'train accuracy : {accuracy_score(self.y_train,self.reg_knn.predict(self.x_train))*100}')
            logger.info(f'test accuracy : {accuracy_score(self.y_test, self.reg_knn.predict(self.x_test))*100}')
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")
    def logistic_regression(self):
        try:
            self.reg_lr = LogisticRegression()
            self.reg_lr.fit(self.x_train, self.y_train)
            logger.info(f'==============Logistic Regression================')
            logger.info(f'train accuracy : {accuracy_score(self.y_train, self.reg_lr.predict(self.x_train)) * 100}')
            logger.info(f'test accuracy : {accuracy_score(self.y_test, self.reg_lr.predict(self.x_test)) * 100}')

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")
    def naive_bayes(self):
        try:
            self.reg_nb = GaussianNB()
            self.reg_nb.fit(self.x_train, self.y_train)
            logger.info(f'==============Naive bayes================')
            logger.info(f'train accuracy : {accuracy_score(self.y_train, self.reg_nb.predict(self.x_train)) * 100}')
            logger.info(f'test accuracy : {accuracy_score(self.y_test, self.reg_nb.predict(self.x_test)) * 100}')

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")
    def Decision_Tree(self):
        try:
            self.reg_dt = DecisionTreeClassifier(criterion='entropy')
            self.reg_dt.fit(self.x_train, self.y_train)
            logger.info(f'==============Decision Tree================')
            logger.info(f'train accuracy : {accuracy_score(self.y_train, self.reg_dt.predict(self.x_train)) * 100}')
            logger.info(f'test accuracy : {accuracy_score(self.y_test, self.reg_dt.predict(self.x_test)) * 100}')

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")
    def Random_forest(self):
        try:
            self.reg_rf = RandomForestClassifier(n_estimators=7,criterion='entropy')
            self.reg_rf.fit(self.x_train, self.y_train)
            logger.info(f'==============Random_forest================')
            logger.info(f'train accuracy : {accuracy_score(self.y_train, self.reg_rf.predict(self.x_train)) * 100}')
            logger.info(f'test accuracy : {accuracy_score(self.y_test, self.reg_rf.predict(self.x_test)) * 100}')

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")
    def adaboost(self):
        try:
            self.reg_ad = LogisticRegression()
            self.reg_ad = AdaBoostClassifier(estimator=self.reg_lr,n_estimators=7,learning_rate=1.0)
            self.reg_ad.fit(self.x_train, self.y_train)
            logger.info(f'==============adaboost================')
            logger.info(f'train accuracy : {accuracy_score(self.y_train, self.reg_ad.predict(self.x_train)) * 100}')
            logger.info(f'test accuracy : {accuracy_score(self.y_test, self.reg_ad.predict(self.x_test)) * 100}')

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

    def gradient_boosting(self):
        try:
            self.reg_gb = GradientBoostingClassifier(n_estimators=7, criterion='friedman_mse')
            self.reg_gb.fit(self.x_train, self.y_train)
            logger.info(f'==============GradientBoosting================')
            logger.info(f'train accuracy : {accuracy_score(self.y_train, self.reg_gb.predict(self.x_train)) * 100}')
            logger.info(f'test accuracy : {accuracy_score(self.y_test, self.reg_gb.predict(self.x_test)) * 100}')

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")
    def XGB(self):
        try:
            self.reg_xgb = XGBClassifier()
            self.reg_xgb.fit(self.x_train, self.y_train)
            logger.info(f'==============XGBoost================')
            logger.info(f'train accuracy : {accuracy_score(self.y_train, self.reg_xgb.predict(self.x_train)) * 100}')
            logger.info(f'test accuracy : {accuracy_score(self.y_test, self.reg_xgb.predict(self.x_test)) * 100}')

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")
    def SVC(self):
        try:
            self.reg_svc = SVC()
            self.reg_svc.fit(self.x_train, self.y_train)
            logger.info(f'==============Support Vector Machine================')
            logger.info(f'train accuracy : {accuracy_score(self.y_train, self.reg_svc.predict(self.x_train)) * 100}')
            logger.info(f'test accuracy : {accuracy_score(self.y_test, self.reg_svc.predict(self.x_test)) * 100}')

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")









if __name__ == "__main__":
    try:
        data = 'heart.csv'
        obj = HEART(data)
        obj.knn_algo()
        obj.logistic_regression()
        obj.naive_bayes()
        obj.Decision_Tree()
        obj.Random_forest()
        obj.adaboost()
        obj.gradient_boosting()
        obj.XGB()
        obj.SVC()
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")