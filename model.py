from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import numpy as np


class Model:
    #the model
    model= DummyClassifier(strategy='stratified', random_state=None, constant=None)
    # classification=0;regression=1
    task=0

    def __init__(self, task, modelName):
        self.task = task
        #create the model
        if task ==0:
            if modelName == "NB":
                self.model = GaussianNB()
            elif modelName == "KNN":
                self.model = KNeighborsClassifier(n_neighbors=3)
            elif modelName == "SVM":
                #C=pow(10, -3), C=pow(10, -2), C=0.1, C=10, C=pow(10, 2) C=pow(10, 3) 
                self.model = SVC(C=pow(10, 3))
            elif modelName == "C45":
                self.model = tree.DecisionTreeClassifier()
            else:
                print("YOU CHOSE WRONG MODEL FOR CLASSIFICATION!")
        else:
            if modelName == "LR":
                self.model = linear_model.LinearRegression()
            elif modelName == "M5":
                self.model = tree.DecisionTreeRegressor()
            elif modelName == "KNN":
                self.model = KNeighborsRegressor(n_neighbors=3)
            else:
                print("YOU CHOSE WRONG MODEL FOR REGRESSION!")
                self.model = linear_model.LinearRegression()

    def train(self,data):
        print("training...")
        scoring = 'accuracy'
        if self.task==1:
            scoring="neg_mean_squared_error"
        scores = cross_val_score(self.model, data.iloc[:, 2:], data["label"], cv=10, scoring=scoring)
        
        if self.task==1:
            print(scoring, np.mean(np.sqrt(np.abs(scores))))
            print(scoring, np.sqrt(np.abs(np.mean(scores))))
        else:
            print(scoring, np.mean(scores))

