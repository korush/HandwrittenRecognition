import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import naive_bayes
from sklearn import neural_network
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score
import os
import sys
import pickle



class dataset:
    def __init__(self, filename):
        self.loadData(filename)
    
    def loadData(self, filename):
       self.dataframe = pd.read_csv(filename, header = None)
       self.features = list(self.dataframe.columns[:1024])
       if len(self.dataframe.columns) == 1025:
            self.target = self.dataframe.columns[1024]

    @property
    def featuresData(self):
        return self.dataframe[self.features]

    @property
    def targetData(self):
        return self.dataframe[self.target]

class AlgorithmTypes:
    DT = 1,
    NB = 2,
    NN = 3,
    SVM = 4,
    KNN = 5

class DataSetTypes:
    DS1 = 1,
    DS2 = 2,

class TestTypes:
    Val = 1,
    Test = 2

class Classifier:

    def __init__(self, algo, trainfile, valfile, testfile):
        self.valfile = valfile
        self.testfile = testfile
        self.AlgorithmType = algo
        self.ds = dataset(trainfile)
        if 'ds1' in trainfile:
            self.dataSetType = DataSetTypes.DS1
        elif 'ds2' in trainfile:
            self.dataSetType = DataSetTypes.DS2
        else:
            self.dataSetType = DataSetTypes.DS1
    def trainModel(self):
        
        if os.path.exists(self.modelFileName()):
           self.clf = pickle.load(open(self.modelFileName(), 'rb'))
           return

        if self.AlgorithmType == AlgorithmTypes.DT:
            
            self.clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best')
            #self.clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='random')
            #self.clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best')
            #self.clf = tree.DecisionTreeClassifier(criterion='gini', splitter='random')
        elif self.AlgorithmType == AlgorithmTypes.NB:
            self.clf = naive_bayes.GaussianNB(var_smoothing=0.5)
            #self.clf = naive_bayes.MultinomialNB(alpha=0.5)
            #self.clf = naive_bayes.ComplementNB(alpha=0.5)
            #self.clf = naive_bayes.BernoulliNB(alpha=1.0)
        elif self.AlgorithmType == AlgorithmTypes.NN:

            self.clf = neural_network.MLPClassifier(hidden_layer_sizes=50, activation='logistic', learning_rate='constant', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=50, activation='relu', learning_rate='constant', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=50, activation='tanh', learning_rate='constant', learning_rate_init=0.001, shuffle=True)

            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=50, activation='logistic', learning_rate='adaptive', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=50, activation='relu', learning_rate='adaptive', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=50, activation='tanh', learning_rate='adaptive', learning_rate_init=0.001, shuffle=True)


            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=100, activation='logistic', learning_rate='constant', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=100, activation='relu', learning_rate='constant', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=100, activation='tanh', learning_rate='constant', learning_rate_init=0.001, shuffle=True)

            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=100, activation='logistic', learning_rate='adaptive', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=100, activation='relu', learning_rate='adaptive', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=100, activation='tanh', learning_rate='adaptive', learning_rate_init=0.001, shuffle=True)

            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=200, activation='logistic', learning_rate='constant', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=200, activation='relu', learning_rate='constant', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=200, activation='tanh', learning_rate='constant', learning_rate_init=0.001, shuffle=True)

            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=200, activation='logistic', learning_rate='adaptive', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=200, activation='relu', learning_rate='adaptive', learning_rate_init=0.001, shuffle=True)
            #self.clf = neural_network.MLPClassifier(hidden_layer_sizes=200, activation='tanh', learning_rate='adaptive', learning_rate_init=0.001, shuffle=True)
        

        elif self.AlgorithmType == AlgorithmTypes.SVM:
             
            self.clf =  SVC(kernel='linear', C=1.0)  
            #self.clf =  SVC(kernel='linear', C=2.0)  
            #self.clf =  SVC(kernel='poly', C=1.0)  
            #self.clf =  SVC(kernel='poly', C=2.0)  
            #self.clf =  SVC(kernel='rbf', C=1.0)  
            #self.clf =  SVC(kernel='rbf', C=2.0)  
            #self.clf =  SVC(kernel='sigmoid', C=1.0)  
            #self.clf =  SVC(kernel='sigmoid', C=2.0)  
            
        elif self.AlgorithmType == AlgorithmTypes.KNN:
            #self.clf = neighbors.KNeighborsClassifier(n_neighbors=5, metric='euclidean')
            #self.clf = neighbors.KNeighborsClassifier(n_neighbors=10, metric='euclidean')
            #self.clf = neighbors.KNeighborsClassifier(n_neighbors=5, metric='manhattan')
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=10, metric='manhattan')

        self.clf.fit(self.ds.featuresData, self.ds.targetData)
        
        scores = cross_val_score(self.clf, self.ds.featuresData, self.ds.targetData, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        with open(self.modelFileName(), 'wb') as file:
            pickle.dump(self.clf, file)


    def modelFileName(self):
        if self.dataSetType == DataSetTypes.DS1:
            str = 'ds1-'
        else:
            str = 'ds2-'

        if self.AlgorithmType == AlgorithmTypes.DT:
            str += "dt"
        elif self.AlgorithmType == AlgorithmTypes.NB:
            str += "nb"
        elif self.AlgorithmType == AlgorithmTypes.NN:
            str += "nn"
        elif self.AlgorithmType == AlgorithmTypes.SVM:
            str += "svm"
        elif self.AlgorithmType == AlgorithmTypes.KNN:
            str += "knn"

        str += '.sav'

        return str


    def outputFileName(self, type):
        if self.dataSetType == DataSetTypes.DS1:
            str = "ds1Test-"
        else:
            str = "ds2Test-"    
        if type == TestTypes.Val:
            str += 'Val-'
        else: 
            str += 'Test'

        if self.AlgorithmType == AlgorithmTypes.DT:
            str += "dt"
        elif self.AlgorithmType == AlgorithmTypes.NB:
            str += "nb"
        elif self.AlgorithmType == AlgorithmTypes.NN:
            str += "3"
        elif self.AlgorithmType == AlgorithmTypes.SVM:
            str += "4"
        elif self.AlgorithmType == AlgorithmTypes.KNN:
            str += "5"

        str += '.csv'

        return str


    

    def generateOutput(self, type):
        if type == TestTypes.Val:
            dstest = dataset(self.valfile)
        else:
            dstest = dataset(self.testfile)

        pred = self.clf.predict(dstest.featuresData)
        
        if type == TestTypes.Val:
            print(accuracy_score(dstest.targetData, pred))

        with open(self.outputFileName(type), 'w') as f:
            for i in range(len(pred)):
                f.write("{0},{1}\n".format(i + 1, pred[i]))
        

        

    def run(self):
        #start_time = time.clock()
        self.trainModel()
        #print(time.clock() - start_time, "seconds")
        self.generateOutput(TestTypes.Val)
        self.generateOutput(TestTypes.Test)

    
def runall(trainfile, valfile, testfile):
    clf = Classifier(AlgorithmTypes.DT, trainfile, valfile, testfile)
    clf.run()

    clf = Classifier(AlgorithmTypes.NB, trainfile, valfile, testfile)
    clf.run()

    clf = Classifier(AlgorithmTypes.NN, trainfile, valfile, testfile)
    clf.run()

    clf = Classifier(AlgorithmTypes.SVM, trainfile, valfile, testfile)
    clf.run()

    
    clf = Classifier(AlgorithmTypes.KNN, trainfile, valfile, testfile)
    clf.run()
if __name__ == "__main__":

    
    #runall('ds1\\ds1Train.csv', 'ds1\\ds1Val.csv', 'ds1\\ds1Test.csv')
    runall('ds2\\ds2Train.csv', 'ds2\\ds2Val.csv', 'ds2\\demo_ds_sklearn.csv')



