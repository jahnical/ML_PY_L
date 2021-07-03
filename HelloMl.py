
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import datasets

url = "..\\datasets\\iris_new.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = pandas.read_csv(url, names=names)

# print(dataset.shape)
# print(dataset.describe())
# print(dataset.groupby("class").size())
# dataset.plot(kind="box", subplots = True, layout=(2,2), sharex=False, sharey=False)
# plt.show()
# dataset.hist()
# plt.show()
# scatter_matrix(dataset)
# plt.show()

array = dataset.values
X = array[:,0:4]
Y = array[:,4]

validation_size = 0.2
seed = 6
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

scoring = "accuracy"

# Spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#Evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     model.max_iter = 50
#     kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
#     cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)

#Test models
for name, model in models:
    model.max_iter = 100
    model.fit(x_train, y_train)
    test_result = model.predict(x_test)
    print(name + " Test")
    accurate = 0
    for i in range(len(y_test)):
        if test_result[i] == y_test[i]: accurate += 1
    print("Accuracy: {0}{1}".format(accurate/len(y_test)*100, "%"))
        
