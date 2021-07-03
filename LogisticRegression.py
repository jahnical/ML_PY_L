import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

url = "..\\datasets\\iris_new.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = pandas.read_csv(url, names=names)

x = dataset.values[:,0:4]
y = dataset.values[:,4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)

print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))

plt.show()