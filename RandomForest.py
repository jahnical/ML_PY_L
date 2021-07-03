
import math
from Combinations import combinations
from DecisionTree import DecisionTree


class RandomForest:
    def __init__(self):
        self.trees = []

    def fit(self, dataset):
        self.__build_trees(dataset)

    def __build_trees(self, dataset):
        self.attributes = len(dataset[0])-1
        self.groups = combinations(list([i for i in range(self.attributes)]), int(len(dataset[0])*0.75))
        for group in self.groups:
            trimed_dataset = self.__get_dataset(group, dataset)
            self.trees.append(DecisionTree())
            self.trees[-1].fit(trimed_dataset)

    def predict(self, data):
        predictions = []
        for row in data:
            predictions.append(self.__predict(row))
        return predictions

    def __predict(self, row):
        predicts = {}
        for i in range(len(self.groups)):
            trimed = self.__trim_row(row, self.groups[i])
            pred = self.trees[i].predict([trimed])[0][-1]
            if pred in predicts:
                predicts[pred] += 1
            else: predicts[pred] = 1
        prediction = ("Unknown", 0)
        print("============================================================")
        print(predicts)
        print("============================================================")
        for k, i in predicts.items():
            if i > prediction[1]:
                prediction = (k, i)
        return prediction[0]

    def __trim_row(self, row, columns):
        trimed = []
        for c in columns:
            trimed.append(row[c])
        return trimed


    def __get_dataset(self, columns, dataset):
        data = []
        for row in dataset:
            trimed = self.__trim_row(row, columns)
            trimed.append(row[-1])
            data.append(trimed)
        return data

    def __get_column(self, dataset, column):
        return list(map(lambda row: row[column], dataset))


#TEST
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

url = "..\\datasets\\iris_new.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = pandas.read_csv(url, names=names)

x = dataset.values[:,0:4]
y = dataset.values[:,4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state=1)

training_data = []
for i in range(len(x_train)):
    training_data.append(list(x_train[i]))
    training_data[i].append(y_train[i])

testing_data = []
for i in range(len(x_test)):
    testing_data.append(list(x_test[i]))
    testing_data[i].append(y_test[i])


# training_data = [
#     ['Green', 3, 'Mango'],
#     ['Yellow', 3, 'Mango'],
#     ['Red', 1, 'Grape'],
#     ['Red', 1, 'Grape'],
#     ['Yellow', 3, 'Lemon']
# ]

# testing_data = [
#     ['Red', 1, 'Grape'],
#     ['Red', 2, 'Grape'],
#     ['Green', 3, 'Mango'],
#     ['Yellow', 4, 'Mango'],
#     ['Yellow', 3, 'Lemon']
# ]

tree = RandomForest()
tree.fit(training_data)
#print(training_data)
tester = list(map(lambda row: row[0:4], testing_data))

result = tree.predict(tester)
predictions = result

print()
print(result)
print()
print(y_test)

print()
for i in range(len(y_test)):
    print(y_test[i] == predictions[i])
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))
