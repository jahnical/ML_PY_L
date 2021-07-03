
import math


class Question:
        def __init__(self, value, column):
            self.value = value
            self.column = column

        def match(self, other):
            if (isinstance(other[self.column], (float, int))):
                return self.value < other[self.column]
            else:
                return self.value == other[self.column]

        def matches(self, dataset):
            matches = 0;
            for row in dataset:
                if self.match(row):
                    matches += 1
            return matches

        def print_q(self):
            condition = "=="
            if isinstance(self.value, (float, int)):
                condition = ">="
            print("value {0} {1}".format(condition, self.value))


class Node:
        def __init__(self, question, true_branch, false_branch):
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch

class Leaf:
        def  __init__(self, data):
            self.data = data

class DecisionTree:
    
    
    def fit(self, dataset):
        """Train the model @args 2 dimession array with classes on -1 column"""
        self.tree = self.__build_tree(dataset)

    def __build_tree(self, dataset):
        question, gain = self.__best_splitter(dataset)
        if gain == 0:
             return Leaf(dataset)

        trues, falses = self.__partition(question, dataset)

        true_branch = self.__build_tree(trues)
        false_branch = self.__build_tree(falses)

        return Node(question, true_branch, false_branch)

    def __best_splitter(self, dataset):
        s_entropy = self.__entropy(dataset)
        gain = self.__inf_gain(0, dataset, s_entropy)
        for i in range(len(dataset[0])-1):
            g = self.__inf_gain(i, dataset, s_entropy)
            if gain[1] < g[1]:
                gain = g
        return gain
            

    def __inf_gain(self, column, dataset, s_entropy):
        question = self.__best_question(column, dataset)
        trues, falses = self.__partition(question, dataset)
        inf_lost = len(trues)/len(dataset) * self.__entropy(trues) + len(falses)/len(dataset) * self.__entropy(falses)
        return question, s_entropy - inf_lost
        
    def __partition(self, question, dataset):
        trues, falses = [], []
        for row in dataset:
            if question.match(row):
                trues.append(row)
            else:
                falses.append(row)
        return trues, falses

    def __best_question(self, column, dataset):
        values = list(set(self.__get_column(dataset, column)))
        question = Question(values[0], column)
        question.print_q()
        for i in range(1, len(values)):
            q = Question(values[i], column)
            q.print_q()
            if abs(question.matches(dataset) - len(dataset)) > abs(q.matches(dataset) - len(dataset)):
                print("Better question found {0} > {1}, {2}".format(question.matches(dataset), q.matches(dataset), q.print_q()))
                question = q
        print("Best question for column {0} is ".format(column))
        question.print_q()
        return question

    def __entropy(self, dataset):
        classes_count = self.__occurrence_counts(self.__get_column(dataset, -1))
        ent = 0
        for k, v in classes_count.items():
            pk = v/len(dataset)
            ent -= (pk * math.log2(pk))
        return ent

    def __get_column(self, dataset, column):
        return list(map(lambda row: row[column], dataset))

    def __occurrence_counts(self, data):
        counts = {}
        for v in data:
            if v not in counts:
                counts[v] = 1
            else:
                counts[v] += 1
        return counts

    
    def predict(self, data):
        """Label unlabeled data @args unlabeled 2d array"""
        for i in range(len(data)):
            data[i].append(self.__classify(data[i]))

        return data

    def __classify(self, data):
        leaf = self.__find_leaf(data, self.tree)
        #print(leaf.data)
        return leaf.data[0][-1]

    def __find_leaf(self, data, node):
        if isinstance(node, Leaf):
            return node
        else:
            #print(data)
            node.question.print_q()
            if node.question.match(data):
                return self.__find_leaf(data, node.true_branch)
            else: return self.__find_leaf(data, node.false_branch)



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

tree = DecisionTree()
tree.fit(training_data)
#print(training_data)
tester = list(map(lambda row: row[0:4], testing_data))

result = tree.predict(tester)
predictions = list(map(lambda row: row[-1], result))

print()
print(result)
print()
print(y_test)

print()
for i in range(len(y_test)):
    print(y_test[i] == predictions[i])
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))
