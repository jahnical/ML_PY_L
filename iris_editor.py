classes = ['setosa','versicolor','virginica']
with open("..\\datasets\\iris.csv") as file:
    with open("..\\datasets\\iris_new.csv", "w") as new_file:
        for line in file.readlines():
            class_code = int(line[-2])
            new_line = line[:-2] + classes[class_code] + "\n"
            print(new_line)
            new_file.write(new_line)