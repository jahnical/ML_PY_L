import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, m, c):
        self.m = m
        self.c = c

class Regressor:
    def __init__(self, independent, dependent):
        self.independent = independent
        self.dependent = dependent
        self.mean_x = np.mean(self.independent)
        self.mean_y = np.mean(self.dependent)
        self.model = self.__build_model()

    def __build_model(self):
       
        #Calculating gradient m
        t_numer = 0
        t_deno = 0
        for i in range(len(self.independent)):
            t_numer += (self.independent[i] - self.mean_x) * (self.dependent[i] - self.mean_y)
            t_deno += (self.independent[i] - self.mean_x) ** 2
        m = t_numer / t_deno

        #Calculating y intercept c
        c = self.mean_y - (m * self.mean_x)

        return Model(m, c)
    
    def predict(self, independent):
        predicted = []
        for i in independent:
            predicted.append(self.calculate_y(i))
        return predicted

    def calculate_y(self, x):
        return self.model.m * x + self.model.c

    def r2_scoring(self):
        t_numer = 0
        t_deno = 0
        for i in range(len(self.dependent)):
            t_numer += (self.dependent[i] - self.mean_y) ** 2
            t_deno += (self.calculate_y(self.independent[i]) - self.mean_y) ** 2
        return t_numer / t_deno

 
x = [i for i in range(100)]
y = [i*5+2 for i in range(0, 200, 2)]

x_test = [i for i in range(95, 120)]
y_test = [i*5+2 for i in range(95, 120)]

reg = Regressor(x, y)
print("m: {0}, c: {1}".format(reg.model.m, reg.model.c))

#Testing
results = reg.predict(x_test)

print(results)
print(y_test)

print("R squared accuracy: {}".format(reg.r2_scoring()))

plt.plot(x, y, color="#dd1212")
plt.show()