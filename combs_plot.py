from math import factorial
from matplotlib import pyplot

x = [i for i in range(100)]
y = [factorial(100)/(factorial(c)*factorial(100-c)) for c in x]

for l in list(" ".join([str(x[i]), str(y[i])]) for i in (range(100))):
    print(l)
