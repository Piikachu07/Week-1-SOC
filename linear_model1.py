import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('LifeExpectancyDataset - Sheet1.csv')

data.dropna(inplace=True)

x = data['Alcohol'].values
y = data['Life expectancy'].values

n = len(x)

sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x_squared = np.sum(x**2)
sum_xy = np.sum(x*y)

m= (n*sum_xy - (sum_x*sum_y))/(n*sum_x_squared-sum_x**2)
c= (sum_y - m*sum_x)/n

y_predic = m*x + c

plt.scatter(x,y, color='blue', label ='Life accpectancy Analysis')
plt.plot(x,y_predic, color='red', label ='linear regression')
plt.xlabel('Alcohol consumption')
plt.ylabel('Life acceptancy')
plt.legend()
plt.show()




# X = np.hstack((np.ones((X.shape[0],1)), X))

# def normal_equation(x,y):
#     thetha = np.linalg.inv(X.T@X) @X.T@y
#     return thetha

# thetha = normal_equation(X,y)

# predictions = X@thetha

# print(f"Thetha (parameters): {thetha}")
# print(f"Predictions: {predictions[:5]}")



