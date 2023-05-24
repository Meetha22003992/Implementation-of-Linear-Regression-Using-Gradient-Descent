# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import numpy as np
2. import matplotlib.pyplot as plt
3. Import pandas as pd
4. Predict the values usind predict() function
5.Display the predicted values

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Meetha Prabhu
RegisterNumber:  212222240065

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1 (1).txt",header =None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
print("Profit Production")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h - y)**2
  
  return 1/(2*m)*np.sum(square_err)
  
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
print("Cost Function:")

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):

  m=len(y)
  J_history=[]
  
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta, J_history
  
  theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
print("Cost fucntion using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color='r')
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
print("Profit Production")

def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  predict1=predict(np.array([1,3.5]),theta)*1000
  print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))
  predict2=predict(np.array([1,7]),theta)*1000
  print("For population = 70,000 we predict a profit of $"+str(round(predict2,0)))

*/
```

## Output:
Profit Prediction:

![image](https://user-images.githubusercontent.com/119401038/229294473-a9c24475-d57e-4d36-81ad-07316f5625db.png)

Compute Cost:

![image](https://user-images.githubusercontent.com/119401038/229294500-318f6017-f818-4f13-b718-ed6daa507973.png)

h(x) Value:

![image](https://user-images.githubusercontent.com/119401038/229294521-10e32811-e559-42c6-bdae-88bfd5c1653b.png)

Cost Function using Gradient Descent Graph:

![image](https://user-images.githubusercontent.com/119401038/229294545-063d1f6b-b021-4898-9c32-8a06f2ec9820.png)

Predict Predcition:

![image](https://user-images.githubusercontent.com/119401038/229294559-92510321-9185-46f1-a7ea-5038cbcb5432.png)

Profit for the Population:

![image](https://user-images.githubusercontent.com/119401038/229294591-9703694e-7692-4add-8289-b172ccade2e9.png)

Profit for the Population:

![image](https://user-images.githubusercontent.com/119401038/229294611-fbe37940-58f1-4f85-a5f9-0a793b8998f1.png)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
