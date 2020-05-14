import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# Function used to calculate the distance between the starting and ending location
def distance_on_unit_sphere(lat1, long1, lat2, long2):
    degrees_to_radians = math.pi / 181.0
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians

    theta1 = long1 * degrees_to_radians
    theta2 = long2 * degrees_to_radians

    a = ((math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2)) + (math.cos(phi1) * math.cos(phi2)))
    if a > 1:
        a = 0.999999
    dis = math.acos(a)
    return dis * 6374


# Function to define different modelling technique for predicting the number of of passholder for each type
def modeling_techniques_and_prediction(df, passholder_type, model_type):
    X = df.drop(passholder_type, axis=1)
    y = df[passholder_type]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    if model_type == 'logistic':
        lr = LogisticRegression()
        model_fit = lr.fit(X_train, y_train)
        pred1 = lr.predict(X_test)
        print(model_fit)
        print("\n The classification report for Logistic Regression is: \n", classification_report(y_test, pred1))
        print("The Confusion Matrix for Logistic Regression is: \n", confusion_matrix(y_test, pred1))

    elif model_type == 'random':
        clf = RandomForestClassifier()
        model_fit = clf.fit(X_train, y_train)
        pred1 = clf.predict(X_test)
        print(model_fit)
        print("\n The classification report for Random Forest is: \n", classification_report(y_test, pred1))
        print("The Confusion Matrix for Random Forest is: \n", confusion_matrix(y_test, pred1))

    else:
        clf2 = DecisionTreeClassifier()
        model_fit = clf2.fit(X_train, y_train)
        pred1 = clf2.predict(X_test)
        print(model_fit)
        print("\n The classification report for Decision Tree is: \n", classification_report(y_test, pred1))
        print("The Confusion Matrix for Decision Tree is: \n", confusion_matrix(y_test, pred1))