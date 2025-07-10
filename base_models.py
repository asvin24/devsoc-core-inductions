'''
this script shows a comparision between some of the 
traditional machine learning models which include 
decision trees, logistic regression, 
support vector machine, KNN and  naive bayes.
'''

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix 
from preprocess_data import preprocess_data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np



X_train, X_test, y_train, y_test = preprocess_data('dataset.csv', 0.2, 1234)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf1=LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
clf1.fit(X_train_scaled,y_train)
y_pred1=clf1.predict(X_test_scaled)

d1 = {
    'accuracy': accuracy_score(y_test, y_pred1),
    'precision': precision_score(y_test, y_pred1, average='weighted'),
    'recall': recall_score(y_test, y_pred1, average='weighted'),
    'F1-score': f1_score(y_test, y_pred1, average='weighted'),
}

print("Logistic Regressor:")
print(d1)
print(confusion_matrix(y_test, y_pred1))
print("---------------------------------------------------")

clf2=DecisionTreeClassifier()
clf2.fit(X_train_scaled,y_train)
y_pred2=clf2.predict(X_test_scaled)

d2 = {
    'accuracy': accuracy_score(y_test, y_pred2),
    'precision': precision_score(y_test, y_pred2, average='weighted'),
    'recall': recall_score(y_test, y_pred2, average='weighted'),
    'F1-score': f1_score(y_test, y_pred2, average='weighted'),
}

print("Decision Tree:")
print(d2)
print(confusion_matrix(y_test, y_pred2))
print("---------------------------------------------------")

clf3=SVC(kernel='rbf', decision_function_shape='ovr')
clf3.fit(X_train_scaled,y_train)
y_pred3=clf3.predict(X_test_scaled)

d3 = {
    'accuracy': accuracy_score(y_test, y_pred3),
    'precision': precision_score(y_test, y_pred3, average='weighted'),
    'recall': recall_score(y_test, y_pred3, average='weighted'),
    'F1-score': f1_score(y_test, y_pred3, average='weighted'),
}

print("Decision Tree:")
print(d2)
print(confusion_matrix(y_test, y_pred3))
print("---------------------------------------------------")

clf4=KNeighborsClassifier(n_neighbors=141)     #following the square root heuristic
clf4.fit(X_train_scaled,y_train)
y_pred4=clf4.predict(X_test_scaled)

d4 = {
    'accuracy': accuracy_score(y_test, y_pred4),
    'precision': precision_score(y_test, y_pred4, average='weighted'),
    'recall': recall_score(y_test, y_pred4, average='weighted'),
    'F1-score': f1_score(y_test, y_pred4, average='weighted'),
}

print("KNN:")
print(d4)
print(confusion_matrix(y_test, y_pred4))
print("---------------------------------------------------")

clf5 = GaussianNB()
clf5.fit(X_train_scaled, y_train)
y_pred5 = clf5.predict(X_test_scaled)

d5 = {
    'accuracy': accuracy_score(y_test, y_pred5),
    'precision': precision_score(y_test, y_pred5, average='weighted'),
    'recall': recall_score(y_test, y_pred5, average='weighted'),
    'F1-score': f1_score(y_test, y_pred5, average='weighted'),
}

print("Naive Bayes:")
print(d5)
print(confusion_matrix(y_test, y_pred5))
print("---------------------------------------------------")



#plotting the results

keys=list(d1.keys())
values1=[d1[k] for k in keys]
values2=[d2[k] for k in keys]
values3=[d3[k] for k in keys]
values4=[d4[k] for k in keys]
values5=[d5[k] for k in keys]

x=np.arange(len(keys))
width=0.2


plt.bar(x - 2*width, values1, width, label='Logistic Regression', color='blue')
plt.bar(x- width , values2, width, label='Decision Tree', color='red')
plt.bar(x, values3, width, label='Support Machine Vector', color='yellow')
plt.bar(x + width, values4, width, label='KNN', color='green')
plt.bar(x + 2*width, values5, width, label='Naive Bayes', color='orange')

# Add labels, title, legend, ticks

plt.title('Comparison of Models')
plt.xticks(x, keys)
plt.legend()

plt.tight_layout()
plt.show()







