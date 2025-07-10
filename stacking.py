from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from preprocess_data import preprocess_data
import matplotlib.pyplot as plt
import numpy as np

X_train, X_test, y_train, y_test = preprocess_data('dataset.csv', 0.2, 1234)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf1 = SVC(kernel='rbf', decision_function_shape='ovr')  # 'ovr' is default
clf1.fit(X_train_scaled, y_train)
y_pred1 = clf1.predict(X_test_scaled)

d1 = {
    'accuracy': accuracy_score(y_test, y_pred1),
    'precision': precision_score(y_test, y_pred1, average='weighted'),
    'recall': recall_score(y_test, y_pred1, average='weighted'),
    'F1-score': f1_score(y_test, y_pred1, average='weighted'),
    
}

print("SVC:-")
print(d1)
print(confusion_matrix(y_test, y_pred1))
print("---------------------------------------------------")

clf2=DecisionTreeClassifier()
clf2.fit(X_train,y_train)
y_pred2=clf2.predict(X_test)

d2 = {
    'accuracy': accuracy_score(y_test, y_pred2),
    'precision': precision_score(y_test, y_pred2, average='weighted'),
    'recall': recall_score(y_test, y_pred2, average='weighted'),
    'F1-score': f1_score(y_test, y_pred2, average='weighted'),
}

print("Decision Tree:-")
print(d2)
print(confusion_matrix(y_test, y_pred2))
print("---------------------------------------------------")

base_learners=[('dt',DecisionTreeClassifier(max_depth=100)),('svm',SVC(kernel='rbf', decision_function_shape='ovr'))]

meta_learner =LogisticRegression(max_iter=2000,solver='saga')

clf3=StackingClassifier(estimators=base_learners,final_estimator=meta_learner,cv=5)

clf3.fit(X_train_scaled, y_train)
y_pred3 = clf3.predict(X_test_scaled)

d3 = {
    'accuracy': accuracy_score(y_test, y_pred3),
    'precision': precision_score(y_test, y_pred3, average='weighted'),
    'recall': recall_score(y_test, y_pred3, average='weighted'),
    'F1-score': f1_score(y_test, y_pred3, average='weighted'),
}

print("Stacking:- \nbase learners:- DecisionTree,SVC \nmeta learners:- logistic regression")
print(d3)
print(confusion_matrix(y_test, y_pred3))


#plotting the results

keys=list(d1.keys())
values1=[d1[k] for k in keys]
values2=[d2[k] for k in keys]
values3=[d3[k] for k in keys]

x=np.arange(len(keys))
width=0.2


plt.bar(x - width, values1, width, label='SVM', color='blue')
plt.bar(x , values2, width, label='Decision Tree', color='red')
plt.bar(x + width, values3, width, label='Stacking', color='yellow')

# Add labels, title, legend, ticks

plt.title('Comparison of Two Models')
plt.xticks(x, keys)
plt.legend()

plt.tight_layout()
plt.show()









