from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import preprocess_data
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

file_name='dataset.csv'

X_train,X_test,y_train,y_test=preprocess_data(file_name,0.2,1234)


clf1 = DecisionTreeClassifier(criterion='gini', max_depth=100, random_state=1234)
clf1.fit(X_train, y_train)

# Make predictions
predictions1 = clf1.predict(X_test)


d1={'accuracy': accuracy_score(y_test,predictions1),
    'precision': precision_score(y_test,predictions1,average='weighted'),
    'recall': recall_score(y_test,predictions1,average='weighted'),
    'F1-score': f1_score(y_test,predictions1,average='weighted')}


print(f"DECISION TREE:\n{d1}")
print(f"confusion matrix:\n{confusion_matrix(y_test,predictions1)}")
print("----------------------------------------------------")


clf2 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=4),n_estimators=400)
clf2.fit(X_train, y_train)
predictions2= clf2.predict(X_test)


d2={'accuracy': accuracy_score(y_test,predictions2),
    'precision': precision_score(y_test,predictions2,average='weighted'),
    'recall': recall_score(y_test,predictions2,average='weighted'),
    'F1-score': f1_score(y_test,predictions2,average='weighted')}

print(f"ADABOOST:\n{d2}")
print(f"confusion matrix:\n{confusion_matrix(y_test,predictions2)}")


keys=list(d1.keys())
values1=[d1[k] for k in keys]
values2=[d2[k] for k in keys]

x=np.arange(len(keys))
width=0.35


plt.bar(x - width/2, values1, width, label='Decision Tree', color='blue')
plt.bar(x + width/2, values2, width, label='AdaBoost', color='red')

# Add labels, title, legend, ticks

plt.title('Comparison of Two Models')
plt.xticks(x, keys)
plt.legend()

plt.tight_layout()
plt.show()