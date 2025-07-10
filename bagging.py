
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix 
from preprocess_data import preprocess_data
import matplotlib.pyplot as plt
import numpy as np

file_name='dataset.csv'

X_train,X_test,y_train,y_test=preprocess_data(file_name,test_size=0.2,random_state=1234)

clf1=DecisionTreeClassifier()
clf1.fit(X_train,y_train)
predictions1=clf1.predict(X_test)   

d1={'accuracy': accuracy_score(y_test,predictions1),
    'precision': precision_score(y_test,predictions1,average='weighted'),
    'recall': recall_score(y_test,predictions1,average='weighted'),
    'F1-score': f1_score(y_test,predictions1,average='weighted')}


print("Decision tree:-")
print(d1)
print(confusion_matrix(y_test,predictions1))
print("----------------------------------------------------------------------")


clf2 = RandomForestClassifier(n_estimators=10, max_depth=100, random_state=1234)

clf2.fit(X_train,y_train)
predictions2=clf2.predict(X_test)

d2={'accuracy': accuracy_score(y_test,predictions2),
    'precision': precision_score(y_test,predictions2,average='weighted'),
    'recall': recall_score(y_test,predictions2,average='weighted'),
    'F1-score': f1_score(y_test,predictions2,average='weighted')}


print("Random Forest")
print(d2)
print(confusion_matrix(y_test,predictions2))



#plotting the results

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




