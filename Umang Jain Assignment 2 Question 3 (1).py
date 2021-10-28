#!/usr/bin/env python
# coding: utf-8

# # Question 3a:

# In[6]:


import tensorflow as tf
from tensorflow import  keras
fashion_mnist = keras.datasets.fashion_mnist
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
import matplotlib.pyplot as plt
import numpy as np
(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()
names=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Normalize the data
X_train_full = X_train_full.astype('float32')
X_test_full = X_test_full.astype('float32')
X_train_full /= 255.0
X_test_full /= 255.0
X_train_full = X_train_full.reshape(X_train_full.shape[0], X_train_full.shape[1] * X_train_full.shape[2])
X_test_full = X_test_full.reshape(X_test_full.shape[0], X_test_full.shape[1] * X_test_full.shape[2])


# In[7]:


label_dictionnary = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 
                     3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 
                     7:'Sneaker', 8:'Bag', 9:'Ankle boot' }
def true_label(x):
    return label_dictionnary[x]


# In[8]:


x_train = X_train_full[:2000 , :]
y_train = y_train_full[:2000]
x_test = X_test_full[:500 , : ]
y_test = y_test_full[:500]


# # Question 3b

# ### Using Gaussian Baes Method and calculation of accuracy, confusion matrix and classification report

# In[9]:


NB = GaussianNB()
NB.fit(x_train, y_train)
y_pred = NB.predict(x_test)
acc_val_nb=accuracy_score(y_test,y_pred)
print(f'Accuracy_Score : {acc_val_nb}')
predProb  = NB.predict_proba(x_test)
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test,y_pred))


# #### We get the accuracy score of 0.622

# #### Plotting 10 examples with labels for pridicted values and actual values

# In[10]:



plt.figure(figsize=(100,100))
for index, (image, label,actual) in enumerate(zip(np.array(x_test)[1:10], y_pred[1:10], y_test[1:10])):
    plt.subplot(1, 10, index + 1)
    plt.imshow(np.reshape(np.array(image), (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted: ' + str(true_label(label)) , fontsize = 45)
    plt.xlabel('Actual Values: ' + str(true_label(actual)) , fontsize = 45)


# ### Build the bar-chart of predicted probability distribution of each of the 10 examples and Label the x-axis by the actual names (e.g., Sandal, Shirt, etc.) and not numbers and also Made sure the labels  readable by writing them with a 90 degree rotation.

# In[11]:


l = np.array(y_test)
g = np.array(y_pred)
for i in range(len(y_test)):
    if input("Do you wish to see performance on the next test digit? y/n")=='n':
        break
    plt.imshow(np.array(x_test[i]).reshape((28,28)),cmap=plt.cm.binary)
    plt.show()
    print("Actual value: ",l[i], "  predicted value: ", g[i])
    print("Predicted probability distribution:")
    #print("Probabilities:\n", predProb[i])
    [print("p(%d): %5.2f "% (j, predProb[i,j])) for j in range(predProb.shape[1])]
    plt.bar(names,height=predProb[i])
    plt.xticks(range(0,10), label = names, rotation = 90)
    plt.show()


# ### Using Multinomial Naive Bayes Method and calculation Accuarcy , Confusion matrix & Classification report

# In[12]:


from sklearn.naive_bayes import MultinomialNB
MB = MultinomialNB()
MB.fit(x_train, y_train)
y_pred = MB.predict(x_test)
acc_val_nb=accuracy_score(y_test,y_pred)
print(f'Accuracy_Score : {acc_val_nb}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
predProb  = MB.predict_proba(x_test)


# #### Here we get the accuracy of 0.682

# ### Plotting 10 examples with labels for pridicted values and actual values

# In[13]:


import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(100,100))
for index, (image, label,actual) in enumerate(zip(np.array(x_test)[1:10], y_pred[1:10], y_test[1:10])):
    plt.subplot(1, 10, index + 1)
    plt.imshow(np.reshape(np.array(image), (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted: ' + str(true_label(label)) , fontsize = 45)
    plt.xlabel('Actual Values: ' + str(true_label(actual)) , fontsize = 45)


# ### Build the bar-chart of predicted probability distribution of each of the 10 examples and Label the x-axis by the actual names (e.g., Sandal, Shirt, etc.) and not numbers and also Made sure the labels readable by writing them with a 90 degree rotation.

# In[14]:


l = np.array(y_test)
g = np.array(y_pred)
for i in range(len(y_test)):
    if input("Do you wish to see performance on the next test digit? y/n")=='n':
        break
    plt.imshow(np.array(x_test[i]).reshape((28,28)),cmap=plt.cm.binary)
    plt.show()
    print("Actual value: ",l[i], "  predicted value: ", g[i])
    print("Predicted probability distribution:")
    #print("Probabilities:\n", predProb[i])
    [print("p(%d): %5.2f "% (j, predProb[i,j])) for j in range(predProb.shape[1])]
    plt.bar(names,height=predProb[i])
    plt.xticks(range(0,10), label = names, rotation = 90)
    plt.show()


# ### Using Categorical Naive Bayes Method and calculation Accuarcy , Confusion matrix & Classification report
# 

# In[15]:


from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
cnb = CategoricalNB(min_categories= 10)
cnb.fit(x_train, y_train)
y_pred = cnb.predict(x_test)
acc_val_nb=accuracy_score(y_test,y_pred)
print(f'Accuracy_Score : {acc_val_nb}')
predProb  = cnb.predict_proba(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))


# ### Plotting 10 examples with labels for predicted and actual values

# In[16]:


import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(100,100))
for index, (image, label,actual) in enumerate(zip(np.array(x_test)[1:10], y_pred[1:10], y_test[1:10])):
    plt.subplot(1, 10, index + 1)
    plt.imshow(np.reshape(np.array(image), (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted: ' + str(true_label(label)) , fontsize = 45)
    plt.xlabel('Actual Values: ' + str(true_label(actual)) , fontsize = 45)


# ### Build the bar-chart of predicted probability distribution of each of the 10 examples and Label the x-axis by the actual names (e.g., Sandal, Shirt, etc.) and not numbers and also Made sure the labels readable by writing them with a 90 degree rotation.

# In[17]:


l = np.array(y_test)
g = np.array(y_pred)
for i in range(len(y_test)):
    if input("Do you wish to see performance on the next test digit? y/n")=='n':
        break
    plt.imshow(np.array(x_test[i]).reshape((28,28)),cmap=plt.cm.binary)
    plt.show()
    print("Actual value: ",l[i], "  predicted value: ", g[i])
    print("Predicted probability distribution:")
    #print("Probabilities:\n", predProb[i])
    [print("p(%d): %5.2f "% (j, predProb[i,j])) for j in range(predProb.shape[1])]
    plt.bar(names,height=predProb[i])
    plt.xticks(range(0,10), label = names, rotation = 90)
    plt.show()


# ### Result: Multinomial Naive Base give highest accuracy  with 68% 

# # Question 3c
# 
# ### Using Cross Validation
# ### Running for different K values from 1 to 201 to know which yields the max accuracy.
# ### Plotting the graph for K and accuarcy score

# In[19]:


import numpy as np
from sklearn.model_selection import cross_val_score
neighbors = np.arange(1, 201, 10)
scores = []
for k in neighbors:   # running for different K values to know which yields the max accuracy. 
    clf = KNeighborsClassifier(n_neighbors = k,  weights = 'distance', p=1)
    clf.fit(x_train, y_train)
    score = cross_val_score(clf, x_train, y_train, cv = 5)
    scores.append(score.mean())
plt.plot(neighbors,scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('accuracy_score')
plt.show()


# ### Accuracy Score

# In[20]:


from sklearn.metrics import accuracy_score
svc_accuracy = accuracy_score(y_test, y_pred)


# In[21]:


svc_accuracy


# ### Using Grid Search CV with K-NN

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
params={'n_neighbors': [1,11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191,201], 
       'weights': ['distance'], 
        'p':[1,2] 
       }


reg_knn = KNeighborsClassifier()
gs = GridSearchCV(estimator=reg_knn, param_grid=params,cv=5) #validate model with his parameters
gs.fit(x_train, y_train)
reg_knn = gs.best_estimator_

#printing best estimator values

print(reg_knn) 
pred_knn =reg_knn.predict(x_test)


# ### Accuracy

# In[23]:


svc_accuracy = accuracy_score(y_test, pred_knn)


# In[24]:


svc_accuracy


# ### Confusion Matrix and Classification Report for K-NN

# In[25]:


print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test,pred_knn))


# ### Plotting 10 examples with labels for predicted and actual values

# In[26]:


import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(100,100))
for index, (image, label,actual) in enumerate(zip(np.array(x_test)[1:10], pred_knn[1:10], y_test[1:10])):
    plt.subplot(1, 10, index + 1)
    plt.imshow(np.reshape(np.array(image), (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted: ' + str(true_label(label)) , fontsize = 45)
    plt.xlabel('Actual Values: ' + str(true_label(actual)) , fontsize = 45)


# ### Build the bar-chart of predicted probability distribution of each of the 10 examples and Label the x-axis by the actual names (e.g., Sandal, Shirt, etc.) and not numbers and also Made sure the labels readable by writing them with a 90 degree rotation.

# In[27]:


l = np.array(y_test)
g = np.array(pred_knn)


# In[28]:


for i in range(len(y_test)):
    if input("Do you wish to see performance on the next test digit? y/n") == 'n':
        break
    plt.imshow(np.array(x_test[i]).reshape((28,28)),cmap=plt.cm.binary)
    plt.show()
    print("Actual value: ",l[i], "  predicted value: ", g[i])
    print("Predicted probability distribution:")
    #print("Probabilities:\n", predProb[i])
    [print("p(%d): %5.2f "% (j, predProb[i,j])) for j in range(predProb.shape[1])]
    plt.bar(names,height=predProb[i])
    plt.xticks(range(0,10), label = names, rotation = 90)
    plt.show()


# ### Result Using K-NN we get an accuarcy of 80% with optimal k value = 11

# # Question 3d

#  ### Using CART with max_depth = 5 with decision tree , accuarcy and confusion matrix

# In[29]:


from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text


# In[31]:


import seaborn as sns
# Experiment with various values and also play with max_depth:
#tree=DecisionTreeClassifier(min_samples_leaf=50)
tree=DecisionTreeClassifier(max_depth=5)
treeModel=tree.fit(x_train, y_train)
y_pred=treeModel.predict(x_test)
print("Number of mislabeled points out of a total %d points : %d" % 
(x_test.shape[0], (y_test != y_pred).sum()))
C=confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))
print("The confusion matrix:\n", C)
print(f"Error rate: {(1.-sum(np.diag(C))/C.sum()):5.3}")
sns.heatmap(C, cmap="Spectral")
plt.show()
print("Visualize the feature importances over all trees")
sns.heatmap(treeModel.feature_importances_.reshape((28,28)), cmap=plt.cm.Spectral)
plt.show("Feature importances in the tree")
plt.show()
input("Hit Enter to see the tree and some of its characteristics")
fig=plt.figure(dpi=600)
sub=fig.add_subplot(1,1,1)
plot_tree(treeModel)
plt.show()
print(export_text(treeModel))
print("Maximum Depth of the the tree is ", treeModel.tree_.max_depth)
print("Number of nodes of the tree is ",treeModel.tree_.node_count)
print("Number of leaves (AKA number of regions) is ", treeModel.tree_.n_leaves)
print("Error rate: %5.3f"%   (1.-sum(np.diag(C))/C.sum()))


# ### Build the bar-chart of predicted probability distribution of each of the 10 examples and Label the x-axis by the actual names (e.g., Sandal, Shirt, etc.) and not numbers and also Made sure the labels readable by writing them with a 90 degree rotation.

# In[32]:


l = np.array(y_test)
g = np.array(y_pred)


# In[33]:


for i in range(len(y_test)):
    if input("Do you wish to see performance on the next test digit? y/n") == 'n':
        break
    plt.imshow(np.array(x_test[i]).reshape((28,28)),cmap=plt.cm.binary)
    plt.show()
    print("Actual value: ",l[i], "  predicted value: ", g[i])
    print("Predicted probability distribution:")
    #print("Probabilities:\n", predProb[i])
    [print("p(%d): %5.2f "% (j, predProb[i,j])) for j in range(predProb.shape[1])]
    plt.bar(names,height=predProb[i])
    plt.xticks(range(0,10), label = names, rotation = 90)
    plt.show()


# ### Using CART with max_depth = 10 with decision tree , accuarcy and confusion matrix

# In[34]:


tree=DecisionTreeClassifier(max_depth= 10 )
treeModel=tree.fit(x_train, y_train)
y_pred=treeModel.predict(x_test)
print("Number of mislabeled points out of a total %d points : %d" % 
(x_test.shape[0], (y_test != y_pred).sum()))
C=confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))
print("The confusion matrix:\n", C)
print(f"Error rate: {(1.-sum(np.diag(C))/C.sum()):5.3}")
sns.heatmap(C, cmap="Spectral")
plt.show()
print("Visualize the feature importances over all trees")
sns.heatmap(treeModel.feature_importances_.reshape((28,28)), cmap=plt.cm.Spectral)
plt.show("Feature importances in the tree")
plt.show()
input("Hit Enter to see the tree and some of its characteristics")
fig=plt.figure(dpi=600)
sub=fig.add_subplot(1,1,1)
plot_tree(treeModel)
plt.show()
print(export_text(treeModel))
print("Maximum Depth of the the tree is ", treeModel.tree_.max_depth)
print("Number of nodes of the tree is ",treeModel.tree_.node_count)
print("Number of leaves (AKA number of regions) is ", treeModel.tree_.n_leaves)
print("Error rate: %5.3f"%   (1.-sum(np.diag(C))/C.sum()))


# ### Build the bar-chart of predicted probability distribution of each of the 10 examples and Label the x-axis by the actual names (e.g., Sandal, Shirt, etc.) and not numbers and also Made sure the labels readable by writing them with a 90 degree rotation.

# In[36]:


l = np.array(y_test)
g = np.array(y_pred)


# In[37]:


for i in range(len(y_test)):
    if input("Do you wish to see performance on the next test digit? y/n") == 'n':
        break
    plt.imshow(np.array(x_test[i]).reshape((28,28)),cmap=plt.cm.binary)
    plt.show()
    print("Actual value: ",l[i], "  predicted value: ", g[i])
    print("Predicted probability distribution:")
    #print("Probabilities:\n", predProb[i])
    [print("p(%d): %5.2f "% (j, predProb[i,j])) for j in range(predProb.shape[1])]
    plt.bar(names,height=predProb[i])
    plt.xticks(range(0,10), label = names, rotation = 90)
    plt.show()


# ### Using CART with min_samples_leaf = 1000 with decision tree , accuarcy and confusion matrix

# In[38]:


tree=DecisionTreeClassifier(min_samples_leaf=1000)
treeModel=tree.fit(x_train, y_train)
y_pred=treeModel.predict(x_test)
print("Number of mislabeled points out of a total %d points : %d" % 
(x_test.shape[0], (y_test != y_pred).sum()))
C=confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))
print("The confusion matrix:\n", C)
print(f"Error rate: {(1.-sum(np.diag(C))/C.sum()):5.3}")
sns.heatmap(C, cmap="Spectral")
plt.show()
print("Visualize the feature importances over all trees")
sns.heatmap(treeModel.feature_importances_.reshape((28,28)), cmap=plt.cm.Spectral)
plt.show("Feature importances in the tree")
plt.show()
input("Hit Enter to see the tree and some of its characteristics")
fig=plt.figure(dpi=600)
sub=fig.add_subplot(1,1,1)
plot_tree(treeModel)
plt.show()
print(export_text(treeModel))
print("Maximum Depth of the the tree is ", treeModel.tree_.max_depth)
print("Number of nodes of the tree is ",treeModel.tree_.node_count)
print("Number of leaves (AKA number of regions) is ", treeModel.tree_.n_leaves)
print("Error rate: %5.3f"%   (1.-sum(np.diag(C))/C.sum()))


# ### Build the bar-chart of predicted probability distribution of each of the 10 examples and Label the x-axis by the actual names (e.g., Sandal, Shirt, etc.) and not numbers and also Made sure the labels readable by writing them with a 90 degree rotation.

# In[39]:


l = np.array(y_test)
g = np.array(y_pred)


# In[40]:


for i in range(len(y_test)):
    if input("Do you wish to see performance on the next test digit? y/n") == 'n':
        break
    plt.imshow(np.array(x_test[i]).reshape((28,28)),cmap=plt.cm.binary)
    plt.show()
    print("Actual value: ",l[i], "  predicted value: ", g[i])
    print("Predicted probability distribution:")
    #print("Probabilities:\n", predProb[i])
    [print("p(%d): %5.2f "% (j, predProb[i,j])) for j in range(predProb.shape[1])]
    plt.bar(names,height=predProb[i])
    plt.xticks(range(0,10), label = names, rotation = 90)
    plt.show()


# ### Using CART with min_samples_leaf = 2000 with decision tree , accuarcy and confusion matrix

# In[41]:


tree=DecisionTreeClassifier(min_samples_leaf=2000)
treeModel=tree.fit(x_train, y_train)
y_pred=treeModel.predict(x_test)
print("Number of mislabeled points out of a total %d points : %d" % 
(x_test.shape[0], (y_test != y_pred).sum()))
C=confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))
print("The confusion matrix:\n", C)
print(f"Error rate: {(1.-sum(np.diag(C))/C.sum()):5.3}")
sns.heatmap(C, cmap="Spectral")
plt.show()
print("Visualize the feature importances over all trees")
sns.heatmap(treeModel.feature_importances_.reshape((28,28)), cmap=plt.cm.Spectral)
plt.show("Feature importances in the tree")
plt.show()
input("Hit Enter to see the tree and some of its characteristics")
fig=plt.figure(dpi=600)
sub=fig.add_subplot(1,1,1)
plot_tree(treeModel)
plt.show()
print(export_text(treeModel))
print("Maximum Depth of the the tree is ", treeModel.tree_.max_depth)
print("Number of nodes of the tree is ",treeModel.tree_.node_count)
print("Number of leaves (AKA number of regions) is ", treeModel.tree_.n_leaves)
print("Error rate: %5.3f"%   (1.-sum(np.diag(C))/C.sum()))


# ### Build the bar-chart of predicted probability distribution of each of the 10 examples and Label the x-axis by the actual names (e.g., Sandal, Shirt, etc.) and not numbers and also Made sure the labels readable by writing them with a 90 degree rotation.

# In[42]:


l = np.array(y_test)
g = np.array(y_pred)


# In[43]:


for i in range(len(y_test)):
    if input("Do you wish to see performance on the next test digit? y/n") == 'n':
        break
    plt.imshow(np.array(x_test[i]).reshape((28,28)),cmap=plt.cm.binary)
    plt.show()
    print("Actual value: ",l[i], "  predicted value: ", g[i])
    print("Predicted probability distribution:")
    #print("Probabilities:\n", predProb[i])
    [print("p(%d): %5.2f "% (j, predProb[i,j])) for j in range(predProb.shape[1])]
    plt.bar(names,height=predProb[i])
    plt.xticks(range(0,10), label = names, rotation = 90)
    plt.show()


# ### Result We notice that using CART Method with max_depth = 10 gives an highest accuarcy from all 4 different instances which is 72%

# # Question 3e
# 
# ## Random Forest
# 
# ### Used Random Forest Classifier and calculated accuracy , out-of-bag error ,  confusion matrix  and classification report for the test set

# In[44]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,n_jobs=-
1,oob_score=True)
rfModel=rf.fit(x_train,y_train)
y_pred=rfModel.predict(x_test)
y_predP=rfModel.predict_proba(x_test)
C=confusion_matrix(y_test,y_pred)
print(C)
print(f"Test Accuracy for the tree model: {(C[0,0]+C[1,1])/np.sum(C):5.3}")
print(f"The Out-of- bag score: {rfModel.oob_score_}")
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test , y_pred))


# #### The accuracy score comes 0.778

#  ### Resut : Using Random Forest Classifier we get an accuarcy of 78%
# 
# # Summary KNN gives highest accuarcy of 80% when compared with other classifier
