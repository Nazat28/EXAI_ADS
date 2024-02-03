#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


pip install seaborn


# In[3]:


pip install keras


# In[4]:


pip install numpy


# In[5]:


pip install pandas


# In[6]:


import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import shap


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv('labeled_dataset_CPSS.csv')

df.shape


# In[7]:


# drop the columns 'Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time'
# df=df.drop(['Formality', 'Frequency', 'Plausibility','Headway Time', 'Protocol','Consistency', 'Correlation','Location'], axis=1)
data_df=df
df.head()


# In[14]:


from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time']
X = data_df[feature_cols] # Features
y = data_df.Label # Target variable

from sklearn.model_selection import train_test_split


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout

# Set random seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)



import time


# In[77]:


# define the keras model
clf = tf.keras.Sequential()
clf.add(tf.keras.Input(shape=(10,)))
clf.add(Dropout(0.1))
clf.add(tf.keras.layers.Dense(16,activation='relu'))
clf.add(tf.keras.layers.Dense(16,activation='relu'))
clf.add(tf.keras.layers.Dense(2,activation=tf.keras.activations.softmax))
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
clf.summary()


#START TIMER MODEL
start = time.time()
clf.fit(X_train_scaled, y_train, epochs=5, batch_size=100)
end = time.time()


# In[91]:


clf.summary()


# In[78]:


predictions =clf.predict(X_test_scaled)
predictions


# In[79]:


predictions = predictions.argmax(axis=1)


# In[80]:

from sklearn.metrics import precision_score, f1_score, recall_score

precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
recall= recall_score(y_test, predictions)


# In[81]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[12]:


import time
import shap
import matplotlib.pyplot as plt

start = time.time()

start_index = 0
end_index = 1

# Your existing code for creating the explainer and computing SHAP values
explainer = shap.KernelExplainer(clf.predict_proba, X_test_scaled[start_index:end_index])
shap_values = explainer.shap_values(X_test_scaled[start_index:end_index])

# Assuming 'feature_cols' contains the names of your feature columns
feature_cols =  ['Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time']

# Generate the summary plot using matplotlib
shap.summary_plot(shap_values=shap_values,
                  features=X_test_scaled[start_index:end_index],
                  feature_names=feature_cols,
                  class_names=['Benign', 'Anomalous'],
                  show=False)

# Get the current axis (assuming it's the only axis in the plot)
ax = plt.gca()

# Move the legend to the desired location (e.g., lower-right)
ax.legend(loc='lower right')

# Show the plot
plt.show()

# END timer
end = time.time()
print('SHAP time for DT: ',(end - start), 'sec')


# In[15]:


import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time']
# Get the class names
class_names=['Benign','Anomalous']

# Get the feature names
feature_names = list(feature_cols)

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train_scaled),
                    feature_names=feature_names, 
                    class_names=class_names,                          
                    verbose=True, mode='classification')
import time
print('---------------------------------------------------------------------------------')
print('Generating LIME explanation')
print('---------------------------------------------------------------------------------')
print('')



# test.pop ('Label')
print('------------------------------------------------------------------------------')

#START TIMER MODEL
start = time.time()
train =  X_train_scaled
test = X_test_scaled
test2 = test
# test = test.to_numpy()
samples = 1

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names= feature_cols, class_names=class_names , discretize_continuous=True)


#creating dict 
feat_list = feature_cols
# print(feat_list)

feat_dict = dict.fromkeys(feat_list, 0)
# print(feat_dict)
c = 0

num_columns = data_df.shape[1] - 1
feature_name = feature_cols
feature_name.sort()
# print('lista',feature_name)
feature_val = []

for i in range(0,num_columns): feature_val.append(0)

for i in range(0,samples):

# i = sample
    # exp = explainer.explain_instance(test[i], rf.predict_proba)
    
    exp = explainer.explain_instance(test[i], clf.predict_proba, num_features=num_columns, top_labels=num_columns)
    # exp.show_in_notebook(show_table=True, show_all=True)
    
    #lime list to string
    lime_list = exp.as_list()
    for i in range(0,len(lime_list)):
    #---------------------------------------------------
    #fix
        my_string = lime_list[i][0]
        for index, char in enumerate(my_string):
            if char.isalpha():
                first_letter_index = index
                break  # Exit the loop when the first letter is found

        my_string = my_string[first_letter_index:]
        modified_tuple = list(lime_list[i])
        modified_tuple[0] = my_string
        lime_list[i] = tuple(modified_tuple)

    #---------------------------------------------------
    
    
    lime_list.sort()
    # print(lime_list)
    for j in range (0,num_columns): feature_val[j]+= abs(lime_list[j][1])
    # print ('debug here',lime_list[1][1])

    # lime_str = ' '.join(str(x) for x in lime_list)
    # print(lime_str)


    #lime counting features frequency 
    # for i in feat_list:
    #     if i in lime_str:
    #         #update dict
    #         feat_dict[i] = feat_dict[i] + 1
    
    c = c + 1 
    print ('progress',100*(c/samples),'%')

# Define the number you want to divide by
divider = samples

# Use a list comprehension to divide all elements by the same number
feature_val = [x / divider for x in feature_val]

# for item1, item2 in zip(feature_name, feature_val):
#     print(item1, item2)


# Use zip to combine the two lists, sort based on list1, and then unzip them
zipped_lists = list(zip(feature_name, feature_val))
zipped_lists.sort(key=lambda x: x[1],reverse=True)

# Convert the sorted result back into separate lists
sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

# print(sorted_list1)
# print(sorted_list2)
print('----------------------------------------------------------------------------------------------------------------')

for item1, item2 in zip(sorted_list1, sorted_list2):
    print(item1, item2)

for k in sorted_list1:
#     with open(output_file_name, "a") as f:print("df.pop('",k,"')", sep='', file = f)
    print("df.pop('",k,"')", sep='')

print('---------------------------------------------------------------------------------')

# # print(feat_dict)
# # Sort values in descending order
# for k,v in sorted(feat_dict.items(), key=lambda x: x[1], reverse=True):
#   print(k,v)

# for k,v in sorted(feat_dict.items(), key=lambda x: x[1], reverse=True):
#   print("df.pop('",k,"')", sep='')

print('---------------------------------------------------------------------------------')


end = time.time()
print('ELAPSE TIME LIME GLOBAL: ',(end - start)/60, 'min')
print('---------------------------------------------------------------------------------')


# In[49]:


import time
import shap

# start time
start=time.time()

start_index = 0
end_index = 1

# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time']

explainer = shap.DeepExplainer(clf, X_test_scaled[start_index:end_index].astype('float'))
shap_values = explainer.shap_values(X_test_scaled[start_index:end_index].astype('float'))

shap.summary_plot(shap_values=shap_values,
                  features=X_test_scaled[start_index:end_index],
                  class_names=['Normal', 'Anomalous'],
                  feature_names=feature_cols,  # pass feature names as a list
                  show=False)

#end time
end=time.time()
print('SHAP time for DNN: ',(end - start), 'sec')


# In[50]:


print('---------------------------------------------------------------------------------')
print('Generating SHAP explanation')
print('---------------------------------------------------------------------------------')
print('')

test = X_test
train = X_train
start_index = 0
end_index = 1

explainer = shap.KernelExplainer(clf.predict_proba, test[start_index:end_index])
shap_values = explainer.shap_values(test[start_index:end_index])

# print('labels: ', Label)
# y_labels = Label
# shap.summary_plot(shap_values = shap_values,
#                   features = test[start_index:end_index],
#                  class_names=[y_labels[0],y_labels[1],y_labels[2],y_labels[3],y_labels[4],y_labels[5],y_labels[6]],show=False)

# plt.savefig('SVM_Shap_Summary_global.png')
# plt.clf()


vals= np.abs(shap_values).mean(1)
feature_importance = pd.DataFrame(list(zip(train.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
feature_importance.head()
print(feature_importance.to_string())



print('---------------------------------------------------------------------------------')
# feature_importance_vals = 'feature_importance_vals'  # Replace with the name of the column you want to extract
feature_val = feature_importance['feature_importance_vals'].tolist()

# col_name = 'col_name'  # Replace with the name of the column you want to extract
feature_name = feature_importance['col_name'].tolist()


# for item1, item2 in zip(feature_name, feature_val):
#     print(item1, item2)


# Use zip to combine the two lists, sort based on list1, and then unzip them
zipped_lists = list(zip(feature_name, feature_val))
zipped_lists.sort(key=lambda x: x[1],reverse=True)

# Convert the sorted result back into separate lists
sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]



# for k in sorted_list1:
#   with open(output_file_name, "a") as f:print("df.pop('",k,"')", sep='',file = f)
# with open(output_file_name, "a") as f:print("Trial_ =[", file = f)
# for k in sorted_list1:
#   with open(output_file_name, "a") as f:print("'",k,"',", sep='', file = f)
# with open(output_file_name, "a") as f:print("]", file = f)
# print('---------------------------------------------------------------------------------')

for k in sorted_list1:
    None
#     print("df.pop('",k,"')", sep='')
#     print("Trial_ =[")
for k in sorted_list1:
    print("'",k,"',", sep='')
#     print("]")
print('---------------------------------------------------------------------------------')


# In[7]:


print('---------------------------------------------------------------------------------')

print('---------------------------------------------------------------------------------')
print('Generating Sparsity Graph')
print('---------------------------------------------------------------------------------')
print('')
# print(feature_importance)

# Find the minimum and maximum values in the list
min_value = min(feature_val)
max_value = max(feature_val)

# Normalize the list to the range [0, 1]
try:
    normalized_list = [(x - min_value) / (max_value - min_value) for x in feature_val]
except:
    normalized_list = [0 for x in feature_val]

# print(feature_name,normalized_list,'\n')
# for item1, item2 in zip(feature_name, normalized_list):
#     print(item1, item2)

#calculating Sparsity

# Define the threshold
threshold = 1e-10

# Initialize a count variable to keep track of values below the threshold
count_below_threshold = 0

# Iterate through the list and count values below the threshold
# for value in normalized_list:
#     if value < threshold:
#         count_below_threshold += 1

# Sparsity = count_below_threshold/len(normalized_list)
Spar = []
# print('Sparsity = ',Sparsity)
X_axis = []
#----------------------------------------------------------------------------
for i in range(1,11):
    i/10
    threshold = i/10
    print(threshold)
    for value in normalized_list:
        if value<= threshold:
            count_below_threshold += 1

    Sparsity = count_below_threshold/len(normalized_list)
    Spar.append(Sparsity)
    X_axis.append(i/10)
    count_below_threshold = 0


# #---------------------------------------------------------------------------

f:print('y_axis_RF = ', Spar ,'')

print('x_axis_RF = ', X_axis ,'')

# plt.clf()


# In[11]:


import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = [ 'Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time']
# Get the class names
class_names=['Benign','Anomalous']

# Get the feature names
feature_names = list(feature_cols)

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train_scaled),
                    feature_names=feature_names, 
                    class_names=class_names,                          
                    verbose=True, mode='classification')


# In[12]:


import time

start = time.time()

for i in range(3000):
  explanation = explainer.explain_instance(X_test_scaled[i], clf.predict_proba)

end = time.time()

print('Total LIME time for 1000 samples:', (end - start), 'sec')


# In[9]:


import time
print('---------------------------------------------------------------------------------')
print('Generating LIME explanation')
print('---------------------------------------------------------------------------------')
print('')



# test.pop ('Label')
print('------------------------------------------------------------------------------')

#START TIMER MODEL
start = time.time()
train =  X_train_scaled
test = X_test_scaled
test2 = test
# test = test.to_numpy()
samples = 500

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names= feature_cols, class_names=class_names , discretize_continuous=True)


#creating dict 
feat_list = feature_cols
# print(feat_list)

feat_dict = dict.fromkeys(feat_list, 0)
# print(feat_dict)
c = 0

num_columns = data_df.shape[1] - 1
feature_name = feature_cols
feature_name=list(feature_name)
feature_name.sort()
# print('lista',feature_name)
feature_val = []

for i in range(0,num_columns): feature_val.append(0)

for i in range(0,samples):

# i = sample
    # exp = explainer.explain_instance(test[i], rf.predict_proba)
    
    exp = explainer.explain_instance(test[i], clf.predict_proba, num_features=num_columns, top_labels=num_columns)
    # exp.show_in_notebook(show_table=True, show_all=True)
    
    #lime list to string
    lime_list = exp.as_list()
    for i in range(0,len(lime_list)):
    #---------------------------------------------------
    #fix
        my_string = lime_list[i][0]
        for index, char in enumerate(my_string):
            if char.isalpha():
                first_letter_index = index
                break  # Exit the loop when the first letter is found

        my_string = my_string[first_letter_index:]
        modified_tuple = list(lime_list[i])
        modified_tuple[0] = my_string
        lime_list[i] = tuple(modified_tuple)

    #---------------------------------------------------
    
    
    lime_list.sort()
    # print(lime_list)
    for j in range (0,num_columns): feature_val[j]+= abs(lime_list[j][1])
    # print ('debug here',lime_list[1][1])

    # lime_str = ' '.join(str(x) for x in lime_list)
    # print(lime_str)


    #lime counting features frequency 
    # for i in feat_list:
    #     if i in lime_str:
    #         #update dict
    #         feat_dict[i] = feat_dict[i] + 1
    
    c = c + 1 
    print ('progress',100*(c/samples),'%')

# Define the number you want to divide by
divider = samples

# Use a list comprehension to divide all elements by the same number
feature_val = [x / divider for x in feature_val]

# for item1, item2 in zip(feature_name, feature_val):
#     print(item1, item2)


# Use zip to combine the two lists, sort based on list1, and then unzip them
zipped_lists = list(zip(feature_name, feature_val))
zipped_lists.sort(key=lambda x: x[1],reverse=True)

# Convert the sorted result back into separate lists
sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

# print(sorted_list1)
# print(sorted_list2)
print('----------------------------------------------------------------------------------------------------------------')

for item1, item2 in zip(sorted_list1, sorted_list2):
    print(item1, item2)

for k in sorted_list1:
#     with open(output_file_name, "a") as f:print("df.pop('",k,"')", sep='', file = f)
    print("df.pop('",k,"')", sep='')

print('---------------------------------------------------------------------------------')

# # print(feat_dict)
# # Sort values in descending order
# for k,v in sorted(feat_dict.items(), key=lambda x: x[1], reverse=True):
#   print(k,v)

# for k,v in sorted(feat_dict.items(), key=lambda x: x[1], reverse=True):
#   print("df.pop('",k,"')", sep='')

print('---------------------------------------------------------------------------------')


end = time.time()
print('ELAPSE TIME LIME GLOBAL: ',(end - start)/60, 'min')
print('---------------------------------------------------------------------------------')


# In[10]:


print('---------------------------------------------------------------------------------')

print('---------------------------------------------------------------------------------')
print('Generating Sparsity Graph')
print('---------------------------------------------------------------------------------')
print('')
# print(feature_importance)

# Find the minimum and maximum values in the list
min_value = min(feature_val)
max_value = max(feature_val)

# Normalize the list to the range [0, 1]
try:
    normalized_list = [(x - min_value) / (max_value - min_value) for x in feature_val]
except:
    normalized_list = [0 for x in feature_val]

# print(feature_name,normalized_list,'\n')
# for item1, item2 in zip(feature_name, normalized_list):
#     print(item1, item2)

#calculating Sparsity

# Define the threshold
threshold = 1e-10

# Initialize a count variable to keep track of values below the threshold
count_below_threshold = 0

# Iterate through the list and count values below the threshold
# for value in normalized_list:
#     if value < threshold:
#         count_below_threshold += 1

# Sparsity = count_below_threshold/len(normalized_list)
Spar = []
# print('Sparsity = ',Sparsity)
X_axis = []
#----------------------------------------------------------------------------
for i in range(1,11):
    i/10
    threshold = i/10
    print(threshold)
    for value in normalized_list:
        if value<= threshold:
            count_below_threshold += 1

    Sparsity = count_below_threshold/len(normalized_list)
    Spar.append(Sparsity)
    X_axis.append(i/10)
    count_below_threshold = 0


# #---------------------------------------------------------------------------

f:print('y_axis_RF = ', Spar ,'')

print('x_axis_RF = ', X_axis ,'')

# plt.clf()

