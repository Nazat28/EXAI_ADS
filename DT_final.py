#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


df=pd.read_csv('modified_data.csv')



df.head()


# In[2]:


# drop the columns 'pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z'
# df = df.drop(['spd_y','pos_x','spd_x'], axis=1)
data_df=df


# In[3]:


df.head()


# In[4]:


from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z'] #pos_y','pos_z', 'spd_x', 'spd_y', 'spd_z']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable

from sklearn.model_selection import train_test_split


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create a StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# transform the training and testing data using the scaler object
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[5]:


# see how the scaled data looks
#print(len(X_train_scaled))
#print(len(X_test_scaled))

from sklearn.tree import DecisionTreeClassifier
# train the decision tree classifier on the normalized training data
clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=50, min_samples_leaf=20)
clf.fit(X_train_scaled, y_train)


# In[21]:


predictions=clf.predict(X_test_scaled)
predictions


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))






# In[6]:


import shap


# In[7]:


import shap
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']

# Get the class names
class_names = ['Benign', 'Anomalous']

# Create a TreeExplainer
explainer = shap.TreeExplainer(clf)

# Set the number of runs
num_runs = 3

for run in range(num_runs):
    # Set the index of the sample for the current run
    sample_index = run

    # Get SHAP values and object for the specified sample
    shap_values = explainer.shap_values(X_test_scaled[sample_index:sample_index + 1])
    shap_obj = explainer(X_test_scaled[sample_index:sample_index + 1])

    # Plot the SHAP summary plot
    shap.summary_plot(shap_values=shap_values,
                      features=feature_cols,
                      class_names=class_names,
                      show=False)

    # Display feature names at the end of each run
    print(f"\nFeature names for run {run + 1}:", feature_cols)
    
    # Show the plot
    plt.show()


# In[8]:


import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
# Get the class names
class_names=['Benign','Anomalous']

# Get the feature names
feature_names = list(feature_cols)

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train_scaled),
                    feature_names=feature_names, 
                    class_names=class_names,                          
                    verbose=True, mode='classification')


# In[9]:


import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
# Get the class names
class_names=['Benign','Anomalous']

# Get the feature names
feature_names = list(feature_cols)

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train_scaled),
                                                    feature_names=feature_names, 
                                                    class_names=class_names,                          
                                                    verbose=True, mode='classification')

# Set the index of the sample you want to explain
sample_index = 1

# Get the explanation for the specified sample
exp = explainer.explain_instance(np.array(X_test_scaled[sample_index]), 
                                 clf.predict_proba, 
                                 num_features=len(feature_cols))

# Plot the Lime explanation
fig = exp.as_pyplot_figure()
plt.show()


# In[23]:


print('SPARSITY EXPERIMENT INITIATING')


# In[20]:


print('---------------------------------------------------------------------------------')
print('Generating SHAP explanation')
print('---------------------------------------------------------------------------------')
print('')

test = X_test
train = X_train
start_index = 0
end_index = 500

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


# In[21]:


print('---------------------------------------------------------------------------------')

print('---------------------------------------------------------------------------------')
print('Generating SHAP Sparsity Graph')
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


# In[17]:


import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'feature_cols' contains the names of your feature columns
feature_cols =  ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
# Get the class names
class_names=['Benign','Anomalous']

# Get the feature names
feature_names = list(feature_cols)

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train_scaled),
                    feature_names=feature_names, 
                    class_names=class_names,                          
                    verbose=True, mode='classification')


# In[18]:


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


# In[19]:


print('---------------------------------------------------------------------------------')

print('---------------------------------------------------------------------------------')
print('Generating LIME Sparsity Graph')
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

