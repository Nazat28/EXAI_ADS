#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


pip install keras


# In[3]:


pip install seaborn


# In[4]:


pip install numpy==1.21


# In[5]:


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


df=pd.read_csv('modified_data.csv')

df.shape


# In[6]:


# drop the columns 'pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z'
# df=df.drop(['spd_y', 'pos_y', 'spd_x','pos_x'], axis=1)
data_df=df
df.head()


# In[7]:


from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
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
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(6,)))
model.add(Dropout(0.1))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(2,activation=tf.keras.activations.softmax))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


#START TIMER MODEL
start = time.time()
model.fit(X_train_scaled, y_train, epochs=5, batch_size=100)
end = time.time()


# In[91]:


model.summary()


# In[78]:


predictions =model.predict(X_test_scaled)
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


# In[26]:


import shap
# Fits the explainer
explainer = shap.Explainer(model.predict, X_test_scaled)
start_index = 0
end_index = 100
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test_scaled[start_index:end_index])

# Define a list of feature names
features = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
from scipy.special import softmax

def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

# Print the feature importances based on the SHAP values
print_feature_importances_shap_values(shap_values, features)


# In[27]:


import time
import shap

# start time
start=time.time()

start_index =0
end_index = 100
# Assuming 'feature_cols' contains the names of your feature columns
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']

explainer = shap.DeepExplainer(model, X_test_scaled[start_index:end_index].astype('float'))
shap_values = explainer.shap_values(X_test_scaled[start_index:end_index].astype('float'))


# In[28]:


import shap
import time
import matplotlib.pyplot as plt

# Assuming shap_values and other variables are defined before this point

# Start time
start = time.time()

# Generate a summary plot
shap.summary_plot(shap_values=shap_values,
                  features=X_test_scaled[0:1],
                  class_names=['Benign', 'Anomalous'],
                  feature_names=feature_cols,  # pass feature names as a list
                  show=False)

# Waterfall plot
shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[1], data=X_test_scaled[0], feature_names=feature_cols), max_display=6)

# End time
end = time.time()
print('SHAP time for DNN:', (end - start), 'sec')

# Show the plots
plt.show()


# In[ ]:


shap.summary_plot(shap_values=shap_values,
                  features=X_test_scaled[0:1],
                  class_names=['Benign', 'Anomalous'],
                  feature_names=feature_cols,  # pass feature names as a list
                  show=False)

#end time
end=time.time()
print('SHAP time for DNN: ',(end - start), 'sec')


# In[13]:


row_to_explain = np.array([X_test_scaled[0]])


# In[14]:


import shap

# model and data already defined

# row_to_explain = X_test_scaled[0] # replace with index of row you want to explain

explainer = shap.DeepExplainer(model, row_to_explain)
shap_values = explainer.shap_values(row_to_explain)

print(shap_values)


# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import shap

# Assume model, X_test_scaled, and other data is already prepared 

row_to_explain = np.array([X_test_scaled[0]]) # shape (1, data_dim)

explainer = shap.DeepExplainer(model, row_to_explain)
shap_values = explainer.shap_values(row_to_explain)[0] # select first output

shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values, row_to_explain)
plt.show()


# In[15]:


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
                                 model.predict_proba, 
                                 num_features=len(feature_cols))

# Plot the Lime explanation
fig = exp.as_pyplot_figure()
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


# In[10]:


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
samples = 50000

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
    
    exp = explainer.explain_instance(test[i], model.predict_proba, num_features=num_columns, top_labels=num_columns)
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


# In[21]:


print('---------------------------------------------------------------------------------')
print('Generating SHAP explanation')
print('---------------------------------------------------------------------------------')
print('')

test = X_test
train = X_train
start_index = 0
end_index = 500

explainer = shap.KernelExplainer(model.predict_proba, test[start_index:end_index])
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


# In[22]:


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


# In[8]:


from sklearn.linear_model import Lasso
import lime
import lime.lime_tabular

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    mode="classification", 
    training_labels=y_train,
    feature_names=feature_cols,
    class_names=['Benign','Anomalous'] 
)


# In[9]:


import time

start = time.time()

for i in range(1000):
  explanation = explainer.explain_instance(X_test_scaled[i], model.predict_proba)

end = time.time()

print('Total LIME time for 1000 samples:', (end - start), 'sec')


# In[26]:


instance_idx=11

print(y_test.iloc[instance_idx])
print(predictions[instance_idx])


# In[27]:


# start time
start=time.time()

x = X_test_scaled[instance_idx]

exp = explainer.explain_instance(x, model.predict_proba,num_samples=500, 
                                 distance_metric='cosine', model_regressor=Lasso()) 

exp.show_in_notebook(show_table=True, show_all=False)

#end time
end=time.time()
print('SHAP time for DNN: ',(end - start), 'sec')

