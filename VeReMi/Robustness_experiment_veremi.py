#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore') 

from adversarial_models import * 
from utils import *
from get_data_veremi import *


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap
from copy import deepcopy


# In[3]:


import pandas as pd
import numpy as np

# Get the data set and do some preprocessing
df = pd.read_csv('modified_data.csv')
np.random.seed(100)

# Define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']
X = df[feature_cols]  # Features
y = df.attackerType  # Target variable

# Add a random column -- this is what we'll have LIME/SHAP explain.
X['unrelated_column'] = np.random.choice([0, 1], size=X.shape[0])


# In[4]:


X


# In[5]:


y


# In[6]:


features = [c for c in X]
categorical_feature_name = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z', 'unrelated_column']
categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]


# In[7]:


race_indc = features.index('pos_x')
unrelated_indcs = features.index('unrelated_column')
X = X.values 


# In[8]:


X


# In[9]:


class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self, X):
        return np.array([0 if x[race_indc] > 0 else 1 for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
class innocuous_model_psi:
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict(self, X):
        return np.array([0 if x[unrelated_indcs] < 0 else 1 for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)


# In[12]:


start_index = 0
end_index = 100


# In[13]:


# Split the data and normalize
xtrain,xtest,ytrain,ytest = train_test_split(X,y)
xtest_not_normalized = deepcopy(xtest)
ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)


# In[14]:


# Train the adversarial model for LIME with f and psi 
adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).\
            train(xtrain[start_index:end_index], ytrain[start_index:end_index], feature_names=features)


# In[22]:


# Let's just look at a the first example in the test set
ex_indc = np.random.choice(xtest.shape[0])
# print(xtest.shape[0])
indices = np.where(ytest == 1)
index_positions = indices[0]
# print(index_positions)
ex_indc = index_positions[0]
# To get a baseline, we'll look at LIME applied to the biased model f
normal_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain,feature_names=adv_lime.get_column_names(),
                                                          discretize_continuous=False,
                                                          categorical_features=categorical_feature_indcs)

normal_exp = normal_explainer.explain_instance(xtest[ex_indc], racist_model_f().predict_proba).as_list()

print('------------------------------------------------------------------')
print('real value', ytest[ex_indc:ex_indc+1])
# print(ytest)
print('------------------------------------------------------------------')
print('------------------------------------------------------------------')
# print(xtest[ex_indc])
# print(xtest)
print('predicted value', racist_model_f().predict(xtest)[ex_indc])

print('------------------------------------------------------------------')


# pred = racist_model_f().predict_proba(xtest[ex_indc])                             
# print('prediction', pred)
print ("Explanation on biased f:\n",normal_exp[:3],"\n\n")

# Now, lets look at the explanations on the adversarial model 
adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain,feature_names=adv_lime.get_column_names(), 
                                                       discretize_continuous=False,
                                                       categorical_features=categorical_feature_indcs)

adv_exp = adv_explainer.explain_instance(xtest[ex_indc], adv_lime.predict_proba).as_list()

print ("Explanation on adversarial model:\n",adv_exp[:3],"\n")

print("Prediction fidelity: {0:3.2}".format(adv_lime.fidelity(xtest[ex_indc:ex_indc+1])))



# In[23]:


# Let's just look at a the first example in the test set
ex_indc = np.random.choice(xtest.shape[0])

# To get a baseline, we'll look at LIME applied to the biased model f
normal_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain,feature_names=adv_lime.get_column_names(),
                                                          discretize_continuous=False,
                                                          categorical_features=categorical_feature_indcs)

normal_exp = normal_explainer.explain_instance(xtest[ex_indc], racist_model_f().predict_proba).as_list()

print ("Explanation on biased f:\n",normal_exp[:3],"\n\n")

# Now, lets look at the explanations on the adversarial model 
adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain,feature_names=adv_lime.get_column_names(), 
                                                       discretize_continuous=False,
                                                       categorical_features=categorical_feature_indcs)

adv_exp = adv_explainer.explain_instance(xtest[ex_indc], adv_lime.predict_proba).as_list()


print ("Explanation on adversarial model:\n",adv_exp[:3],"\n")

print("Prediction fidelity: {0:3.2}".format(adv_lime.fidelity(xtest[ex_indc:ex_indc+1])))


# In[24]:


# 4 buckets, 3 graphs
dict_biased_1 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}
dict_biased_2 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}
dict_biased_3 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}
attack_1 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}

attack_2 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}

attack_3 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}


# In[1]:


# features analyzed
biased_feature = 'pos_x'
Unrelated_1_feature = 'unrelated_column'
# Unrelated_2_feature = 'unrelated_column'
samples = 2



# In[27]:


# explainer = lime.lime_tabular.LimeTabularExplainer(test, feature_names= list(test2.columns.values) , class_names=label , discretize_continuous=True)

#creating dict 
feat_list = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z', 'unrelated_column']
# print(feat_list)

# feat_dict = dict.fromkeys(feat_list, 0)
c = 0

num_columns = len(feat_list)
feature_name = feat_list
feature_name_2 = feat_list
feature_name_2.sort()
feature_name.sort()
feature_val = []
feature_val_2 = []
feature_val_abs = []
feature_val_abs_2 = []

# position = y_labels.index(rf.predict(Dos_sample2))
# position =  np.argmax(rf.predict_proba(((Dos_sample2))))
# print(len(y_labels))
# print(rf.predict(Dos_sample2))


# sample = Dos_sample
# samples = 1 
# sample = PS_sample


for i in range(0,num_columns): 
    feature_val.append(0)
    feature_val_abs.append(0)
    feature_val_2.append(0)
    feature_val_abs_2.append(0)    

# # i = sample
#     # exp = explainer.explain_instance(test[i], rf.predict_proba)

#     exp = explainer.explain_instance(sample, rf.predict_proba, num_features=num_columns, top_labels=len(y_labels))
#     exp.show_in_notebook(show_table=True, show_all=True)
# Let's just look at a the first example in the test set
# ex_indc = np.random.choice(xtest.shape[0])

samples = 1000

for j in range(0,samples):
    ex_indc = j
    # print(xtest.shape[0])
    # indices = np.where(ytest == 1)
    # index_positions = indices[0]
    # print(index_positions)
    # ex_indc = index_positions[0]
    # To get a baseline, we'll look at LIME applied to the biased model f
    normal_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain,feature_names=adv_lime.get_column_names(),
                                                        discretize_continuous=False,
                                                        categorical_features=categorical_feature_indcs)

    normal_exp = normal_explainer.explain_instance(xtest[ex_indc], racist_model_f().predict_proba).as_list()

    # print('------------------------------------------------------------------')
    # print('real value', ytest[ex_indc:ex_indc+1])
    # print(ytest)
    # print('------------------------------------------------------------------')
    # print('------------------------------------------------------------------')
    # print(xtest[ex_indc])
    # print(xtest)
    # print('predicted value', racist_model_f().predict(xtest)[ex_indc])

    # print('------------------------------------------------------------------')


    # pred = racist_model_f().predict_proba(xtest[ex_indc])                             
    # print('prediction', pred)
    # print ("Explanation on biased f:\n",normal_exp[:3],"\n\n")

    # Now, lets look at the explanations on the adversarial model 
    adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain,feature_names=adv_lime.get_column_names(), 
                                                    discretize_continuous=False,
                                                    categorical_features=categorical_feature_indcs)

    adv_exp = adv_explainer.explain_instance(xtest[ex_indc], adv_lime.predict_proba).as_list()

    # print ("Explanation on adversarial model:\n",adv_exp[:3],"\n")

    # print("Prediction fidelity: {0:3.2}".format(adv_lime.fidelity(xtest[ex_indc:ex_indc+1])))

    lime_list = adv_exp
    lime_list_2 = normal_exp
    lime_list.sort()
    lime_list_2.sort()
    # print(lime_list)

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



    for i in range(0,len(lime_list_2)):
    #---------------------------------------------------
    #fix
        my_string_2 = lime_list_2[i][0]
        for index, char in enumerate(my_string_2):
            if char.isalpha():
                first_letter_index_2 = index
                break  # Exit the loop when the first letter is found  
        my_string_2 = my_string_2[first_letter_index_2:]
        modified_tuple_2 = list(lime_list_2[i])
        modified_tuple_2[0] = my_string_2
        lime_list_2[i] = tuple(modified_tuple_2)
            
    #---------------------------------------------------

    lime_list.sort()
    lime_list_2.sort()
    # print(lime_list)
    # for j in range (0,num_columns): feature_val[j]+= abs(lime_list[j][1])
    for j in range (0,num_columns):feature_val_abs[j] = abs(lime_list[j][1])
    for j in range (0,num_columns):feature_val_abs_2[j] = abs(lime_list_2[j][1])
    for j in range (0,num_columns):feature_val[j] = lime_list[j][1]
    for j in range (0,num_columns):feature_val_2[j] = lime_list_2[j][1]
    c = c + 1 
    print ('progress',100*(c/samples),'%')



    # Define the number you want to divide by
    divider = samples

    # Use a list comprehension to divide all elements by the same number
    feature_val = [x / divider for x in feature_val]
    feature_val_2 = [x / divider for x in feature_val_2]

    # for item1, item2 in zip(feature_name, feature_val):
    #     print(item1, item2)


    # Use zip to combine the two lists, sort based on list1, and then unzip them
    zipped_lists = list(zip(feature_name, feature_val,feature_val_abs))
    zipped_lists.sort(key=lambda x: x[2],reverse=True)

    zipped_lists_2 = list(zip(feature_name_2, feature_val_2,feature_val_abs_2))
    zipped_lists_2.sort(key=lambda x: x[2],reverse=True)

    # Convert the sorted result back into separate lists
    sorted_list1, sorted_list2,sorted_list3 = [list(x) for x in zip(*zipped_lists)]

    sorted_list1_2, sorted_list2_2,sorted_list3_2 = [list(x) for x in zip(*zipped_lists_2)]

    # print('Adversarial')
    # print(sorted_list1)
    # print(sorted_list2)
    # print(sorted_list3)

    # print('Biased')
    # print('----------------------------------------------------------------------------------------------------------------')
    # print(sorted_list1_2)
    # print(sorted_list2_2)
    # print(sorted_list3_2)
    # print('----------------------------------------------------------------------------------------------------------------')



    biased_df = pd.DataFrame({
            'shap_values': sorted_list2_2,
            'shap_values_abs': sorted_list3_2,
            'features': sorted_list1_2  
        })
    attack1_df = pd.DataFrame({
            'shap_values': sorted_list2 ,
            'shap_values_abs':sorted_list3,
            'features':  sorted_list1  
        })
    # info = [adv_shap_values,features]

    attack1_df.sort_values(by=['shap_values_abs'], ascending=False,inplace=True)
    biased_df.sort_values(by=['shap_values_abs'], ascending=False,inplace=True)

    attack1_df = attack1_df.reset_index(drop=True)
    biased_df = biased_df.reset_index(drop=True)

    # print('Attack')
    # print('------------------------------')
    # print (attack1_df)
    # print('Biased')
    # print('------------------------------')
    # print(biased_df)


    #biased columns

    #For 1st position
    # print(biased_df['shap_values'][0] )
    if biased_df['features'][0] == biased_feature: dict_biased_1['Biased'] = dict_biased_1['Biased'] + 1
    elif biased_df['features'][0] == Unrelated_1_feature: dict_biased_1['Unrelated_1'] = dict_biased_1['Unrelated_1'] + 1
    # elif biased_df['features'][0] == Unrelated_2_feature: dict_biased_1['Unrelated_2'] = dict_biased_1['Unrelated_2'] + 1
    else: dict_biased_1['Others'] = dict_biased_1['Others'] + 1

    #For 2st position
    if biased_df['features'][1] == biased_feature: dict_biased_2['Biased'] = dict_biased_2['Biased'] + 1
    elif biased_df['features'][1] == Unrelated_1_feature: dict_biased_2['Unrelated_1'] = dict_biased_2['Unrelated_1'] + 1
    # elif biased_df['features'][1] == Unrelated_2_feature: dict_biased_2['Unrelated_2'] = dict_biased_2['Unrelated_2'] + 1
    else: dict_biased_2['Others'] = dict_biased_2['Others'] + 1

    #For 3st position
    if biased_df['features'][2] == biased_feature: dict_biased_3['Biased'] = dict_biased_3['Biased'] + 1
    elif biased_df['features'][2] == Unrelated_1_feature: dict_biased_3['Unrelated_1'] = dict_biased_3['Unrelated_1'] + 1
    # elif biased_df['features'][2] == Unrelated_2_feature: dict_biased_3['Unrelated_2'] = dict_biased_3['Unrelated_2'] + 1
    else: dict_biased_3['Others'] = dict_biased_3['Others'] + 1


    #Attack 1 columns

    #For 1st position
    if attack1_df['features'][0] == biased_feature: attack_1['Biased'] = attack_1['Biased'] + 1
    elif attack1_df['features'][0] == Unrelated_1_feature: attack_1['Unrelated_1'] = attack_1['Unrelated_1'] + 1
    # elif attack1_df['features'][0] == Unrelated_2_feature: attack_1['Unrelated_2'] = attack_1['Unrelated_2'] + 1
    else: attack_1['Others'] = attack_1['Others'] + 1

    #For 2st position
    if attack1_df['features'][1] == biased_feature: attack_2['Biased'] = attack_2['Biased'] + 1
    elif attack1_df['features'][1] == Unrelated_1_feature: attack_2['Unrelated_1'] = attack_2['Unrelated_1'] + 1
    # elif attack1_df['features'][1] == Unrelated_2_feature: attack_2['Unrelated_2'] = attack_2['Unrelated_2'] + 1
    else: attack_2['Others'] = attack_2['Others'] + 1

    #For 3st position
    if attack1_df['features'][2] == biased_feature: attack_3['Biased'] = attack_3['Biased'] + 1
    elif attack1_df['features'][2] == Unrelated_1_feature: attack_3['Unrelated_1'] = attack_3['Unrelated_1'] + 1
    # elif attack1_df['features'][2] == Unrelated_2_feature: attack_3['Unrelated_2'] = attack_3['Unrelated_2'] + 1
    else: attack_3['Others'] = attack_3['Others'] + 1

# for item1, 


# In[28]:


print(dict_biased_1 )
print(dict_biased_2 )
print(dict_biased_3 )
print('--------------')
print(attack_1 )
print(attack_2 )
print(attack_3 )



# In[29]:


#Normalized
keys = ['Biased','Unrelated_1','Unrelated_2','Others' ]
for x in keys:

    dict_biased_1[x] = dict_biased_1[x]/samples
    dict_biased_2[x] = dict_biased_2[x]/samples
    dict_biased_3[x] = dict_biased_3[x]/samples

    attack_1[x] = attack_1[x]/samples
    attack_2[x] = attack_2[x]/samples
    attack_3[x] = attack_3[x]/samples


# In[48]:


# importing package
import matplotlib.pyplot as plt
import numpy as np
total = samples 
# create data
x = ['1st', '2nd', '3rd']
y1 = np.array([dict_biased_1['Biased'], dict_biased_2['Biased'], dict_biased_3['Biased']])
y2 = np.array([dict_biased_1['Unrelated_1'], dict_biased_2['Unrelated_1'], dict_biased_3['Unrelated_1']])
y3 = np.array([dict_biased_1['Others'], dict_biased_2['Others'], dict_biased_3['Others']])
y4 = np.array([dict_biased_1['Unrelated_2'], dict_biased_2['Unrelated_2'],dict_biased_3['Unrelated_2']])


# plot bars in stack manner
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.bar(x, y3, bottom=y1+y2, color='grey')
# plt.bar(x, y4, bottom=y1+y2+y3, color='g')
plt.xlabel("Features Importance Rank", fontsize = 18)
plt.ylabel("% Occurrence", fontsize = 18)
# plt.legend(["Biased", "Unrelated", "Unrelated_2", "Others"])
plt.legend(["Biased", "Unrelated", "Others"], fontsize = 14)

plt.show()



# In[49]:


# importing package
import matplotlib.pyplot as plt
import numpy as np
total = samples 
# create data
x = ['1st', '2nd', '3rd']
y1 = np.array([attack_1['Biased'], attack_2['Biased'], attack_3['Biased']])
y2 = np.array([attack_1['Unrelated_1'], attack_2['Unrelated_1'], attack_3['Unrelated_1']])
y3 = np.array([attack_1['Others'], attack_2['Others'],attack_3['Others']])
# y4 = np.array([attack_1['Unrelated_2'], attack_2['Unrelated_2'], attack_3['Unrelated_2']])

# plot bars in stack manner
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.bar(x, y3, bottom=y1+y2, color='grey')
# plt.bar(x, y4, bottom=y1+y2+y3, color='g')
plt.xlabel("Features Importance Rank", fontsize = 18)
plt.ylabel("% Occurrence", fontsize = 18)
# plt.legend(["Biased", "Unrelated_1", "Unrelated_2", "Others"])
plt.legend(["Biased", "Unrelated_1", "Others"], fontsize = 14)

plt.show()


# In[32]:


#

import time
import numpy as np

# Show the result of the model's explanation
explanation = normal_explainer.explain_instance(xtest[ex_indc], racist_model_f().predict_proba)
explanation.show_in_notebook(show_table=True, show_all=False)


# In[111]:


import time
import numpy as np


# Show the result of the model's explanation
explanation = adv_explainer.explain_instance(xtest[ex_indc], adv_lime.predict_proba)
explanation.show_in_notebook(show_table=True, show_all=False)



# In[38]:


# SHAP
# Train the adversarial model
adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).\
            train(xtrain[start_index:end_index], ytrain[start_index:end_index], feature_names=features)


# In[39]:


import numpy as np
import shap
import matplotlib.pyplot as plt

# Set the background distribution for the SHAP explainer using kmeans
background_distribution = shap.kmeans(xtrain, 10)

# Let's use the SHAP kernel explainer and grab a point to explain
to_examine = np.random.choice(xtest.shape[0])

# Explain the biased model
biased_kernel_explainer = shap.KernelExplainer(racist_model_f().predict, background_distribution)
biased_shap_values = biased_kernel_explainer.shap_values(xtest[to_examine:to_examine+1])

# Explain the adversarial model
adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
adv_shap_values = adv_kerenel_explainer.shap_values(xtest[to_examine:to_examine+1])

# Plot and save the SHAP summary plots.
print('Racist Model')
shap.summary_plot(biased_shap_values, feature_names=features, plot_type="bar")
plt.savefig('SHAP_biased_model.png')  # Save the summary plot for the biased model
plt.clf()  # Clear the plot before creating the next one

print('Adversarial Model')  # Corrected print statement
shap.summary_plot(adv_shap_values, feature_names=features, plot_type="bar")
plt.savefig('SHAP_adversarial_model.png')  # Save the summary plot for the adversarial model

print("Fidelity: {0:3.2}".format(adv_shap.fidelity(xtest[to_examine:to_examine+1])))


# In[40]:


xtest.shape[0]


# In[41]:


# 4 buckets, 3 graphs
dict_biased_1 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}
dict_biased_2 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}
dict_biased_3 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}
attack_1 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}

attack_2 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}

attack_3 = {
    'Biased': 0,
    'Unrelated_1':0,
    'Unrelated_2':0,
    'Others':0
}


# In[42]:


# features analyzed
biased_feature = 'pos_x'
Unrelated_1_feature = 'unrelated_column'
# Unrelated_2_feature = 'unrelated_column'
samples = 100



# In[43]:


for i in range(0,samples):   
 # Set the background distribution for the shap explainer using kmeans
    background_distribution = shap.kmeans(xtrain,10)

    # Let's use the shap kernel explainer and grab a point to explain
    to_examine = i

    # to_examine = np.random.choice(xtest.shape[0])

    # Explain the biased model
    biased_kernel_explainer = shap.KernelExplainer(racist_model_f().predict, background_distribution)
    biased_shap_values = biased_kernel_explainer.shap_values(xtest[to_examine:to_examine+1])

    # print(biased_kernel_explainer)

    # Explain the adversarial model
    adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
    adv_shap_values = adv_kerenel_explainer.shap_values(xtest[to_examine:to_examine+1])

    # Plot it using SHAP's plotting features.
    # shap.summary_plot(biased_shap_values, feature_names=features, plot_type="bar")
    # shap.summary_plot(adv_shap_values, feature_names=features, plot_type="bar")

    # print(adv_shap_values)
    # print(features)

    biased_df = pd.DataFrame({
            'shap_values': biased_shap_values[0],
            'shap_values_abs': abs(biased_shap_values[0]),
            'features': features    
        })
    attack1_df = pd.DataFrame({
            'shap_values': adv_shap_values[0],
            'shap_values_abs':abs(adv_shap_values[0]),
            'features': features    
        })
    # info = [adv_shap_values,features]

    attack1_df.sort_values(by=['shap_values_abs'], ascending=False,inplace=True)
    biased_df.sort_values(by=['shap_values_abs'], ascending=False,inplace=True)

    attack1_df = attack1_df.reset_index(drop=True)
    biased_df = biased_df.reset_index(drop=True)

    # print('Attack')
    # print('------------------------------')
    # print (attack1_df)
    # print('Biased')
    # print('------------------------------')
    # print(biased_df)


    #biased columns

    #For 1st position
    if biased_df['features'][0] == biased_feature: dict_biased_1['Biased'] = dict_biased_1['Biased'] + 1
    elif biased_df['features'][0] == Unrelated_1_feature: dict_biased_1['Unrelated_1'] = dict_biased_1['Unrelated_1'] + 1
    # elif biased_df['features'][0] == Unrelated_2_feature: dict_biased_1['Unrelated_2'] = dict_biased_1['Unrelated_2'] + 1
    else: dict_biased_1['Others'] = dict_biased_1['Others'] + 1

    #For 2st position
    if biased_df['features'][1] == biased_feature: dict_biased_2['Biased'] = dict_biased_2['Biased'] + 1
    elif biased_df['features'][1] == Unrelated_1_feature: dict_biased_2['Unrelated_1'] = dict_biased_2['Unrelated_1'] + 1
    # elif biased_df['features'][1] == Unrelated_2_feature: dict_biased_2['Unrelated_2'] = dict_biased_2['Unrelated_2'] + 1
    else: dict_biased_2['Others'] = dict_biased_2['Others'] + 1

    #For 3st position
    if biased_df['features'][2] == biased_feature: dict_biased_3['Biased'] = dict_biased_3['Biased'] + 1
    elif biased_df['features'][2] == Unrelated_1_feature: dict_biased_3['Unrelated_1'] = dict_biased_3['Unrelated_1'] + 1
    # elif biased_df['features'][2] == Unrelated_2_feature: dict_biased_3['Unrelated_2'] = dict_biased_3['Unrelated_2'] + 1
    else: dict_biased_3['Others'] = dict_biased_3['Others'] + 1


    #Attack 1 columns

    #For 1st position
    if attack1_df['features'][0] == biased_feature: attack_1['Biased'] = attack_1['Biased'] + 1
    elif attack1_df['features'][0] == Unrelated_1_feature: attack_1['Unrelated_1'] = attack_1['Unrelated_1'] + 1
    # elif attack1_df['features'][0] == Unrelated_2_feature: attack_1['Unrelated_2'] = attack_1['Unrelated_2'] + 1
    else: attack_1['Others'] = attack_1['Others'] + 1

    #For 2st position
    if attack1_df['features'][1] == biased_feature: attack_2['Biased'] = attack_2['Biased'] + 1
    elif attack1_df['features'][1] == Unrelated_1_feature: attack_2['Unrelated_1'] = attack_2['Unrelated_1'] + 1
    # elif attack1_df['features'][1] == Unrelated_2_feature: attack_2['Unrelated_2'] = attack_2['Unrelated_2'] + 1
    else: attack_2['Others'] = attack_2['Others'] + 1

    #For 3st position
    if attack1_df['features'][2] == biased_feature: attack_3['Biased'] = attack_3['Biased'] + 1
    elif attack1_df['features'][2] == Unrelated_1_feature: attack_3['Unrelated_1'] = attack_3['Unrelated_1'] + 1
    # elif attack1_df['features'][2] == Unrelated_2_feature: attack_3['Unrelated_2'] = attack_3['Unrelated_2'] + 1
    else: attack_3['Others'] = attack_3['Others'] + 1


# print ("Fidelity: {0:3.2}".format(adv_shap.fidelity(xtest[to_examine:to_examine+1])))


# In[44]:


#Normalized
keys = ['Biased','Unrelated_1','Unrelated_2','Others' ]
for x in keys:

    dict_biased_1[x] = dict_biased_1[x]/samples
    dict_biased_2[x] = dict_biased_2[x]/samples
    dict_biased_3[x] = dict_biased_3[x]/samples

    attack_1[x] = attack_1[x]/samples
    attack_2[x] = attack_2[x]/samples
    attack_3[x] = attack_3[x]/samples



# In[45]:


print(dict_biased_1 )
print(dict_biased_2 )
print(dict_biased_3 )
print('--------------')
print(attack_1 )
print(attack_2 )
print(attack_3 )



# In[46]:


# importing package
import matplotlib.pyplot as plt
import numpy as np
total = samples 
# create data
x = ['1st', '2nd', '3rd']
y1 = np.array([dict_biased_1['Biased'], dict_biased_2['Biased'], dict_biased_3['Biased']])
y2 = np.array([dict_biased_1['Unrelated_1'], dict_biased_2['Unrelated_1'], dict_biased_3['Unrelated_1']])
y3 = np.array([dict_biased_1['Others'], dict_biased_2['Others'], dict_biased_3['Others']])
y4 = np.array([dict_biased_1['Unrelated_2'], dict_biased_2['Unrelated_2'],dict_biased_3['Unrelated_2']])


# plot bars in stack manner
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.bar(x, y3, bottom=y1+y2, color='grey')
# plt.bar(x, y4, bottom=y1+y2+y3, color='g')
plt.xlabel("Features Importance Rank", fontsize = 18)
plt.ylabel("% Ocurrence", fontsize = 18)
# plt.legend(["Biased", "Unrelated", "Unrelated_2", "Others"])
plt.legend(["Biased", "Unrelated", "Others"], fontsize = 14)

plt.show()


# In[47]:


# importing package
import matplotlib.pyplot as plt
import numpy as np
total = samples 
# create data
x = ['1st', '2nd', '3rd']
y1 = np.array([attack_1['Biased'], attack_2['Biased'], attack_3['Biased']])
y2 = np.array([attack_1['Unrelated_1'], attack_2['Unrelated_1'], attack_3['Unrelated_1']])
y3 = np.array([attack_1['Others'], attack_2['Others'],attack_3['Others']])
# y4 = np.array([attack_1['Unrelated_2'], attack_2['Unrelated_2'], attack_3['Unrelated_2']])

# plot bars in stack manner
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.bar(x, y3, bottom=y1+y2, color='grey')
# plt.bar(x, y4, bottom=y1+y2+y3, color='g')
plt.xlabel("Features Importance Rank", fontsize = 18)
plt.ylabel("% Ocurrence", fontsize = 18)
# plt.legend(["Biased", "Unrelated_1", "Unrelated_2", "Others"])
plt.legend(["Biased", "Unrelated_1", "Others"], fontsize = 14)

plt.show()


