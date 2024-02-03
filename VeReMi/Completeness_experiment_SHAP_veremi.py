#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
dataset = pd.read_csv('modified_data.csv')

# Extract the features you want to normalize
features_to_normalize = ['pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the selected features
dataset[features_to_normalize] = scaler.fit_transform(dataset[features_to_normalize])

# Save the normalized dataset to a new CSV file
dataset.to_csv('normalized_data.csv', index=False)


# In[13]:


df=pd.read_csv('normalized_data.csv')

df.head()
data_df=df
data_df.head()


# In[14]:


# from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['pos_x', 'pos_y', 'pos_z', 'spd_x' ,'spd_y', 'spd_z']
X = data_df[feature_cols] # Features
y = data_df.attackerType # Target variable

from sklearn.model_selection import train_test_split


# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=100)

# # create a StandardScaler object and fit it to the training data
# scaler = StandardScaler()
# scaler.fit(X_train)

# # transform the training and testing data using the scaler object
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)




#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(criterion="gini",max_depth=50, n_estimators=100, min_samples_leaf=4)
clf.fit(X_train,y_train)


predictions=clf.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[15]:


X_test


# In[5]:


type(X_test)


# In[16]:


import numpy as np
import shap
import matplotlib.pyplot as plt 

#Define function to test sample with the waterfall plot
def waterfall_explanator(sample):
    # datapoint to explain
    explainer = shap.TreeExplainer(clf)
    prediction = clf.predict(sample)[0] 
#     prediction = label[clf.predict(sample)[0]] # Prediction of the sample
    #extract the index accordingly to prediction
#     print(prediction)
    index = prediction
#     index = label.index(prediction)
    
    #generating shap values explainer
    sv = explainer(sample) 
    bv = explainer.expected_value[index]
    
#     exp = shap.Explanation(sv[:,:,index], sv.base_values[:,index], sample, feature_names=test.columns.tolist())

    exp = shap.Explanation(sv[:,:,index], sv.base_values[:,index], sample, feature_names=X_test.columns.tolist())
    # generating plot
    shap.waterfall_plot(exp[0],max_display=10,show= None)
#     plt.savefig('RF_Shap_Waterfall.png')
#     plt.clf()

    feature_importance = pd.DataFrame({
        'row_id': sample.index.values.repeat(sample.shape[1]),
        'feature': sample.columns.to_list() * sample.shape[0],
        'feature_value': sample.values.flatten(),
        'base_value': bv,
        'shap_values': sv.values[:,:,index].flatten()
    
    })

    feature_importance['shap_values'] = abs(feature_importance['shap_values'])
    feature_importance.sort_values(by=['shap_values'], ascending=False,inplace=True)
    feature_importance.head()
    shap_val = feature_importance['shap_values'].tolist()
    feature_val = feature_importance['feature_value'].tolist()
    feature_name = feature_importance['feature'].tolist()
    
    return (prediction, shap_val,feature_val,feature_name)



# In[35]:


X_test2 = X_test.to_numpy()
# X_test


# In[17]:


X_test2


# In[18]:


X_test


# In[19]:


sample = X_test[0:1]


# In[20]:


sample


# In[21]:


u = waterfall_explanator(sample)


# In[22]:


u[0] # prediction 
u[1] # shap_val,
u[2] #feature_val
u[3] #,feature_name)


# In[23]:


u[0]


# In[24]:


sample['pos_x'] = 0.98


# In[25]:


sample['pos_y'] = 0.50


# In[26]:


sample


# In[27]:


u = waterfall_explanator(sample)


# In[28]:


u[0]


# In[29]:


sample['pos_y']= 0.11


# In[30]:


sample


# In[31]:


u = waterfall_explanator(sample)


# In[32]:


u[0]


# In[33]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
dataset = pd.read_csv('modified_data.csv')

# Separate the dataset based on attackerType
attacker_type_0 = dataset[dataset['attackerType'] == 0].drop('attackerType', axis=1)
attacker_type_1 = dataset[dataset['attackerType'] == 1].drop('attackerType', axis=1)

# Normalize using Min-Max scaling
scaler = MinMaxScaler()

# Fit and transform the scaler on attacker_type_0
attacker_type_0_normalized = pd.DataFrame(scaler.fit_transform(attacker_type_0), columns=attacker_type_0.columns)

# Fit and transform the scaler on attacker_type_1
attacker_type_1_normalized = pd.DataFrame(scaler.fit_transform(attacker_type_1), columns=attacker_type_1.columns)

# Display the normalized datasets
print("Normalized Attacker Type 0:")
print(attacker_type_0_normalized)

print("\nNormalized Attacker Type 1:")
print(attacker_type_1_normalized)


# In[34]:


def completeness_all(single_class_samples,number_samples, number_of_features_pertubation):
    Bucket = {
    '0.0': 0,
    '0.1':0,
    '0.2':0,
    '0.3':0,
    '0.4':0,
    '0.5':0,
    '0.6':0,
    '0.7':0,
    '0.8':0,
    '0.9':0,
    '1.0':0,

           }
    # Counter_chart = 0
    Counter_all_samples = 0
    counter_samples_changed_class = 0
    print('------------------------------------------------')
    print('Initiating Completeness Experiment')
    print('------------------------------------------------')
    for i in range(0,number_samples):
        #select sample
        try:
            sample = single_class_samples[i:i+1]
        except:
            break # break if there more samples requested than samples in the dataset
        # Explanate the original sample
        u = waterfall_explanator(sample)
        #select top 5 features from the original sample
        top_k_features = []
        top_k_features.append(u[3][0]) #append first feature
        break_condition = False
        for k in range(1,number_of_features_pertubation):
            for j in range(11):  # 11 steps to include 1.0 (0 to 10)
                if break_condition == True: break
                perturbation = j / 10.0  # Divide by 10 to get steps of 0.1
                temp_var = sample[top_k_features[k-1]]
                result = np.where((temp_var - perturbation) < 0, True, False)
                if result < 0: 
                    sample[top_k_features[k-1]] = 1 - perturbation
                else: sample[top_k_features[k-1]] = temp_var - perturbation
                # sample[top_k_features[k-1]] = perturbation
                v = waterfall_explanator(sample)
                if v[0] != u[0]: 
                    print(str(perturbation))
                    Bucket[str(perturbation)] += 1              
                    break_condition = True
                    counter_samples_changed_class += 1     
                    # Bucket[str(perturbation)] = counter_samples_changed_class              
                    break
                else: sample[top_k_features[k-1]] = abs(temp_var - 1) # set the sample feature value as the symetric opposite
            top_k_features.append(u[3][k]) #append second, third feature .. and so on
            if break_condition == True: break
        Counter_all_samples += 1
        progress  = 100*Counter_all_samples/number_samples
        if progress%10 == 0: print('Progress', progress ,'%')
    # print('Number of Normal samples that changed classification: ',counter_samples_changed_class)
    # print('Number of all samples analyzed: ',Counter_all_samples)
    # for key in Bucket:
    #     Bucket[key]=number_samples - Bucket[key]
    # Bucket['0.0'] = number_samples
    # for key in Bucket:
    #     Bucket[key]=Bucket[key]
    dict = Bucket
    temp = 0
    for k in dict:
        dict[k] = dict[k] + temp
        temp = dict[k]
    total = number_samples
    y_axis = []
    for k in dict:
        dict[k] = abs(dict[k] - total)
        y_axis.append(dict[k]/total)    
    return(counter_samples_changed_class,Counter_all_samples,y_axis)
        


# In[35]:


K_samples = 10
K_feat =  attacker_type_0_normalized.shape[1]
num_samples = K_samples
num_feat_pertubation = K_feat


# In[40]:


num_samples = K_samples
num_feat_pertubation = K_feat

p = completeness_all(attacker_type_0_normalized,num_samples,num_feat_pertubation)
print(p)
print('Number of benign samples that changed classification: ',p[0])
print('Number of all samples analyzed: ',p[1])
percentage = 100*p[0]/p[1]
print(percentage,'%','- samples are complete ')
y_axis_dos = p[2]
y_axis_dos[-1] = 0 


# In[52]:


y_axis=[1.0, 0.8, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0]


# In[41]:


K_samples = 10
K_feat =  attacker_type_1_normalized.shape[1]
num_samples = K_samples
num_feat_pertubation = K_feat


# In[42]:


num_samples = K_samples
num_feat_pertubation = K_feat

p = completeness_all(attacker_type_1_normalized,num_samples,num_feat_pertubation)
print(p)
print('Number of DoS samples that changed classification: ',p[0])
print('Number of all samples analyzed: ',p[1])
percentage = 100*p[0]/p[1]
print(percentage,'%','- samples are complete ')
y_axis_dos = p[2]
y_axis_dos[-1] = 0 


# In[44]:


x_axis = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]


# In[53]:


x_axis = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
plt.clf()

# Plot the first line
plt.plot(x_axis, y_axis_dos, label='Anomalous', color='blue', linestyle='--', marker='o')
# # Plot the third line
plt.plot(x_axis, y_axis, label='Benign', color='green', linestyle='--', marker='s')
# Enable grid lines (both major and minor grids)
plt.grid()

# Customize grid lines (optional)
# plt.grid()

# Add labels and a legend
plt.xlabel('Perturbations')
plt.ylabel('Samples remaining')
plt.legend()

