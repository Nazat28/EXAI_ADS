#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
dataset = pd.read_csv('labeled_dataset_CPSS.csv')

# Extract the features you want to normalize
features_to_normalize = ['Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the selected features
dataset[features_to_normalize] = scaler.fit_transform(dataset[features_to_normalize])

# Save the normalized dataset to a new CSV file
dataset.to_csv('normalized_data_CPSS.csv', index=False)


# In[39]:


df=pd.read_csv('normalized_data_CPSS.csv')

df.head()
data_df=df
data_df.head()


# In[105]:


# from sklearn.preprocessing import StandardScaler

# define the feature columns and target variable
feature_cols = ['Formality' ,'Location' ,'Frequency' ,'Speed', 'Correlation' ,'Lane Alignment','Headway Time' ,'Protocol', 'Plausibility' ,'Consistency']
X = data_df[feature_cols] # Features
y = data_df.Label # Target variable

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


# In[106]:


X_test


# In[107]:


sample = X_test[0:1]


# In[108]:


sample


# In[109]:


class_names = ['Benign', 'Anomalous']


# In[110]:


X_test2


# In[111]:


feature_names= list(X_test.columns.values)
feature_names


# In[112]:


X_test2


# In[113]:


feature_cols = ['Formality' ,'Location' ,'Frequency' ,'Speed', 'Correlation' ,'Lane Alignment','Headway Time' ,'Protocol', 'Plausibility' ,'Consistency']
#Define function to test sample with the waterfall plot
def lime_explanator(sample):
    # print(sample)
    sample_df = sample
    sample = sample.to_numpy()
    sample = sample[0]

#     explainer = lime.lime_tabular.LimeTabularExplainer(X_test2, feature_names= list(X_test.columns.values) , class_names=class_names , discretize_continuous=True)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_test2, feature_names= list(X_test.columns.values) , class_names=class_names, discretize_continuous=True)

    #creating dict 
#     feat_list = req_cols[:-1]
    # print(feat_list)

    c = 0
    

    num_columns = df.shape[1] - 1
    feature_name = feature_cols
    feature_name.sort()
    feature_val = []
    feature_val_abs = []
    samples = 1
    # position = y_labels.index(rf.predict(Dos_sample2))
    
    position =  np.argmax(clf.predict_proba(((sample_df))))
    prediction = clf.predict(sample_df)[0] 
    # print(len(y_labels))
    # print(rf.predict(Dos_sample2))


    # sample = Dos_sample
    # sample = Normal_sample
    # sample = PS_sample


    for i in range(0,num_columns): 
        feature_val.append(0)
        feature_val_abs.append(0)

    # for i in range(0,samples):

    # i = sample
        # exp = explainer.explain_instance(test[i], rf.predict_proba)
        
    exp = explainer.explain_instance(sample, clf.predict_proba, num_features=num_columns, top_labels=1)
    exp.show_in_notebook(show_table=True, show_all=True)
    
    lime_list = exp.as_list(position)
    lime_list.sort()
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
        
        #---------------------------------------------------
    
    lime_list.sort()
    # print(lime_list)
    # for j in range (0,num_columns): feature_val[j]+= abs(lime_list[j][1])
    for j in range (0,num_columns):feature_val_abs[j] = abs(lime_list[j][1])
    for j in range (0,num_columns):feature_val[j] = lime_list[j][1]
    c = c + 1 
    # print ('progress',100*(c/samples),'%')

    # Define the number you want to divide by
    # divider = samples

    # Use a list comprehension to divide all elements by the same number
    # feature_val = [x / divider for x in feature_val]

    # for item1, item2 in zip(feature_name, feature_val):
    #     print(item1, item2)


    # Use zip to combine the two lists, sort based on list1, and then unzip them
    zipped_lists = list(zip(feature_name, feature_val,feature_val_abs))
    zipped_lists.sort(key=lambda x: x[2],reverse=True)

    # Convert the sorted result back into separate lists
    sorted_list1, sorted_list2,sorted_list3 = [list(x) for x in zip(*zipped_lists)]
    feature_name = sorted_list1
    lime_val = sorted_list2
    # print(sorted_list1)
    # print(sorted_list2)
    # print(sorted_list3)
    return (prediction, lime_val,feature_name)



# In[114]:


sample_df = sample


# In[115]:


position =  np.argmax(clf.predict_proba(((sample_df))))
position


# In[116]:


type(X_test)


# In[117]:


X_test2 = X_test.to_numpy()
# X_test


# In[118]:


X_test2


# In[119]:


import lime
import lime.lime_tabular
import numpy as np


# In[120]:


sample


# In[121]:


y=lime_explanator(sample)


# In[19]:


y[0] # prediction 
y[1] # lime_val,
y[2] #feature_name


# In[52]:


y[0]


# In[53]:


sample = X_test[0:1]


# In[54]:


sample


# In[55]:


sample['Formality']=0.8


# In[56]:


sample


# In[57]:


y=lime_explanator(sample)


# In[58]:


y[0]


# In[59]:


sample['Location']=0.8


# In[60]:


sample


# In[61]:


y=lime_explanator(sample)


# In[62]:


y[0]


# In[63]:


import pandas as pd

# Load the dataset
dataset = pd.read_csv('labeled_dataset_CPSS.csv')

# Separate the dataset based on attackerType
attacker_type_0 = dataset[dataset['Label'] == 0]
attacker_type_1 = dataset[dataset['Label'] == 1]

# Display the separated datasets
print("'Label' 0:")
print(attacker_type_0)

print("\n 'Label' 1:")
print(attacker_type_1)


# In[64]:


import pandas as pd

# Load the dataset
dataset = pd.read_csv('labeled_dataset_CPSS.csv')

# Separate the dataset based on attackerType
attacker_type_0 = dataset[dataset['Label'] == 0].drop('Label', axis=1)
attacker_type_1 = dataset[dataset['Label'] == 1].drop('Label', axis=1)

# Display the separated datasets
print("'Label' 0:")
print(attacker_type_0)

print("\n 'Label' 1:")
print(attacker_type_1)


# In[65]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
dataset = pd.read_csv('labeled_dataset_CPSS.csv')

# Separate the dataset based on attackerType
attacker_type_0 = dataset[dataset['Label'] == 0].drop('Label', axis=1)
attacker_type_1 = dataset[dataset['Label'] == 1].drop('Label', axis=1)


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


# In[122]:


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
        u = lime_explanator(sample)
        #select top 5 features from the original sample
        top_k_features = []
        top_k_features.append(u[2][0]) #append first feature
        break_condition = False
        for k in range(1,number_of_features_pertubation):
            for j in range(11):  # 11 steps to include 1.0 (0 to 10)
                if break_condition == True: break
                perturbation = j / 10.0  # Divide by 10 to get steps of 0.1
                try:temp_var = sample[top_k_features[k-1]]
                except: None
                result = np.where((temp_var - perturbation) < 0, True, False)
                if result < 0: 
                    sample[top_k_features[k-1]] = 1 - perturbation
                else: 
                    try:sample[top_k_features[k-1]] = temp_var - perturbation
                    except: None

                # sample[top_k_features[k-1]] = perturbation
                v = lime_explanator(sample)
                if v[0] != u[0]:
                    Bucket[str(perturbation)] += 1 
                    break_condition = True
                    counter_samples_changed_class += 1                   
                    break
                else: sample[top_k_features[k-1]] = abs(temp_var - 1) # set the sample feature value as the symetric opposite
            top_k_features.append(u[2][k]) #append second, third feature .. and so on
            if break_condition == True: break
        Counter_all_samples += 1
        progress  = 100*Counter_all_samples/number_samples
        if progress%10 == 0: print('Progress', progress ,'%')
        # if progress >= 1: break
    # print('Number of Normal samples that changed classification: ',counter_samples_changed_class)
    # print('Number of all samples analyzed: ',Counter_all_samples)
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
        


# In[123]:


attacker_type_0_normalized.head()


# In[124]:


K_samples = 10
K_feat =  attacker_type_0_normalized.shape[1]
num_samples = K_samples
num_feat_pertubation = K_feat


# In[125]:


num_samples = K_samples
num_feat_pertubation = K_feat

p = completeness_all(attacker_type_0_normalized,num_samples,num_feat_pertubation)
print(p)
print('Number of benign (0) samples that changed classification: ',p[0])
print('Number of all samples analyzed: ',p[1])
percentage = 100*p[0]/p[1]
print(percentage,'%','- samples are complete ')
y_axis_benign = p[2]
y_axis_benign[-1] = 0 


# In[84]:


num_samples = K_samples
num_feat_pertubation = K_feat

p = completeness_all(attacker_type_1_normalized,num_samples,num_feat_pertubation)
print(p)
print('Number of anomalouss (1) samples that changed classification: ',p[0])
print('Number of all samples analyzed: ',p[1])
percentage = 100*p[0]/p[1]
print(percentage,'%','- samples are complete ')
y_axis_anomalous = p[2]
y_axis_anomalous[-1] = 0 

