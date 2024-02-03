#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd


# In[10]:


import pandas as pd

df=pd.read_csv('labeled_dataset_CPSS.csv')

df.head()


# In[11]:


def get_and_preprocess_compas(params):
    """Handle processing of COMPAS according to: https://github.com/propublica/compas-analysis
    
    Parameters
    ----------
    params : Params
    
    Returns
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """
    PROTECTED_CLASS = params.protected_class
    UNPROTECTED_CLASS = params.unprotected_class
    POSITIVE_OUTCOME = params.positive_outcome
    NEGATIVE_OUTCOME = params.negative_outcome
    fraction = 0.5
    frac_normal = 0.3

    compas_df = pd.read_csv('labeled_dataset_CPSS.csv')
    # compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
    #                           (compas_df['days_b_screening_arrest'] >= -30) &
    #                           (compas_df['is_recid'] != -1) &
    #                           (compas_df['c_charge_degree'] != "O") &
    #                           (compas_df['score_text'] != "NA")]

    # compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
    X = compas_df[['Formality', 'Location', 'Lane Alignment', 'Protocol', 'Plausibility','Frequency','Consistency', 'Speed', 'Correlation','Headway Time']]

    # if the person has a high score, give them the _negative_ model outcome
    y = np.array([NEGATIVE_OUTCOME if attackerType == 1 else POSITIVE_OUTCOME for attackerType in compas_df['Label']])
    sens = X['Formality']

    # assign African-American as the protected class
    X = pd.get_dummies(X)
    # sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
#   check X
    print(X)
    sensitive_attr = sens

    
#     df = pd.concat(frames,ignore_index=True)
    compas_df =compas_df.sample(frac = .05)

    # make sure everything is lining up
#     assert all(sens == (X['pos_x'] == PROTECTED_CLASS))
    cols = [col for col in compas_df]

    return X, y, cols


# In[13]:


compas_df = pd.read_csv('labeled_dataset_CPSS.csv')
# compas_df['attackerType']
# X['pos_x']


# In[14]:


"""
Train a bunch of models to create metric vs perturbation task score graphs.
"""
from adversarial_models import *
from utils import *
from get_data import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import shap
from copy import deepcopy

# Flip LIME flag to vary between lime shap
LIME = False

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
# X, y, cols = get_and_preprocess_compas_data(params)

X, y, cols = get_and_preprocess_compas(params)
# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])

features = [c for c in X]

race_indc = features.index('Formality')
# race_indc = features.index('race')

unrelated_indcs = features.index('unrelated_column_one')
unrelated_indcs1 = features.index('unrelated_column_two')

X = X.values

# categorical_features = [i for i,f in enumerate(features) if f not in ['age', 'length_of_stay', 'priors_count']]
categorical_features = [i for i,f in enumerate(features) if f not in ['Location', 'Lane Alignment', 'Protocol']]
###
## The models f and psi for COMPAS.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
#
# the biased model
class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])
    def predict_proba(self, X):
        return one_hot_encode(self.predict(X))
    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict_proba(self, X):
        return one_hot_encode(np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X]))
# the display model with two unrelated features
class innocuous_model_psi_two:
	def predict_proba(self, X):
		A = np.where(X[:,unrelated_indcs] > 0, params.positive_outcome, params.negative_outcome)
		B = np.where(X[:,unrelated_indcs1] > 0, params.positive_outcome, params.negative_outcome)
		preds = np.logical_xor(A, B).astype(int)
		return one_hot_encode(preds)
#
##
###

def experiment_main(X, y):
	from sklearn.metrics import f1_score as f1

	xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)

	ss = StandardScaler().fit(xtrain)
	xtrain = ss.transform(xtrain)
	xtest = ss.transform(xtest)
	rates, pct_first = [], []

	data_dict = {'trial':[], 'yhat':[], 'y':[], 'pct_occur_first':[], 'pct_occur_second':[], 'pct_occur_third':[]}

	trial = 0 
	for n_estimators in [1,2,4,8,16,32,64]:
		for max_depth in [1,2,4,8,None]:
			for min_samples_split in [2,4,8,16,32,64]:
				# Train the adversarial model for LIME with f and psi
				estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

				if LIME:
					adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).\
								train(xtrain, ytrain, estimator=estimator, feature_names=features, perturbation_multiplier=1)
					adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), discretize_continuous=False)
					
					formatted_explanations = []
					for i in range(xtest.shape[0]):
						exp = adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list()
						formatted_explanations.append(exp)
						if i >= 100: break

					adv_model = adv_lime

				else:
					background_distribution = shap.kmeans(xtrain,10)
					adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).\
										train(xtrain, ytrain, estimator=estimator, feature_names=features)
					
					adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
					explanations = adv_kerenel_explainer.shap_values(xtest[:100])
					
					formatted_explanations = []
					for exp in explanations:
						formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

					adv_model = adv_shap

				summary = experiment_summary(formatted_explanations, features)

				pct_occur = [0]
				for indc in [1,2,3]:
					found = False
					for tup in summary[indc]:
						if tup[0] == 'Formality':
							pct_occur.append(sum([pct_occur[-1], tup[1]]))
							found = True

					if not found:
						pct_occur.append(pct_occur[-1])

				pct_occur = pct_occur[1:]

				y = adv_model.ood_training_task_ability[0]
				yhat = adv_model.ood_training_task_ability[1]
				trial_df = np.array([trial for _ in range(y.shape[0])])

				data_dict['trial'] = np.concatenate((data_dict['trial'], trial_df))
				data_dict['yhat'] = np.concatenate((data_dict['yhat'], yhat))
				data_dict['y'] = np.concatenate((data_dict['y'], y))
				data_dict['pct_occur_first'] = np.concatenate((data_dict['pct_occur_first'], [pct_occur[0] for _ in range(y.shape[0])]))
				data_dict['pct_occur_second'] = np.concatenate((data_dict['pct_occur_second'], [pct_occur[1] for _ in range(y.shape[0])]))			
				data_dict['pct_occur_third'] = np.concatenate((data_dict['pct_occur_third'], [pct_occur[2] for _ in range(y.shape[0])]))			

				trial += 1

				if trial % 50 == 0:
					print ("Complete {}".format(trial+1))

	df = pd.DataFrame(data_dict)

	if LIME:
		limeorshap = 'lime'
	else:
		limeorshap = 'shap'
	df.to_csv('data/threshold_results_veremi_{}.csv'.format(limeorshap))


experiment_main(X, y)

