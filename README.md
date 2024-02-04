**EXAI-ADS: On Evaluating Black-box Explainable AI Methods for Enhancing Anomaly Detection in Autonomous Driving Systems**

**Abstract:**
The recent advancement in autonomous driving comes with the associated cybersecurity issue of compromising networks of autonomous vehicles (AVs). This security issue motivates leveraging AI models for detecting anomalies on these networks of AVs. In this context, the usage of explainable AI (XAI) for explaining the behavior of these AI models that detect anomalies in networks of AVs is crucial. In this work, we introduce a comprehensive framework to assess black-box XAI techniques in
the context of anomaly detection within autonomous vehicles. Our framework facilitates the examination of both global and local XAI methods, shedding light on the decisions made by XAI techniques that elucidate the behavior of AI models classifying anomalous autonomous vehicle behavior. By considering six evaluation metrics (descriptive accuracy, sparsity, stability, efficiency, robustness, and completeness), we conducted evaluations on two well-known black-box XAI techniques, SHAP and LIME. The evaluation process involved applying XAI techniques to identify primary features crucial for anomaly classification, followed by extensive experiments assessing SHAP and LIME across the specified six XAI evaluation metrics using two prevalent autonomous driving datasets, VeReMi and Sensor. This study marks a pivotal advancement in the deployment of black-box XAI methods for real-world anomaly detection in autonomous driving systems, contributing valuable insights into the strengths and limitations of current black-box XAI methods within this critical domain.

**Framework:**
![overview_XAI](https://github.com/Nazat28/EXAI_ADS/assets/101791995/9e194b8a-0f11-4659-8a36-e1a06e0227d1)

At a broad level, the pipeline will ingest raw autonomous driving datasets encompassing sensor, position, and speed data of an AV. These will be the main inputs into multiple black-box AI models to generate predictions (whether an AV has anomalous behavior or not). Subsequently, the predictions will be analyzed by XAI techniques (including SHAP and LIME) to produce model's top features and explanations. These  explanations will then be evaluated on the following six key dimensions: descriptive accuracy, sparsity, stability, efficiency, robustness, and completeness.

**Evaluation Results:**

1. Descriptive Accuracy:
The authors evaluated the descriptive accuracy of SHAP and LIME explanation methods on two datasets - VeReMi and Sensor - for autonomous driving models. Descriptive accuracy refers to the drop in model accuracy when removing the most influential features identified by the explanation methods. A greater drop implies the features are more explanatory. For VeReMi, SHAP outperformed LIME in terms of global explainability and causing a steeper decline in accuracy when removing top features. For Sensor data, SHAP and LIME performed similarly in decreasing model accuracy when removing top features. The DNN model did not show a consistent decline in accuracy on VeReMi data, indicating insufficient features. The security analysts should rely on the majority model behavior, not just DNN. Overall, SHAP exhibited greater descriptive accuracy than LIME on the VeReMi data, but both performed similarly on the Sensor data. Evaluating explanation methods on multiple datasets provides more robust assessment. Figure 4 and 5 shows the descriptive accuracy of SHAP and LIME on VeReMi and Sensor daataset respectively after removal of top fetaures.

![2024](https://github.com/Nazat28/EXAI_ADS/assets/101791995/cf360f3f-8182-44f1-b1f4-1cbcca05cf8f)

2. Sparsity:
Sparsity refers to how many input features are deemed highly relevant by the explanation methods. More sparse explanations are better as they identify a smaller subset of important features.  For the VeReMi dataset, SHAP exhibited higher sparsity than LIME across all 6 models tested.  SHAP showed more exponential growth in sparsity, meaning explanations concentrated on fewer top features.  The area under the curve (AUC) for sparsity was higher for SHAP than LIME for all models, indicating SHAP discarded more irrelevant features.  Higher sparsity for SHAP means analysts need to examine fewer features to understand decisions.  For LIME, lower sparsity meant more features were used in decisions, making it harder to interpret.  For the Sensor dataset, SHAP and LIME performed more similarly in terms of sparsity.  But SHAP still showed slightly higher AUC for sparsity for most models, indicating it was still more sparse.  Overall, SHAP demonstrated higher sparsity and identified a smaller set of influential features than LIME on both datasets. Figure 6 and 7 shows the sparsity of SHAP and LIME on VeReMi and Sensor daataset respectively.

![sparsity_veremi](https://github.com/Nazat28/EXAI_ADS/assets/101791995/37e0977c-df2d-450e-b8d9-64fb738f25d8)
![sparsity_sensor](https://github.com/Nazat28/EXAI_ADS/assets/101791995/c5512a9e-35d4-4b9e-9720-a70707f1befc)

3. Efficiency:
Efficiency refers to the time required to generate explanations by SHAP and LIME. For the multiclass VeReMi dataset, LIME was more efficient than SHAP overall across different sample sizes and models.  However, for lower sample sizes, SHAP was sometimes more efficient for certain models. For 50k samples, SHAP runtime could not be estimated for DNN and SVM models.  Similarly for binary class VeReMi data, LIME was generally more efficient except for 50k SVM samples. For the Sensor dataset, LIME again outperformed SHAP in efficiency for all models and sample sizes.  The Sensor data had up to 10k test samples due to dataset size limitations. Overall, LIME exhibited better computational efficiency and required less time to generate explanations than SHAP. The relative efficiency did depend somewhat on sample size and model complexity.  But in most cases, especially for large samples, LIME provided explanations faster than SHAP for autonomous driving datasets.  This suggests LIME has advantages for efficiently explaining models in time-critical applications like autonomous driving. Table 10 and table 11 is for VeReMi dataset (multicall and binary class) and table 12 is for the efficiency Sensor dataset respectively.

![eff_1](https://github.com/Nazat28/EXAI_ADS/assets/101791995/d8abc0ce-a32b-4936-b8dd-ec24aa13e3fb)
![eff_2](https://github.com/Nazat28/EXAI_ADS/assets/101791995/c2a27b1f-0c8e-41c2-ac7d-522f1dd770c5)
![eff_3](https://github.com/Nazat28/EXAI_ADS/assets/101791995/0960d66a-51f0-4e0b-a509-4fc00526d1b9)

4. Stability:
Stability refers to how consistent the explanations are across multiple runs of the XAI methods. For VeReMi dataset we checked the stability of top 3 features and for Sensor dataset we checked the stability of top 5 features. For global stability on VeReMi data, SHAP showed higher average intersection sizes than LIME, indicating more consistent top features. For Sensor data, SHAP outperformed LIME in global stability for 3 models, tied for 2, and did worse in 1 model. But overall SHAP was more globally stable. For local stability on a single sample from VeReMi, SHAP again showed higher stability than LIME, with more models having identical top features across runs. For Sensor data, SHAP beat LIME in local stability for 4 models, but LIME was more stable for 1 model. Overall SHAP was still more locally stable.  The local stability tests repeatedly explained the same sample, checking feature consistency. The global tests explained many samples, checking if top features remained the same. In both global and local tests, SHAP produced more consistent and stable feature attributions than LIME. This suggests SHAP explanations may be more reliable for autonomous driving systems. Unstable explanations that change across runs could mislead analysts monitoring these systems.  So SHAP's greater stability is advantageous for understandable and dependable explanations. Table 6, 7, 8 and 9 depicts the result from our stability experiment.

![stability](https://github.com/Nazat28/EXAI_ADS/assets/101791995/7a7a2938-2049-4a40-897b-a7686a766acb)

5. Robustness:

6. Completeness:
Completeness refers to whether the XAI methods can provide valid explanations for all input samples, including corner cases.  For local completeness on a VeReMi sample, LIME required fewer feature perturbations to change the class than SHAP, suggesting more complete local explanations. For a Sensor sample, LIME again needed fewer feature tweaks to alter the prediction, indicating more complete local explanations. For global completeness across thousands of VeReMi samples, both SHAP and LIME were incomplete, failing to modify some predictions when perturbing top features. But LIME performed slightly better, fully explaining 90% of benign and anomalous samples, versus 80% for SHAP.  For Sensor data, neither method was fully globally complete, but SHAP outperformed LIME for anomalous samples by 10%. 
The global tests reveal the prevalence of explanation failures across the problem space. To be fully comprehensive, XAI methods need to capture the model logic for all samples. 
Neither SHAP nor LIME achieved full completeness on the autonomous driving datasets. But LIME showed more complete local explanations for individual samples. 
While SHAP had a slight edge for global anomalous sample explanations on Sensor data. Overall, both methods require improvement to provide complete explanations in all cases. Figure 16 and 17 depicts the local completeness of SHAP and LIME on VeReMi dataset respectively. Local completeness of Sensor dataset for SHAP and LIME is showed in figure 18 and 19 respectively.

![complete_1](https://github.com/Nazat28/EXAI_ADS/assets/101791995/75859866-01b3-46e1-8f54-f4f1db74eec1)
![complete_2](https://github.com/Nazat28/EXAI_ADS/assets/101791995/e7d98749-a8be-449c-b179-7b1dae915eb8)


**Datasets:**
1. VeReMi dataset: Download link -> https://github.com/josephkamel/VeReMi-Dataset
2. Sensor dataset: Sensor dataset is provided in this repository named "Sensor_dataset.csv"




**To run experiments:**

1. For descriptive accuracy: First run the code till "initiating sparsity experiment". Then after getting the top features use the "df.drop" to drop the top features and run till this again. In this way descriptive accuracy without the top-k features can be got.
2. For sparsity: Run the whole experiment to get the sparsity values across different threshold.
3. For efficiency: Run the model first and then run cell named "Generating SHAP explanation"  and "Generating LIME explanation" for your desired sample number and note the time.
4. For stability: First Run the model first and then run cell named "Generating SHAP explanation"  and "Generating LIME explanation" then according to the features implement the "stability" equation in  the paper "EXAI-ADS: On Evaluating Black-box Explainable AI Methods for Enhancing Anomaly Detection in Autonomous Driving Systems" and calculate the stability.
5. For robustness experiment: Inside the VeReMi folder, first run the code: get_data_veremi.py to generate a csv file. Then run Sensitivity_SHAP and Sensitivity_LIME to generate the Robustness Sensitivity graph and Robustness_veremi.py for robustness bar graphs for LIME and SHAP respectively. Similar process should be followed for Sensor dataset.
6. For completeness experiment: To run completeness experiment for VeReMi dataset, go to VeReMi folder and run the experiment named Completeness_experiment_SHAP_veremi.py and Completeness_experiment_LIME_veremi.py for having the completeness. Same goes for Sensor  dataset.
   


**References:**

Robustness based on: https://github.com/dylan-slack/Fooling-LIME-SHAP/

Six metrics evaluated are based on: https://arxiv.org/abs/1906.02108

