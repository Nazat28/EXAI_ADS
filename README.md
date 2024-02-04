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
Robustness refers to whether adversarial attacks can fool the XAI methods into generating misleading explanations. For the occurrence percentage test on VeReMi data, SHAP and LIME performed similarly. Despite attacks, they still identified the biased feature in top positions, giving analysts a chance to detect attacks (figure 12). For Sensor data, LIME showed higher resilience, with the biased feature still appearing despite attacks (figure 13). For sensitivity testing (figure 14 and figure 15), LIME was more robust than SHAP up to a higher OOD-F1 threshold. Above those thresholds both methods became vulnerable. But LIME endured more attacks initially. For the robustness test on a sample VeReMi explanation (figure 8 and figure 10), the attack succeeded in fooling both SHAP and LIME. The biased feature was replaced by an unrelated synthetic feature as the top feature. For Sensor data (figure 9 and figure 11), SHAP was again fooled, but LIME was partially fooled, with a real feature replacing the biased one. Overall, while both methods exhibited some vulnerability to attacks, LIME showed more resilience. It required more sophisticated attacks and endured more attacks before explanations were fooled. So LIME appears more robust than SHAP, but improvements in adversarial resilience are still needed. Robust explainability is crucial for autonomous systems to avoid misleading analysts. 

![o_p](https://github.com/Nazat28/EXAI_ADS/assets/101791995/3ae963e3-909c-48a2-8a3d-529064bf46f3)
![sensitivity](https://github.com/Nazat28/EXAI_ADS/assets/101791995/85411e8d-27a7-45a1-8ed1-457e7b23a629)
![robust_SHAP_sample](https://github.com/Nazat28/EXAI_ADS/assets/101791995/a19cce18-a2ba-4178-a437-949d4debd943)
![robust_LIME_sample](https://github.com/Nazat28/EXAI_ADS/assets/101791995/35491ddd-ea4b-4f51-ace5-ef6f28f5343c)
![robust_LIME](https://github.com/Nazat28/EXAI_ADS/assets/101791995/98e9674e-74d0-4376-bc24-6e69fa8eaf64)




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



[1] F. Ahmad, A. Adnane, and V. N. Franqueira. A systematic approach for cyber
security in vehicular networks. Journal of Computer and Communications,
4(16):38–62, 2016.
[2] K. M. Ali Alheeti and K. McDonald-Maier. Intelligent intrusion detection in
external communication systems for autonomous vehicles. Systems Science &
Control Engineering, 6(1):48–56, 2018.
[3] A. A. Alsulami, Q. Abu Al-Haija, A. Alqahtani, and R. Alsini. Symmetrical
simulation scheme for anomaly detection in autonomous vehicles based on lstm
model. Symmetry, 14(7), 2022.
[4] O. Arreche, T. Guntur, and M. Abdallah. Xai-ids: Towards proposing an
explainable artificial intelligence framework for enhancing network intrusion
detection systems. Available at SSRN 4567885.
[5] S. Atakishiyev, M. Salameh, H. Yao, and R. Goebel. Explainable artificial
intelligence for autonomous driving: a comprehensive overview and field guide
for future research directions. arXiv preprint arXiv:2112.11561, 2021.
[6] S. A. Bagloee, M. Tavana, M. Asadi, and T. Oliver. Autonomous vehicles:
challenges, opportunities, and future implications for transportation policies.
Journal of Modern Transportation, 24(4):284–303, Dec. 2016.
[7] A. Bhattacharya. Understand the workings of shap and shapley values used in
explainable ai. https://shorturl.at/kloHThttps://shorturl.at/kloHT.
[8] D. Bogdoll, M. Nitsche, and J. M. Zöllner. Anomaly detection in autonomous
driving: A survey. In Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 4488–4499, 2022.
[9] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan,
G. Baldan, and O. Beijbom. nuscenes: A multimodal dataset for autonomous
driving, 2020.
[10] N. Capuano, G. Fenza, V. Loia, and C. Stanzione. Explainable artificial intelligence
in cybersecurity: A survey. IEEE Access, 10:93575–93600, 2022.
[11] Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu. Recurrent neural networks
for multivariate time series with missing values. Scientific reports, 8(1):6085, 2018.
[12] F. Chollet et al. Keras, 2015.
[13] A. Chowdhury, G. Karmakar, J. Kamruzzaman, A. Jolfaei, and R. Das. Attacks
on self-driving cars and their countermeasures: A survey. IEEE Access,
8:207308–207342, 2020.
[14] A. Das and P. Rad. Opportunities and challenges in explainable artificial
intelligence (xai): A survey. arXiv preprint arXiv:2006.11371, 2020.
[15] C. Dilmegani. Explainable ai (xai) in 2023: Guide to enterprise-ready ai.
https://research.aimultiple.com/xai/.
[16] P. Dixit, P. Bhattacharya, S. Tanwar, and R. Gupta. Anomaly detection in
autonomous electric vehicles using ai techniques: A comprehensive survey. Expert
Systems, 39(5):e12754, 2022.
[17] J. Dong, S. Chen, M. Miralinaghi, T. Chen, P. Li, and S. Labi. Why did the ai make
that decision? towards an explainable artificial intelligence (xai) for autonomous
driving systems. Transportation research part C: emerging technologies, 156:104358,
2023.
[18] R. Dwivedi, D. Dave, H. Naik, S. Singhal, R. Omer, P. Patel, B. Qian, Z. Wen, T. Shah,
G. Morgan, et al. Explainable ai (xai): Core ideas, techniques, and solutions. ACM
Computing Surveys, 55(9):1–33, 2023.
[19] S. Ercan, M. Ayaida, and N. Messai. Misbehavior detection for position falsification
attacks in vanets using machine learning. IEEE Access, 10:1893–1904, 2022.
[20] D. Feng, L. Rosenbaum, and K. Dietmayer. Towards safe autonomous driving:
Capture uncertainty in the deep neural network for lidar 3d vehicle detection.
In 2018 21st international conference on intelligent transportation systems (ITSC),
pages 3266–3273. IEEE, 2018.
[21] D. Fernández Llorca and E. Gómez. Trustworthy autonomous vehicles.
Publications Office of the European Union, Luxembourg„ EUR, 30942, 2021.
[22] C. Gao, G. Wang, W. Shi, Z. Wang, and Y. Chen. Autonomous driving security:
State of the art and challenges. IEEE Internet of Things Journal, 9(10):7572–7595,
2021.
[23] J. Geyer, Y. Kassahun, M. Mahmudi, X. Ricou, R. Durgesh, A. S. Chung,
L. Hauswald, V. H. Pham, M. Mühlegg, S. Dorn, T. Fernandez, M. Jänicke,
S. Mirashi, C. Savani, M. Sturm, O. Vorobiov, M. Oelker, S. Garreis, and
P. Schuberth. A2d2: Audi autonomous driving dataset, 2020.
[24] J. Grover, N. K. Prajapati, V. Laxmi, and M. S. Gaur. Machine learning approach
for multiple misbehavior detection in vanet. In Advances in Computing and
Communications: First International Conference, ACC 2011, Kochi, India, July 22-24,
2011, Proceedings, Part III 1, pages 644–653. Springer, 2011.
[25] R. Guidotti, A. Monreale, S. Ruggieri, F. Turini, F. Giannotti, and D. Pedreschi.
A survey of methods for explaining black box models. ACM computing surveys
(CSUR), 51(5):1–42, 2018.
[26] R. Gundu and M. Maleki. Securing can bus in connected and autonomous
vehicles using supervised machine learning approaches. In 2022 IEEE International
Conference on Electro Information Technology (eIT), pages 042–046, 2022.
[27] D. Holliday, S. Wilson, and S. Stumpf. User trust in intelligent systems: A journey
over time. In Proceedings of the 21st International Conference on Intelligent User
Interfaces, IUI ’16, page 164–168, New York, NY, USA, 2016. Association for
Computing Machinery.
[28] Z. Hu, J. Shen, S. Guo, X. Zhang, Z. Zhong, Q. A. Chen, and K. Li. Pass: A
system-driven evaluation platform for autonomous driving safety and security.
In NDSS Workshop on Automotive and Autonomous Vehicle Security (AutoSec),
2022.
[29] C. Hwang and T. Lee. E-sfd: Explainable sensor fault detection in the ics anomaly
detection system. IEEE Access, 9:140470–140486, 2021.
[30] B. W. Israelsen and N. R. Ahmed. “dave...i can assure you ...that it’s going to
be all right ...” a definition, case for, and survey of algorithmic assurances in
human-autonomy trust relationships. ACM Comput. Surv., 51(6), jan 2019.
[31] A. R. Javed, M. Usman, S. U. Rehman, M. U. Khan, and M. S. Haghighi.
Anomaly detection in automated vehicles using multistage attention-based
convolutional neural network. IEEE Transactions on Intelligent Transportation
Systems, 22(7):4291–4300, 2020.
[32] A. R. Javed, M. Usman, S. U. Rehman, M. U. Khan, and M. S. Haghighi.
Anomaly detection in automated vehicles using multistage attention-based
convolutional neural network. IEEE Transactions on Intelligent Transportation
Systems, 22(7):4291–4300, 2021.
[33] W. Li, P. Yi, Y. Wu, L. Pan, and J. Li. A new intrusion detection system based on
knn classification algorithm in wireless sensor network. Journal of Electrical and
Computer Engineering, 2014, 2014.
[34] Z. C. Lipton. The mythos of model interpretability, 2017.
[35] H. Lundberg, N. I. Mowla, S. F. Abedin, K. Thar, A. Mahmood, M. Gidlund,
and S. Raza. Experimental analysis of trustworthy in-vehicle intrusion
detection system using explainable artificial intelligence (xai). IEEE Access,
10:102831–102841, 2022.
[36] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz,
J. Himmelfarb, N. Bansal, and S.-I. Lee. From local explanations to global
understanding with explainable ai for trees. Nature machine intelligence,
2(1):56–67, 2020.
[37] A. S. Madhav and A. K. Tyagi. Explainable artificial intelligence (xai): connecting
artificial decision-making and human trust in autonomous vehicles. In
Proceedings of Third International Conference on Computing, Communications, and
Cyber-Security: IC4S 2021, pages 123–136. Springer, 2022.
[38] B. Mahbooba, M. Timilsina, R. Sahal, and M. Serrano. Explainable artificial
intelligence (xai) to enhance trust management in intrusion detection systems
using decision tree model. Complexity, 2021:1–11, 2021 
[39] H. Mankodiya, M. S. Obaidat, R. Gupta, and S. Tanwar. Xai-av: Explainable
artificial intelligence for trust management in autonomous vehicles. In 2021
International Conference on Communications, Computing, Cybersecurity, and
Informatics (CCCI), pages 1–5. IEEE, 2021.
[40] A. Martinho, N. Herber, M. Kroesen, and C. Chorus. Ethical issues in focus by
the autonomous vehicles industry. Transport Reviews, 41(5):556–577, 2021.
[41] C. Molnar. Interpretable Machine Learning. 2019. https://christophm.github.io/
interpretable-ml-book/.
[42] M. Müter, A. Groll, and F. C. Freiling. A structured approach to anomaly detection
for in-vehicle networks. In 2010 Sixth International Conference on Information
Assurance and Security, pages 92–98, 2010.
[43] S. Nazat and M. Abdallah. Anomaly detection framework for securing
next generation networks of platoons of autonomous vehicles in a
vehicle-to-everything system. In Proceedings of the 9th ACM Cyber-Physical
System Security Workshop, pages 24–35, 2023.
[44] C. I. Nwakanma, L. A. C. Ahakonye, J. N. Njoku, J. C. Odirichukwu, S. A.
Okolie, C. Uzondu, C. C. Ndubuisi Nweke, and D.-S. Kim. Explainable artificial
intelligence (xai) for intrusion detection and mitigation in intelligent connected
vehicles: A review. Applied Sciences, 13(3), 2023.
[45] S. Parkinson, P. Ward, K. Wilson, and J. Miller. Cyber threats facing autonomous
and connected vehicles: Future challenges. IEEE transactions on intelligent
transportation systems, 18(11):2898–2915, 2017.
[46] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel,
M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos,
D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine
learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.
[47] M. D. Pesé, J. W. Schauer, J. Li, and K. G. Shin. S2-can: Sufficiently secure controller
area network. In Annual Computer Security Applications Conference, ACSAC ’21,
page 425–438, New York, NY, USA, 2021. Association for Computing Machinery.
[48] S. B. Prathiba, G. Raja, S. Anbalagan, A. K. S, S. Gurumoorthy, and K. Dev. A hybrid
deep sensor anomaly detection for autonomous vehicles in 6g-v2x environment.
IEEE Transactions on Network Science and Engineering, 10(3):1246–1255, 2023.
[49] R. Raja and B. K. Sarkar. Chapter 12 - an entropy-based hybrid feature selection
approach for medical datasets. In P. Kumar, Y. Kumar, and M. A. Tawhid,
editors, Machine Learning, Big Data, and IoT for Medical Informatics, Intelligent
Data-Centric Systems, pages 201–214. Academic Press, 2021.
[50] M. T. Ribeiro, S. Singh, and C. Guestrin. " why should i trust you?" explaining the
predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international
conference on knowledge discovery and data mining, pages 1135–1144, 2016.
[51] S. Santini, A. Salvi, A. Valente, A. Pescapè, M. Segata, and R. Lo Cigno. A
consensus-based approach for platooning with inter-vehicular communications.
04 2015.
[52] A. Simkute, E. Luger, B. Jones, M. Evans, and R. Jones. Explainability for experts:
A design framework for making algorithms supporting expert decisions more
explainable. Journal of Responsible Technology, 7:100017, 2021.
[53] V. K. K. Sivaramakrishnan Rajendar. Sensor data based anomaly detection in
autonomous vehicles using modified convolutional neural network. Intelligent
Automation & Soft Computing, 32(2):859–875, 2022.
[54] D. Slack, S. Hilgard, E. Jia, S. Singh, and H. Lakkaraju. Fooling lime and shap:
Adversarial attacks on post hoc explanation methods. In Proceedings of the
AAAI/ACM Conference on AI, Ethics, and Society, pages 180–186, 2020.
[55] C. Tang, N. Luktarhan, and Y. Zhao. Saae-dnn: Deep learning method on intrusion
detection. Symmetry, 12(10):1695, 2020.
[56] P. Tao, Z. Sun, and Z. Sun. An improved intrusion detection algorithm based on
ga and svm. Ieee Access, 6:13624–13631, 2018.
[57] R. W. van der Heijden, T. Lukaseder, and F. Kargl. Veremi: A dataset for
comparable evaluation of misbehavior detection in vanets, 2018.
[58] F. van Wyk, Y. Wang, A. Khojandi, and N. Masoud. Real-time sensor anomaly
detection and identification in automated vehicles. IEEE Transactions on Intelligent
Transportation Systems, 21(3):1264–1276, 2020.
[59] A. Warnecke, D. Arp, C. Wressnegger, and K. Rieck. Evaluating explanation
methods for deep learning in security. In 2020 IEEE European Symposium on
Security and Privacy (EuroS&P), pages 158–174, 2020.
[60] S. Waskle, L. Parashar, and U. Singh. Intrusion detection system using pca with
random forest approach. In 2020 International Conference on Electronics and
Sustainable Communication Systems (ICESC), pages 803–808. IEEE, 2020.
[61] J. Xiong, R. Bi, M. Zhao, J. Guo, and Q. Yang. Edge-assisted privacy-preserving
raw data sharing framework for connected autonomous vehicles. IEEE Wireless
Communications, 27(3):24–30, 2020.
[62] A. Yulianto, P. Sukarno, and N. A. Suwastika. Improving adaboost-based intrusion
detection system (ids) performance on cic ids 2017 dataset. In Journal of Physics:
Conference Series, volume 1192, page 012018. IOP Publishing, 2019.
[63] A. Zekry, A. Sayed, M. Moussa, and M. Elhabiby. Anomaly detection using
iot sensor-assisted convlstm models for connected vehicles. In 2021 IEEE 93rd
Vehicular Technology Conference (VTC2021-Spring), pages 1–6. IEEE, 2021.
[64] Q. Zhang, J. Shen, M. Tan, Z. Zhou, Z. Li, Q. A. Chen, and H. Zhang. Play the
imitation game: Model extraction attack against autonomous driving localization.
In Proceedings of the 38th Annual Computer Security Applications Conference,
pages 56–70, 2022.
[65] Q. Zheng, Z. Wang, J. Zhou, and J. Lu. Shap-cam: Visual explanations for
convolutional neural networks based on shapley value. In Computer Vision–ECCV
2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings,
Part XII, pages 459–474. Springer, 2022.

