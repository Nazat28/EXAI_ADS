**EXAI-ADS: On Evaluating Black-box Explainable AI Methods for Enhancing Anomaly Detection in Autonomous Driving Systems**

**Abstract:**
The recent advancement in autonomous driving comes with the associated cybersecurity issue of compromising networks of autonomous vehicles (AVs). This security issue motivates leveraging AI models for detecting anomalies on these networks of AVs. In this context, the usage of explainable AI (XAI) for explaining the behavior of these AI models that detect anomalies in networks of AVs is crucial. In this work, we introduce a comprehensive framework to assess black-box XAI techniques in
the context of anomaly detection within autonomous vehicles. Our framework facilitates the examination of both global and local XAI methods, shedding light on the decisions made by XAI techniques that elucidate the behavior of AI models classifying anomalous autonomous vehicle behavior. By considering six evaluation metrics (descriptive accuracy, sparsity, stability, efficiency, robustness, and completeness), we conducted evaluations on two well-known black-box XAI techniques, SHAP and LIME. The evaluation process involved applying XAI techniques to identify primary features crucial for anomaly classification, followed by extensive experiments assessing SHAP and LIME across the specified six XAI evaluation metrics using two prevalent autonomous driving datasets, VeReMi and Sensor. This study marks a pivotal advancement in the deployment of black-box XAI methods for real-world anomaly detection in autonomous driving systems, contributing valuable insights into the strengths and limitations of current black-box XAI methods within this critical domain.

**Framework:**
![overview_XAI](https://github.com/Nazat28/EXAI_ADS/assets/101791995/9e194b8a-0f11-4659-8a36-e1a06e0227d1)

At a broad level, the pipeline will ingest raw autonomous driving datasets encompassing sensor, position, and speed data of an AV. These will be the main inputs into multiple black-box AI models to generate predictions (whether an AV has anomalous behavior or not). Subsequently, the predictions will be analyzed by XAI techniques (including SHAP and LIME) to produce model's top features and explanations. These  explanations will then be evaluated on the following six key dimensions: descriptive accuracy, sparsity, stability, efficiency, robustness, and completeness.
