# EtaNet
Implementations of EtaNet - a translational Quantum Machine Intelligence for Modeling Tumor Dynamics in Oncology. Pre-print: (Update Soon)
## Abstract
Quantifying the dynamics of tumor burden reveals useful information about cancer evolution concerning treatment effects and drug resistance, which play a crucial role in advancing model-informed drug developments (MIDD) towards personalized medicine and precision oncology. The emergence of Quantum Machine Intelligence offers unparalleled insights into tumor dynamics via a quantum mechanics perspective. This paper introduces a novel hybrid quantum-classical neural architecture named <img src="https://render.githubusercontent.com/render/math?math=\eta">-Net that enables quantifying quantum dynamics of tumor burden concerning treatment effects. We evaluate our proposed neural solution on two major use cases, including cohort-specific and patient-specific modeling. In silico numerical results show a high capacity and expressivity of <img src="https://render.githubusercontent.com/render/math?math=\eta">-Net to the quantified biological problem. Moreover, the close connection to representation learning - the foundation for successes of modern AI, enables efficient transferability of empirical knowledge from relevant cohorts to targeted patients. Finally, we leverage Bayesian optimization to quantify the epistemic uncertainty of model predictions, paving the way for <img src="https://render.githubusercontent.com/render/math?math=\eta">-Net towards reliable AI in decision-making for clinical usages.
## Neural Architecture of EtaNet
![plot](./figures/EtaNet.png)
## Model Capacity
![plot](./figures/model_capacity.png)
## Experiment History
![Alt Text](https://github.com/namnguyen0510/EtaNet/blob/main/figures/bayes_infer_top_5.gif)
## Experiment Results
### Cohort-specific Modeling
<a href="url"><img src="https://github.com/namnguyen0510/EtaNet/blob/main/figures/cohort_spec.png" align="left" height="800" ></a>
### Patient-specific Modeling
<a href="url"><img src="https://github.com/namnguyen0510/EtaNet/blob/main/figures/pat_spec.png" align="left" height="800" ></a>
## Some Illustrations
<a href="url"><img src="https://github.com/namnguyen0510/EtaNet/blob/main/figures/arts/0_random_99_4_4.jpg" align="left" height="300" ></a>
<a href="url"><img src="https://github.com/namnguyen0510/EtaNet/blob/main/figures/arts/1_random_99_4_4.jpg" align="left" height="300" ></a>
<a href="url"><img src="https://github.com/namnguyen0510/EtaNet/blob/main/figures/arts/2_random_99_4_4.jpg" align="left" height="300" ></a>
<a href="url"><img src="https://github.com/namnguyen0510/EtaNet/blob/main/figures/arts/random_49_4_4.jpg" align="left" height="300" ></a>

## Requirement
```
python >= 3.6.12, pennylane == 0.13.0, multiprocessing == 2.6.2.1, optuna == 2.4.0, scipy == 1.5.4
```
## Code usage
Training Model End-to-end
```
cd code
python main.py
```
