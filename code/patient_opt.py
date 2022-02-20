import os
from models.simulators import *
from models import BayesEtaNet
import multiprocessing
import multiprocessing as mp
import optuna

def main(seed,n_trials,study_name,math_model,
        TMax,n_qubits,basis,n_bases,spin,
        growth_rate,death_rate,V0,VMax,EMax,decay,
        std,num_samples):
    try:
        os.mkdir(study_name)
    except:
        pass
    study = Simulator(model = math_model, study_name = study_name,
            growth_rate = growth_rate,
            death_rate = death_rate,
            V0 = V0,
            VMax = VMax,
            decay = decay, seed = seed)
    trainset = study.generate(TMax = TMax, std = std, num_samples = num_samples)
    model = BayesEtaNet.fit(df = trainset, TMax=TMax, EMax = EMax,
                        n_qubits=n_qubits, basis=basis, n_bases=n_bases, spin=spin,
                        study_name = study_name, seed = seed)
    model.patient_opt(n_trials)
