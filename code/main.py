import sys
import argparse
import logging
import multiprocessing
import multiprocessing as mp
import cohort_opt
import patient_opt
import inference
import numpy as np
parser = argparse.ArgumentParser("EtaNet")
parser.add_argument('--math_model', type = str, default = 'ExpDepFOTEResistance', help = 'Mathematical Models\n. Supported Models: LinearGrowth, LinearGrowth_FOS, ExponentialGrowth, ExponentialGrowth_FOS, LogisticGrowth, GompertzGrowth, FOTE, ExpDepFOTE, ExpDepFOTEResistance, NonLinearDrugExp')
parser.add_argument('--num_samples', type =int, default = 500, help = 'Number of Samples')
parser.add_argument('--std', type = float, default = 2.5, help = 'Standard Deviation of Gaussian Noise in Generated Data')
parser.add_argument('--growth_rate', type = float, default = 0.1, help = 'Growth Rate of Mathematical Models')
parser.add_argument('--death_rate', type = float, default = 0.04, help = 'Death Rate of Mathematical Models')
parser.add_argument('--initial_volume', type = float, default = 30, help = 'Initial Volume')
parser.add_argument('--VMax', type = float, default = 120, help = 'Max Volume (for Gompertz and Logistic Growth)')
parser.add_argument('--EMax', type = float, default = 30, help = 'EMax')
parser.add_argument('--drug_decay', type = float, default = 0.1, help = 'Exponential Decay Rate of Drugs')
parser.add_argument('--cohort_seed', type = int, default = 0, help = 'Seed for Generate Cohort Data')
parser.add_argument('--patient_seed', type = int, default = 1, help = 'Seed for Generate Patient Data')
parser.add_argument('--n_trials', type = int, default = 50, help = 'Numbers of Trials per Workers')
parser.add_argument('--num_workers', type = int, default = 15, help = 'Numbers of Workers')
parser.add_argument('--n_sims', type = int, default = 15, help = 'Numbers of Bayesian Inference Simulations')
parser.add_argument('--TMax', type = int, default = 50, help = 'Maximum Mornitoring Time for Generated Data')
parser.add_argument('--num_qubits', type = int, default = 1, help = 'Number of Qubits')
parser.add_argument('--num_qnn_layers', type = int, default = 8, help = 'Number of Quantum Layers')
parser.add_argument('--num_eta_functions', type = int, default = 4, help = 'Number of Quantum Kernels (Eta Functions)')
parser.add_argument('--spin', type = float, default = np.pi/2, help = 'Qubit Spin')
args = parser.parse_args()
print("Number of processors: ", mp.cpu_count())
print("Create Study: {}".format(args))
study_name = '{}_{}_{}_{}_{:.4f}_{}_{}_{:.4f}'.format(args.math_model,args.cohort_seed,args.patient_seed,args.num_samples,args.std,args.num_qnn_layers,args.num_eta_functions,args.spin)
print('Create Directory: {}'.format(study_name))
def main():
    cohort_opt.main(args.cohort_seed,args.n_trials,study_name,args.math_model,
            args.TMax,args.num_qubits,args.num_qnn_layers,args.num_eta_functions,args.spin,
            args.growth_rate,args.death_rate,args.initial_volume,args.VMax,args.EMax,args.drug_decay,
            args.std,args.num_samples)

    patient_opt.main(args.patient_seed,args.n_trials,study_name,args.math_model,
            args.TMax,args.num_qubits,args.num_qnn_layers,args.num_eta_functions,args.spin,
            args.growth_rate,args.death_rate,args.initial_volume,args.VMax,args.EMax,args.drug_decay,
            args.std,1)

    inference.main(args.patient_seed,args.n_trials,study_name,args.math_model,
            args.TMax,args.num_qubits,args.num_qnn_layers,args.num_eta_functions,args.spin,
            args.growth_rate,args.death_rate,args.initial_volume,args.VMax,args.EMax,args.drug_decay,
            args.std,1)

if __name__ == '__main__':
    import time
    jobs = []
    for i in range(args.num_workers):
        print('Start Worker {}'.format(i))
        p = multiprocessing.Process(target=main)
        jobs.append(p)
        p.start()
        if i == 0:
            time.sleep(25)
        else:
            time.sleep(5)
