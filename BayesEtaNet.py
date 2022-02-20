import pennylane as qml
from pennylane import numpy as np
import torch
from lib.models.primitives import *
import matplotlib
import matplotlib.pyplot as plt
import tqdm as tqdm
import pandas as pd
import torch.nn as nn
from lib.models.model import *
import optuna
from optuna.samplers import TPESampler
import joblib
import time
import sys
import random

class fit():
	def __init__(self, df, TMax, EMax, n_qubits, basis, n_bases,spin,
				 study_name = None, seed = None):
		super(fit, self)
		self.seed = seed
		self.study_name = study_name
		if study_name is None:
			print("Study name Required!")
		self.data = df
		self.n_qubits = n_qubits
		self.basis = basis
		self.n_bases = n_bases
		self.TMax = TMax
		self.t = np.linspace(0, TMax, TMax)
		self.fig, self.ax = plt.subplots(1,1, figsize = (5,5))
		self.spin = spin

	def inference(self):
		np.random.seed = np.random.randint(10000, size=1)
		cohort_study  = optuna.load_study(study_name="CohortOpt_{}".format(self.study_name),
								storage= os.path.join('sqlite:///{}/CohortOpt_{}.db'.format(self.study_name,self.study_name)))
		patient_study  = optuna.load_study(study_name="PatientOpt_{}".format(self.study_name),
								storage= os.path.join('sqlite:///{}/PatientOpt_{}.db'.format(self.study_name,self.study_name)))
		cohort_params = cohort_study.best_params
		pat_params = patient_study.best_params
		Q_w = [cohort_params['theta_{:04d}'.format(i)] for i in range(self.basis)]
		q_w = [[np.random.uniform(0, Q_w[i], size=((i+1)* self.n_qubits), requires_grad=True) for i in range(self.basis)] for _ in range(self.n_bases)]
		alpha = [cohort_params['alpha_{:04d}'.format(i)] for i in range(self.basis)]
		W = [pat_params['W_{:04d}'.format(i)] for i in range(self.n_bases)]
		t = np.linspace(0, self.spin, self.TMax)
		df = self.data
		idx = 0
		Th = 39
		V_true = df.iloc[idx,:].to_numpy()
		V_true = V_true/V_true[0]-1
		model = Regressors(t, alpha, W, q_w, self.n_qubits, self.basis, self.n_bases, self.spin)
		y = model.forward(t, q_w)
		y = np.array(y)
		#loss = np.sqrt(np.mean(np.mean((V_pred-V_true)**2)))
		train_loss = np.mean(np.mean((y[:Th+1]-V_true[:Th+1])**2))
		test_loss = np.mean(np.mean((y[Th:]-V_true[Th:])**2))
		# CAN UPDATE CONTINUOUS COLOUR SCALE FOR VISUALIZING HARDENING MODEL
		self.ax.plot(t[:Th+1]/(self.spin), y[:Th+1], alpha = 0.3, linewidth = 0.7, color = 'blue')
		self.ax.plot(t[Th:]/(self.spin), y[Th:], alpha = 0.3, linewidth = 0.7, color = 'green')
		'''
		for i in range(len(V_true)):
			self.ax.scatter(self.t[:Th +1]/self.TMax, V_true[:Th+1], alpha = 0.2, s = 5, color = 'purple')
			#self.ax.vlines(self.t[Th]/self.TMax, ymin=-1, ymax= 0., color='black', lw=0.2, ls ="--")
			self.ax.scatter(self.t[Th:]/self.TMax, V_true[Th:], alpha = 0.2, s = 5, color = 'green')
		'''
		plt.savefig('{}/patient_infer.jpg'.format(self.study_name), dpi = 300)
		print('Train Loss: {:4f}'.format(train_loss))
		print('Test  Loss: {:4f}'.format(test_loss))
		return y


	def fit_patient(self, trial):
		loaded_study = optuna.create_study(study_name="CohortOpt_{}".format(self.study_name), sampler=TPESampler(seed=self.seed, n_startup_trials=30, n_ei_candidates=100),
					storage= os.path.join('sqlite:///{}/CohortOpt_{}.db'.format(self.study_name,self.study_name)),load_if_exists=True)
		best_params = loaded_study.best_params
		Q_w = [best_params['theta_{:04d}'.format(i)] for i in range(self.basis)]
		q_w = [[np.random.uniform(0, Q_w[i], size=((i+1)* self.n_qubits), requires_grad=True) for i in range(self.basis)] for _ in range(self.n_bases)]
		alpha = [best_params['alpha_{:04d}'.format(i)] for i in range(self.basis)]
		best_params = np.array(best_params)
		# Loading and Preprocessing data
		t = np.linspace(0, self.spin, self.TMax)
		df = self.data
		idx = 0
		Th = 39
		V_true = df.iloc[idx,:].to_numpy()
		V_true = V_true/V_true[0]-1
		# Initialize Parameters
		W = [trial.suggest_uniform('W_{:04d}'.format(i), 0, 1) for i in range(self.n_bases)]
		# Fitting
		model = Regressors(t, alpha, W, q_w, self.n_qubits, self.basis, self.n_bases, self.spin)
		y = model.forward(t, q_w)
		y = np.array(y)
		#loss = np.sqrt(np.mean(np.mean((V_pred-V_true)**2)))
		train_loss = np.mean(np.mean((y[:Th+1]-V_true[:Th+1])**2))
		test_loss = np.mean(np.mean((y[Th:]-V_true[Th:])**2))
		# CAN UPDATE CONTINUOUS COLOUR SCALE FOR VISUALIZING HARDENING MODEL
		self.ax.plot(t[:Th+1]/(self.spin), y[:Th+1], alpha = 0.5, linewidth = 1, color = plt.cm.Blues(trial.number*2+100))
		self.ax.plot(t[Th:]/(self.spin), y[Th:], alpha = 0.5, linewidth = 1, color = plt.cm.Greens(trial.number*2+100))
		for i in range(len(V_true)):
			self.ax.scatter(self.t[:Th +1]/self.TMax, V_true[:Th+1], alpha = 0.5, s = 10, color = 'purple')
			self.ax.scatter(self.t[Th:]/self.TMax, V_true[Th:], alpha = 0.5, s = 10, color = 'green')
		plt.savefig('{}/patient_opt.jpg'.format(self.study_name, trial), dpi = 300)
		print('Train Loss: {:4f}'.format(train_loss))
		print('Test  Loss: {:4f}'.format(test_loss))
		return train_loss

	def fit_cohort(self, trial):
		# Loading and Preprocessing data
		t = np.linspace(0, self.spin, self.TMax)
		df = self.data
		score = []
		V_true = df/df.iloc[:,0].mean()-1
		# Initialize Parameters
		Q_w = [trial.suggest_uniform('theta_{:04d}'.format(i), -2*np.pi, 2*np.pi) for i in range(self.basis)]
		q_w = [[np.random.uniform(0, Q_w[i], size=((i+1)* self.n_qubits), requires_grad=True) for i in range(self.basis)] for _ in range(self.n_bases)]
		alpha = [trial.suggest_uniform('alpha_{:04d}'.format(i), -10e-1, 10e-1) for i in range(self.basis)]
		W = [trial.suggest_uniform('W_{:04d}'.format(i), 0, 1) for i in range(self.n_bases)]
		# Fitting
		model = Regressors(t, alpha, W, q_w, self.n_qubits, self.basis, self.n_bases, self.spin)
		y = model.forward(t, q_w)
		y = np.array(y)
		V_pred = np.repeat(y.reshape(1,-1), len(df), axis = 0)
		loss = np.mean(np.mean((V_pred-V_true)**2))
		# CAN UPDATE CONTINUOUS COLOUR SCALE FOR VISUALIZING HARDENING MODEL
		self.ax.plot(t/(self.spin), y, alpha = 0.5, linewidth = 2, color = plt.cm.Blues(trial.number*2+150))
		for i in range(len(V_true)):
			normalize = matplotlib.colors.Normalize(vmin=0, vmax=len(V_true))
			self.ax.scatter(self.t/self.TMax, V_true.iloc[i], alpha = 0.5, s = 2, norm=normalize)
		plt.savefig('{}/cohort_opt.jpg'.format(self.study_name, trial), dpi = 300)
		return loss


	def cohort_opt(self, n_trials):
		optuna.logging.set_verbosity(3)
		study = optuna.create_study(study_name="CohortOpt_{}".format(self.study_name), sampler=TPESampler(seed=self.seed, n_startup_trials=30, n_ei_candidates=100),
					storage= os.path.join('sqlite:///{}/CohortOpt_{}.db'.format(self.study_name,self.study_name)),load_if_exists=True)
		print(f"Sampler is {study.sampler.__class__.__name__}")
		study.optimize(self.fit_cohort, n_trials=n_trials)
		joblib.dump(study, "{}/CohortOpt.pkl".format(self.study_name))

	def patient_opt(self, n_trials):
		optuna.logging.set_verbosity(3)
		study = optuna.create_study(study_name="PatientOpt_{}".format(self.study_name), sampler=TPESampler(seed=self.seed, n_startup_trials=30, n_ei_candidates=100),
					storage= os.path.join('sqlite:///{}/PatientOpt_{}.db'.format(self.study_name,self.study_name)),load_if_exists=True)
		print(f"Sampler is {study.sampler.__class__.__name__}")
		study.optimize(self.fit_patient, n_trials=n_trials)
		joblib.dump(study, "{}/PatientOpt.pkl".format(self.study_name))
