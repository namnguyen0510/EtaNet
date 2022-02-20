import os
from models.simulators import *
from models import BayesEtaNet
import multiprocessing
import multiprocessing as mp
import optuna
import tqdm
import numpy as np
import scipy.stats
print("Number of processors: ", mp.cpu_count())
def main(seed,n_trials,study_name,math_model,
        TMax,n_qubits,basis,n_bases,spin,
        growth_rate,death_rate,V0,VMax,EMax,decay,
        std,num_samples,n_sims):
    try:
        os.mkdir(study_name)
    except:
        pass
    Th = 39
    study = Simulator(model = math_model, study_name = study_name,
            growth_rate = growth_rate,
            death_rate = death_rate,
            V0 = V0,
            VMax = VMax,
            decay = decay, seed = seed)
    trainset = study.generate(TMax = TMax, std = std, num_samples = num_samples)
    V_0 = trainset
    model = BayesEtaNet.fit(df = trainset, TMax=TMax, EMax = EMax,
                        n_qubits=n_qubits, basis=basis, n_bases=n_bases, spin=spin,
                        study_name = study_name, seed = seed)
    time = np.linspace(0, spin, TMax)
    y = np.zeros(time.shape)
    for i in tqdm.tqdm(range(n_sims)):
        y+=model.inference()
    y = y/n_sims
    trainset = trainset.to_numpy()[0]
    trainset = trainset/trainset[0]-1
    train_loss = np.mean(np.mean((y[:Th+1]-trainset[:Th+1])**2))
    test_loss = np.mean(np.mean((y[Th:]-trainset[Th:])**2))
    t = np.linspace(0, TMax, TMax)
    fig, ax = model.fig, model.ax
    ax.plot(time[:Th+1]/(np.pi/2), y[:Th+1], alpha = 0.5, linewidth = 1, color = 'blue', label = 'Train Loss: {:4f}'.format(train_loss))
    ax.plot(time[Th:]/(np.pi/2), y[Th:], alpha = 0.5, linewidth = 1, color = 'green', label = 'Test Loss: {:4f}'.format(test_loss))
    for i in range(len(trainset)):
        ax.scatter(t[:Th +1]/TMax, trainset[:Th+1], alpha = 0.5, s = 10, color = 'purple')
        ax.scatter(t[Th:]/TMax, trainset[Th:], alpha = 0.5, s = 10, color = 'green')
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return h
    ci = mean_confidence_interval(y,0.95)
    ax.legend(loc = 'lower right', fontsize = 'xx-small')
    plt.fill_between(time[:Th+1]/(np.pi/2), (y-ci)[:Th+1], (y+ci)[:Th+1], color='blue', alpha=0.2)
    plt.fill_between(time[Th:]/(np.pi/2), (y-ci)[Th:], (y+ci)[Th:], color='green', alpha=0.2)
    plt.savefig('{}/patient_final.jpg'.format(study_name), dpi = 300)
    df = pd.DataFrame([])
    df['V_true_norm'] = trainset
    df['V_pred'] = y
    df.to_csv('{}/patient_pred.csv'.format(study_name), index = False)
