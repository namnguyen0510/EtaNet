import numpy as np
import pandas as pd
from lib.models.math_model import *
import matplotlib.pyplot as plt

class Simulator():
    def __init__(self, model, study_name , growth_rate, death_rate, V0, VMax=None, decay=None, EMax=None, IC50=None, seed = None):
        super(Simulator, self).__init__()
        np.random.seed(seed)
        self.study_name = study_name
        self.seed = seed
        self.model = model
        self.alpha = growth_rate
        self.beta = death_rate
        self.V0 = V0
        self.VMax = VMax
        self.decay = decay
        self.EMax = EMax
        self.IC50 = IC50
        self.simulator = self.get_simulator()

    def generate(self, TMax, std, num_samples):
        np.random.seed(self.seed)
        t = np.linspace(0,TMax,TMax)
        df = self.simulator.generate(t, std, num_samples)
        df = pd.DataFrame(df)
        df.to_csv('{}/{}.csv'.format(self.study_name,self.model), index = False, sep = '\t')
        for i in range(len(df)):
            plt.scatter(t,df.iloc[i,:])
        plt.savefig('{}/{}.jpg'.format(self.study_name,self.model), dpi = 300)
        return df

    def get_simulator(self):
        if self.model == "LinearGrowth":
            simulators = LinearGrowth(Gr = self.alpha, Sr = self.beta, V0=self.V0)
            return simulators
        if self.model == "LinearGrowth_FOS":
            simulators = LinearGrowthFOS(Gr = self.alpha, Sr = self.beta, V0=self.V0)
            return simulators
        if self.model == "ExponentialGrowth":
            simulators = ExponentialGrowth(Gr = self.alpha, Sr = self.beta, V0=self.V0, VMax = self.VMax)
            return simulators
        if self.model == "ExponentialGrowth_FOS":
            simulators = ExponentialGrowthFOS(Gr = self.alpha, Sr = self.beta, V0=self.V0, VMax = self.VMax)
            return simulators
        if self.model == "LogisticGrowth":
            simulators = LogisticGrowth(Gr = self.alpha, Sr = self.beta, V0=self.V0)
            return simulators
        if self.model == "GompertzGrowth":
            simulators = GompertzGrowth(Gr = self.alpha, Sr = self.beta, V0=self.V0, VMax = self.VMax)
            return simulators
        if self.model == "FOTE":
            simulators = FirstOrderTreatmentEffect(Gr = self.alpha, Sr = self.beta, V0=self.V0)
            return simulators
        if self.model == "ExpDepFOTE":
            simulators = ExposureDependentFOTE(Gr = self.alpha, Sr = self.beta, V0=self.V0, EpMax = self.V0)
            return simulators
        if self.model == "ExpDepFOTEResistance":
            simulators = ExposureDependentFOTEResistance(Gr = self.alpha, Sr = self.beta, V0=self.V0, EpMax = self.V0, decay = self.decay)
            return simulators
        if self.model == "NonLinearDrugExp":
            simulators = NonLinearDrugExposureEffect(Gr = self.alpha, Sr = self.beta, V0=self.V0, EpMax = self.V0, EMax = self.EMax, IC50 = self.IC50)
            return simulators
        else:
            print("Not Implemented Simulator!")
