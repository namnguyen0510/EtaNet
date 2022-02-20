import numpy as np
from scipy.integrate import odeint


class LinearGrowth():
	def __init__(self, Gr, Sr, V0, VMax):
		super(LinearGrowth, self).__init__()
		self.a = Gr
		self.V0 = V0

	def dVdt(self, V, t):
		dVdt = self.a
		return dVdt

	def forward(self, t, std = None):
		V = odeint(self.dVdt, self.V0, t).flatten()
		if std is not None:
			eps = np.random.normal(0, std, len(t))
			V += eps
		return V

	def generate(self, t, std, num_samples):
		output = []
		for i in range(num_samples):
			output.append(self.forward(t, std))
		output = np.array(output)
		return output

class LinearGrowthFOS():
	def __init__(self, Gr, Sr, V0, VMax):
		super(LinearGrowthFOS, self).__init__()
		self.a = Gr
		self.b = Sr
		self.V0 = V0

	def dVdt(self, V, t):
		dVdt = self.a-self.b*V
		return dVdt

	def forward(self, t, std = None):
		V = odeint(self.dVdt, self.V0, t).flatten()
		if std is not None:
			eps = np.random.normal(0, std, len(t))
			V += eps
		return V

	def generate(self, t, std, num_samples):
		output = []
		for i in range(num_samples):
			output.append(self.forward(t, std))
		output = np.array(output)
		return output

class ExponentialGrowth():
	def __init__(self, Gr, Sr, V0, VMax):
		super(ExponentialGrowth, self).__init__()
		self.a = Gr
		self.b = Sr
		self.V0 = V0

	def dVdt(self, V, t):
		dVdt = self.a*V
		return dVdt

	def forward(self, t, std = None):
		V = odeint(self.dVdt, self.V0, t).flatten()
		if std is not None:
			eps = np.random.normal(0, std, len(t))
			V += eps
		return V

	def generate(self, t, std, num_samples):
		output = []
		for i in range(num_samples):
			output.append(self.forward(t, std))
		output = np.array(output)
		return output

class ExponentialGrowthFOS():
	def __init__(self, Gr, Sr, V0, VMax):
		super(ExponentialGrowthFOS, self).__init__()
		self.a = Gr
		self.b = Sr
		self.V0 = V0

	def dVdt(self, V, t):
		dVdt = self.a*V-self.b*V
		return dVdt

	def forward(self, t, std = None):
		V = odeint(self.dVdt, self.V0, t).flatten()
		if std is not None:
			eps = np.random.normal(0, std, len(t))
			V += eps
		return V

	def generate(self, t, std, num_samples):
		output = []
		for i in range(num_samples):
			output.append(self.forward(t, std))
		output = np.array(output)
		return output

class LogisticGrowth():
	def __init__(self, Gr, Sr, V0, VMax):
		super(LogisticGrowth, self).__init__()
		self.a = Gr
		self.b = Sr
		self.V0 = V0
		self.VMax =VMax

	def dVdt(self, V, t):
		dVdt = self.a*V*(1-V/self.VMax)
		return dVdt

	def forward(self, t, std = None):
		V = odeint(self.dVdt, self.V0, t).flatten()
		if std is not None:
			eps = np.random.normal(0, std, len(t))
			V += eps
		return V

	def generate(self, t, std, num_samples):
		output = []
		for i in range(num_samples):
			output.append(self.forward(t, std))
		output = np.array(output)
		return output

class GompertzGrowth():
	def __init__(self, Gr, Sr, V0, VMax):
		super(GompertzGrowth, self).__init__()
		self.a = Gr
		self.b = Sr
		self.V0 = V0
		self.VMax =VMax

	def dVdt(self, V, t):
		dVdt = self.a*V*np.log(self.VMax/V)
		return dVdt

	def forward(self, t, std = None):
		V = odeint(self.dVdt, self.V0, t).flatten()
		if std is not None:
			eps = np.random.normal(0, std, len(t))
			V += eps
		return V

	def generate(self, t, std, num_samples):
		output = []
		for i in range(num_samples):
			output.append(self.forward(t, std))
		output = np.array(output)
		return output


class FirstOrderTreatmentEffect():
	def __init__(self, Gr, Sr, V0):
		super(FirstOrderTreatmentEffect, self).__init__()
		self.a = Gr
		self.b = Sr
		self.V0 = V0

	def dVdt(self, V, t):
		dVdt = self.a*V - self.b*V
		return dVdt

	def forward(self, t, std = None):
		V = odeint(self.dVdt, self.V0, t).flatten()
		if std is not None:
			eps = np.random.normal(0, std, len(t))
			V += eps
		return V

	def generate(self, t, std, num_samples):
		output = []
		for i in range(num_samples):
			output.append(self.forward(t, std))
		output = np.array(output)
		return output

class FirstOrderTreatmentEffect():
	def __init__(self, Gr, Sr, V0):
		super(FirstOrderTreatmentEffect, self).__init__()
		self.a = Gr
		self.b = Sr
		self.V0 = V0

	def dVdt(self, V, t):
		dVdt = self.a*V - self.b*V
		return dVdt

	def forward(self, t, std = None):
		V = odeint(self.dVdt, self.V0, t).flatten()
		if std is not None:
			eps = np.random.normal(0, std, len(t))
			V += eps
		return V

	def generate(self, t, std, num_samples):
		output = []
		for i in range(num_samples):
			output.append(self.forward(t, std))
		output = np.array(output)
		return output

class ExposureDependentFOTE():
	def __init__(self, Gr, Sr, V0, EpMax):
		super(ExposureDependentFOTE, self).__init__()
		self.a = Gr
		self.b = Sr
		self.V0 = V0
		self.EpMax = EpMax
		self.Ept50 = 10


	def dVdt(self, V, t):
		Exposure = self.EpMax*(t**0.5/(self.Ept50**0.5+t**0.5))
		dVdt = self.a*V-self.b*Exposure*V
		return dVdt

	def forward(self, t, std = None):
		V = odeint(self.dVdt, self.V0, t).flatten()
		if std is not None:
			eps = np.random.normal(0, std, len(t))
			V += eps
		return V

	def generate(self, t, std, num_samples):
		output = []
		for i in range(num_samples):
			output.append(self.forward(t, std))
		output = np.array(output)
		return output

class ExposureDependentFOTEResistance():
	def __init__(self, Gr, Sr, V0, EpMax, decay):
		super(ExposureDependentFOTEResistance, self).__init__()
		self.a = Gr
		self.b = Sr
		self.V0 = V0
		self.decay = decay
		self.EpMax = EpMax
		self.Ept50 = 10


	def dVdt(self, V, t):
		Exposure = self.EpMax*(t**0.5/(self.Ept50**0.5+t**0.5))
		dVdt = self.a*V - self.b*np.exp(-self.decay*t)*Exposure*V
		return dVdt

	def forward(self, t, std = None):
		V = odeint(self.dVdt, self.V0, t).flatten()
		if std is not None:
			eps = np.random.normal(0, std, len(t))
			V += eps
		return V

	def generate(self, t, std, num_samples):
		output = []
		for i in range(num_samples):
			output.append(self.forward(t, std))
		output = np.array(output)
		return output

class NonLinearDrugExposureEffect():
	def __init__(self, Gr, Sr, V0, EpMax, EMax, IC50):
		super(NonLinearDrugExposureEffect, self).__init__()
		self.a = Gr
		self.b = Sr
		self.V0 = V0
		self.EpMax = EpMax
		self.EMax = 2
		self.IC50 = 5
		self.Ept50 = 10


	def dVdt(self, V, t):
		Exposure = self.EpMax*(t**0.5/(self.Ept50**0.5+t**0.5))
		dVdt = self.a*V*(1- (self.EMax*Exposure)/(self.IC50+Exposure))
		return dVdt

	def forward(self, t, std = None):
		V = odeint(self.dVdt, self.V0, t).flatten()
		if std is not None:
			eps = np.random.normal(0, std, len(t))
			V += eps
		return V

	def generate(self, t, std, num_samples):
		output = []
		for i in range(num_samples):
			output.append(self.forward(t, std))
		output = np.array(output)
		return output
