import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from network import *
from fn_PINN import *
from fn_physics import *



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sns.set_theme()
torch.manual_seed(31)
np.random.seed(7)

Tenv = 25
T0 = 100
R = 0.005
times = np.linspace(0, 1000, 500)
eq = functools.partial(cooling_law, Tenv=Tenv, T0=T0, R=R)
temps = eq(times)

# Make training data
t = np.linspace(0, 400, 15)
T = eq(t) +  2 * np.random.randn(15)

print('-'*50)

ph_loss = functools.partial(physics_loss, R=R, Tenv=Tenv)


net1 = MainNet(1,1, loss2=ph_loss, epochs=50000, loss2_weight=1, lr=5e-6).to(DEVICE)
losses1 = net1.fit(t, T)
preds1 = net1.predict(times)

print('-'*50)
ph_loss = functools.partial(physics_loss_discovery, Tenv=Tenv)

net2 = NetWithParameter(1, 1, loss2=ph_loss, loss2_weight=1, epochs=50000, lr= 5e-6).to(DEVICE)
losses2 = net2.fit(t, T)
preds2 = net2.predict(times)

print('estimated cooling rate: %.4f' %net2.r)
print('exact cooling rate: %.4f' %R)
print('-'*50)


plt.figure()
plt.plot(times,preds1,'r',label='net1')
plt.plot(times,preds2,'b',label='net2 [no cooling rate given]')
plt.plot(times,temps,'k--',label='exact solution')
plt.plot(t,T,'sg',label='datapoints')
plt.legend(loc='center left',bbox_to_anchor=(1,.5))
plt.show()

print('Done :)')