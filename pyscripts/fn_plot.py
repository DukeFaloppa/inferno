import matplotlib.pyplot as plt

def PlotData(times,temps,t,T):
    plt.figure()
    plt.plot(times, temps)
    plt.plot(t, T, 'o')
    plt.legend(['Equation', 'Training data'])
    plt.ylabel('Temperature (C)')
    plt.xlabel('Time (s)')
    plt.show()


def PlotLosses(losses):
    plt.figure()
    plt.plot(losses)
    plt.yscale('log')
    plt.show()

def PlotPredictions(times,temps,preds,t,T):
    plt.figure()
    plt.plot(times, temps, alpha=0.8)
    plt.plot(t, T, 'o')
    plt.plot(times, preds, alpha=0.8)
    plt.legend(labels=['Equation','Training data', 'PINN'])
    plt.ylabel('Temperature (C)')
    plt.xlabel('Time (s)')
    plt.show()