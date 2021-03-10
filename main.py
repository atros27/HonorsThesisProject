from Simulation import Simulation
from TestSim1 import TestSim1
from TestSim2 import TestSim2
from TestSim3 import TestSim3
from WindModel import WindModel

import matplotlib

def __main__():
    #WindModel_1 = WindModel('C:/Users/atros27/Documents/WindData2.xlsx')
    #Simulation_1 = Simulation()
    TestSim2_1 = TestSim2()
    #print(WindModel_1.results.summary())
    #figure = WindModel_1.results.plot_forecast(10000)
    matplotlib.pyplot.show()
    #print(WindModel_1.synthetic)

__main__()
