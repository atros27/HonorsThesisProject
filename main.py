import math
from Simulation import Simulation
from TestSim1 import TestSim1
from TestSim2 import TestSim2
from TestSim3 import TestSim3
import time
from WindModel import WindModel

import matplotlib

def __main__():
    start = time.time()
    #WindModel_1 = WindModel('C:/Users/atros27/Documents/WindData2.xlsx')
    end = time.time()
    print("Wind Model Created. Time Elapsed:",math.floor(end-start),"seconds")
    Simulation_1 = Simulation()#WindModel_1)
    #TestSim3_1 = TestSim3()
    #print(WindModel_1.results.summary())
    #figure = WindModel_1.results.plot_forecast(10000)
    matplotlib.pyplot.show()
    #print(WindModel_1.synthetic)

__main__()
