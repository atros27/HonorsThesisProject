import math
from Material import Material
from Simulation import Simulation
from Strip import Strip
from TestSim1 import TestSim1
from TestSim2 import TestSim2
from TestSim3 import TestSim3
import time
from WindModel import WindModel

import matplotlib

def __main__():
    inch2meter = .0254
    start = time.time()
    #WindModel_1 = WindModel('C:/Users/atros27/Documents/WindData2.xlsx') #Laptop Version
    WindModel_1 = WindModel('C:/Users/docto/Downloads/WindData2.xlsx') #Desktop Version
    end = time.time()
    print("Wind Model Created. Time Elapsed:",math.floor(end-start),"seconds")
    aluminum7075 = Material("7075 T6 Aluminum",False,68e9,2.71e3,482.63e6,(68.95e6,6))
    brass = Material("Brass",False,100e9,8.73e3,250e6,(100e6,8)) #70Cu-30Zn Brass
    pzt = Material("PZT (Lead Zirconate Titanate",True,) #FINISH
    Simulation_1 = Simulation(WindModel_1)
    #TestSim3_1 = TestSim3()
    #print(WindModel_1.results.summary())
    #figure = WindModel_1.results.plot_forecast(10000)
    matplotlib.pyplot.show()
    #print(WindModel_1.synthetic)

__main__()
