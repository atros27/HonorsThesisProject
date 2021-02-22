from WindModel import WindModel

import matplotlib

def __main__():
    WindModel_1 = WindModel('C:/Users/atros27/Documents/WindData2 - Copy.xlsx')
    print(WindModel_1.results.summary())
    #figure = WindModel_1.results.plot_forecast(10000)
    matplotlib.pyplot.show()
    #print(WindModel_1.synthetic)

__main__()
