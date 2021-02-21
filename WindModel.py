import math
import matplotlib
import numpy as np
import openpyxl
import pandas as pd
import scipy as sp
from statsmodels.tsa.api import VAR
from windrose import WindroseAxes


class WindModel:
    def __init__(self, filename):
        table = pd.read_excel(filename)
        table.dropna(inplace=True)
        uv_data = table.to_numpy()

        ax = WindroseAxes.from_ax()
        ws_initial = [math.sqrt(pow(row[0],2) + pow(row[1],2)) for row in uv_data]
        wd_initial = [math.atan2(row[1], row[0])/math.pi*180 for row in uv_data]
        print(ws_initial[-10:])
        print(wd_initial[-10:])
        ax.bar(wd_initial, ws_initial, normed=True)
        ax.set_legend()

        #v_data = np.array([sheet.cell_values(r, 2) for r in range(1, sheet.nrows)])
        self.model = VAR(table)
        self.results = self.model.fit(1)
        self.synthetic = self.results.forecast(uv_data[0:], 30)
        print(uv_data[-30:])
        print(self.synthetic)

        ax1 = WindroseAxes.from_ax()
        ws_cont = [math.sqrt(pow(row[0],2) + pow(row[1],2)) for row in self.synthetic]
        wd_cont = [math.atan2(row[1], row[0])/math.pi*180 for row in self.synthetic]
        print(ws_cont[:])
        print(wd_cont[:])
        ax1.contourf(wd_cont, ws_cont)
        ax1.set_legend()