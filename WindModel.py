import math
import matplotlib
import numpy as np
import openpyxl
import pandas as pd
import scipy as sp
from statsmodels.tsa.api import VARMAX
from windrose import WindroseAxes


class WindModel:
    def __init__(self, filename):
        table = pd.read_excel(filename)
        table.dropna(inplace=True)
        sd_data = table.to_numpy()

        ax = WindroseAxes.from_ax()
        ws_initial = sd_data[:, 0]
        wd_initial = sd_data[:, 1]
        #sd_data = [[i, j] for i in wd_initial for j in ws_initial]
        print(ws_initial)
        print(wd_initial)
        ax.contourf(wd_initial, ws_initial, normed=True)
        ax.set_legend()

        #v_data = np.array([sheet.cell_values(r, 2) for r in range(1, sheet.nrows)])
        self.model = VARMAX(sd_data, order=(1,1))
        self.results = self.model.fit(maxiter=500, disp=False)
        #self.coefficients = self.results.params
        #self.mu = self.coefficients[0, :]
        #self.k = self.coefficients[1:3, :]
        #self.mu = np.zeros((2, 1))
        #self.k = np.array([[1, 0.0], [0.0, 1.0]])
        #self.sigma = np.array([[1.0, 0], [0, 1.0]])
        #print(self.mu)
        #print(self.k)
        #print(self.sigma)
        print(sd_data[-3:, :])
        self.synthetic = self.results.forecast(10**3)
        #self.current_vector = np.array([5.0, 5.0])
        #self.synthetic = self.OU(30)
        #print(uv_data[-30:])
        #print(self.synthetic)

        ax1 = WindroseAxes.from_ax()
        #ws_cont = [math.sqrt(pow(row[0],2) + pow(row[1],2)) for row in self.synthetic]
        #wd_cont = [math.atan2(row[1], row[0])/math.pi*180 for row in self.synthetic]
        wd_cont = self.synthetic[:, 1]
        ws_cont = self.synthetic[:, 0]
        print(ws_cont[0:200])
        print(wd_cont[0:200])
        print(ws_cont[-50:])
        print(wd_cont[-50:])
        ax1.contourf(wd_cont, ws_cont, normed=True)
        ax1.set_legend()

        ax2 = WindroseAxes.from_ax()
        ws_test = np.random.random(500)*20
        wd_test = np.random.random(500)*360
        ax2.contourf(wd_test, ws_test)
        ax2.set_legend()

    def OU(self, steps): #propagates OU-process from fit params for arbitrary number of steps
        ans = np.zeros((steps, 2))
        for i in range(steps):
            #print(np.random.normal(0, 1, size=(2, 1)))
            dummy = self.mu - self.k @ self.current_vector + self.sigma @ np.random.normal(0, 1, size=(2,1))
            #print(dummy)
            duv_dt = dummy[:, 0]
            self.current_vector += duv_dt
            ans[i, :] = self.current_vector
        return ans
