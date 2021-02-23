import math
import matplotlib
import numpy as np
import openpyxl
import pandas as pd
from scipy.stats import rv_discrete
from statsmodels.tsa.api import VARMAX
from windrose import WindroseAxes


class WindModel:
    def __init__(self, filename):
        table = pd.read_excel(filename)
        table.dropna(inplace=True)
        sd_data = table.to_numpy()
        sd_data = [[row[0], (row[1]+360) % 360] for row in sd_data] #Turn negative angles to positive
        sd_data = np.array(sd_data)

        speed_resolution = 10.0 #Meaning: 0.1 m/s resolution
        direction_resolution = 1.0 #Meaning: 1 degree resolution
        intermediate_1 = sd_data[:, 0]
        intermediate_2 = max(intermediate_1)*speed_resolution
        num_speed_rows_and_columns = math.floor(intermediate_2)+1
        num_direction_rows_and_columns = math.floor(max(sd_data[:, 1]) * direction_resolution) + 1
        self.S_matrix = np.ndarray(dtype=float, shape=(num_speed_rows_and_columns, num_speed_rows_and_columns)) #Speed Transition Matrix
        self.D_matrix = np.ndarray(dtype=float, shape=(num_direction_rows_and_columns, num_direction_rows_and_columns)) #Direction Transition Matrix
        self.S_Markov = [] #Speed Markov Chain (list of random variables)
        self.D_Markov = [] #Direction Markov Chain (list of random variables)

        current_speed_index = math.floor(sd_data[0, 0] * speed_resolution)
        current_direction_index = math.floor(sd_data[0, 1] * direction_resolution)
        for i in range(1, len(sd_data)):
            previous_speed_index = current_speed_index
            previous_direction_index = current_direction_index

            current_speed_index = math.floor(sd_data[i, 0] * speed_resolution)
            current_direction_index = math.floor(sd_data[i, 1] * direction_resolution)

            self.S_matrix[previous_speed_index, current_speed_index] += 1
            self.D_matrix[previous_direction_index, current_direction_index] += 1 #Setup Transition histograms

        speed_empty_rows = []
        direction_empty_rows = []
        for i in range(len(self.S_matrix)):
            speed_row_sum = sum(self.S_matrix[i, :])
            #print(speed_row_sum)
            if speed_row_sum <= 0 or np.isnan(speed_row_sum):
                speed_empty_rows.append(i)
            else:
                for j in range(len(self.S_matrix[0])):
                    self.S_matrix[i, j] /= speed_row_sum #Normalize all speed rows to create PDF rows
        for i in range(len(self.D_matrix)):
            direction_row_sum = sum(self.D_matrix[i, :])
            #print(direction_row_sum)
            if direction_row_sum <= 0 or np.isnan(direction_row_sum):
                direction_empty_rows.append(i)
            else:
                for j in range(len(self.D_matrix[0])):
                    self.D_matrix[i, j] /= direction_row_sum #Normalize all direction rows to create PDF rows
        #print(self.S_matrix[0])
        #print(self.S_matrix[1])
        speed_range = np.arange(num_speed_rows_and_columns) # "Name" of every row in m/s
        direction_range = np.arange(num_direction_rows_and_columns)
        for i in range(len(self.S_matrix)):
            if i not in speed_empty_rows:
                #print(self.S_matrix[i, :])
                self.S_Markov.append(rv_discrete(values=(speed_range, self.S_matrix[i, :]) ) )
            else:
                self.S_Markov.append(rv_discrete(values=((5,10), (0.5, 0.5)))) #Dummy random variable for unused rows
        for i in range(len(self.D_matrix)):
            if i not in direction_empty_rows:
                #print(self.D_matrix[i, :])
                self.D_Markov.append(rv_discrete(values=(direction_range, self.D_matrix[i, :]) ) )
            else:
                self.D_Markov.append(rv_discrete(values=((90,180), (0.5, 0.5))))
        #print(self.S_Markov[0].pmf(1))

        ax = WindroseAxes.from_ax()
        ws_initial = sd_data[:, 0]
        wd_initial = sd_data[:, 1]
        #sd_data = [[i, j] for i in wd_initial for j in ws_initial]
        #print(ws_initial[0:200])
        #print(wd_initial[0:200])
        #print(ws_initial[-50:])
        #print(wd_initial[-50:])
        ax.contourf(wd_initial, ws_initial, normed=True)
        ax.set_legend()

        #v_data = np.array([sheet.cell_values(r, 2) for r in range(1, sheet.nrows)])
        #self.model = VARMAX(sd_data, order=(1,1))
        #self.results = self.model.fit(maxiter=500, disp=False)
        #self.coefficients = self.results.params
        #self.mu = self.coefficients[0, :]
        #self.k = self.coefficients[1:3, :]
        #self.mu = np.zeros((2, 1))
        #self.k = np.array([[1, 0.0], [0.0, 1.0]])
        #self.sigma = np.array([[1.0, 0], [0, 1.0]])
        #print(self.mu)
        #print(self.k)
        #print(self.sigma)
        #print(sd_data[-3:, :])
        #self.synthetic = self.results.forecast(10**3)
        #self.current_vector = np.array([5.0, 5.0])
        #self.synthetic = self.OU(30)
        #print(uv_data[-30:])
        #print(self.synthetic)

        synthetic_size = 10**6
        self.synthetic = np.ndarray(dtype=float, shape=(synthetic_size, 2) )
        self.synthetic[0] = sd_data[0, :]
        next_speed_index = math.floor(self.synthetic[0, 0] * speed_resolution) + 1
        next_direction_index = math.floor(self.synthetic[0, 0] * direction_resolution) + 1
        for i in range(1, len(self.synthetic)):
            current_speed_index = next_speed_index
            current_direction_index = next_direction_index
            next_speed_index = self.S_Markov[current_speed_index].rvs()
            next_direction_index = self.D_Markov[current_direction_index].rvs()
            #print(next_speed_index, next_direction_index)
            self.synthetic[i] = [next_speed_index/speed_resolution, next_direction_index/direction_resolution] #Generate n-step Markov Walk

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

        #ax2 = WindroseAxes.from_ax()
        #ws_test = np.random.random(500)*20
        #wd_test = np.random.random(500)*360
        #ax2.contourf(wd_test, ws_test)
        #ax2.set_legend()

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
