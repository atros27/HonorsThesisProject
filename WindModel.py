import math
import numpy as np
import pandas as pd
import time
from windrose import WindroseAxes

from Markov import Markov

class WindModel:
    def __init__(self, filename):
        start = time.time()
        table = pd.read_excel(filename)
        table.dropna(inplace=True)
        sd_data = table.to_numpy()
        sd_data = [[row[0], (row[1]+360) % 360] for row in sd_data] #Turn negative angles to positive
        sd_data = np.array(sd_data)
        self.initial_state = sd_data[0, :]
        self.current_state = self.initial_state
        #self.current_state = [1, 0]
        end = time.time()
        print("Sample Wind Data Imported. Time Elapsed:",math.floor(end-start),"seconds")

        speed_resolution = 10.0 #Meaning: 0.1 m/s resolution
        direction_resolution = 1.0 #Meaning: 1 degree resolution

        start = time.time()
        self.Speed_Markov = Markov(sd_data[:, 0], speed_resolution)
        self.Direction_Markov = Markov(sd_data[:, 1], direction_resolution)
        end = time.time()
        print("Markov Models Created. Time Elapsed:",math.floor(end-start),"seconds")

        #start = time.time()
        #self.test(sd_data)
        #end = time.time()
        #print("Sample Test Completed. Time Elapsed:",math.floor(end-start),"seconds")

    def test(self, data):
        self.current_state = self.initial_state
        synthetic = np.ndarray(dtype=float, shape=(len(data), 2) )
        synthetic[0, :] = self.current_state
        for i in range(1,len(data)):
            self.update()
            synthetic[i, :] = self.current_state

        #Provided Data Wind Rose
        ax = WindroseAxes.from_ax()
        ws_initial = data[:, 0]
        wd_initial = data[:, 1]
        ax.contourf(wd_initial, ws_initial, normed=True)
        ax.set_legend()

        #Synthetic Data Wind Rose
        ax1 = WindroseAxes.from_ax()
        ws_cont = synthetic[:, 0]
        wd_cont = synthetic[:, 1]
        ax1.contourf(wd_cont, ws_cont, normed=True)
        ax1.set_legend()

        self.current_state = self.initial_state #Reset the current state

    def update(self):
        new_speed = self.Speed_Markov.RandomWalk(self.current_state[0])
        new_direction = self.Direction_Markov.RandomWalk((self.current_state[1]))
        self.current_state = [new_speed, new_direction]
        return self.current_state
