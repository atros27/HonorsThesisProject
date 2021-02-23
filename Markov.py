import math
import numpy as np
from scipy.stats import rv_discrete

class Markov:
    def __init__(self, data, resolution):
        #Initialize Variables
        self.resolution = resolution
        num_rows_and_columns = math.floor(max(data) * resolution) + 1
        self.histogram = np.ndarray(dtype=float, shape=(num_rows_and_columns, num_rows_and_columns))
        self.TransitionMatrix = []  # Markov Chain Object (list of random variables)
        dummy_random_variable = rv_discrete(values=(np.arange(num_rows_and_columns)/resolution, [1/num_rows_and_columns] * num_rows_and_columns))
        #This dummy RV is currently set to a uniform distribution of all values. Functionally, it selects a purely random seed.

        #Fill Out Histogram with Provided Data
        current_index = math.floor(data[0] * resolution)
        for i in range(1, len(data)):
            previous_index = current_index
            current_index = math.floor(data[i] * resolution)
            self.histogram[previous_index, current_index] += 1

        #Normalize Histogram Rows & Identify Unused Rows
        empty_rows = []
        for i in range(len(self.histogram)):
            row_sum = sum(self.histogram[i, :])
            if row_sum <= 0 or np.isnan(row_sum): #Empty or Nonsense Rows
                empty_rows.append(i)
            else:
                for j in range(len(self.histogram[0])):
                    self.histogram[i, j] /= row_sum  # Normalize all speed rows to create PDF rows

        #Transform histogram rows to RV variables
        row_range = np.arange(num_rows_and_columns)  #Tuple of all rows
        for i in range(len(self.histogram)):
            if i not in empty_rows:
                self.TransitionMatrix.append(rv_discrete(values=(row_range, self.histogram[i, :])))
            else:
                self.TransitionMatrix.append(dummy_random_variable)  # Dummy random variable for unused rows

    def RandomWalk(self, seed):
        index = math.floor(seed * self.resolution) #Transform seed value into a row index
        new_index = self.TransitionMatrix[index].rvs()
        return new_index / self.resolution