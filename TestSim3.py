import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp

#TESTSIM3 : Biaxial Bending-only Rayleigh-Ritz 2DOF ODE

class TestSim3:
    def __init__(self):
        self.E = 68e6 #Aluminum Young's Modulus Todo: Replace with Material Class that holds these constants
        self.G = 25e6 #Aluminum Shear Modulus
        self.rho = 2.71e3 #kg/m^3
        self.air_density = 1.2 #kg/m^3
        self.c_t = .01
        self.c_b = .01

        self.r = .01 #1 cm radius for now Todo: Determine Geometric Settings, possibly customize
        self.b = .01
        self.e = .1
        self.L = .25 #Note: Tip bodies and shaft-beam have identical length

        self.U = 10
        self.delta = 120/180*math.pi

        self.J = 0.5*math.pi*math.pow(self.r, 4)
        self.I = 0.25*math.pi*math.pow(self.r, 4)
        self.A = math.pi*math.pow(self.r, 2)

        self.iterations = 0

        aoa_tuple = [-180, -170, -160, -150, -145, -140, -130, -120, -110, -100, -95, -90, -85, -80, -70, -60, -50, -40,
                     -30, -25, -20, -10, 0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100, 110, 120, 130, 140,
                     145, 150, 160, 170, 180]
        aoa_tuple = [i * math.pi / 180 for i in aoa_tuple]
        CL = [0, -.4, -.8, -1.1, -1.2, -1.1, -.4, 0, .4, 1, 1.125, .95, .8, .7, .4, 0, -.4, -.8, -1.05, -1.2, -.9, -.3,
              0, .3, .9, 1.2, 1.05, .8, .4, 0, -.4, -.7, -.8, -.95, -1.125, -1, -.4, 0, .4, 1.1, 1.2, 1.1, .8, .4, 0]
        CD = [1.75, 1.7, 1.55, 1.1, .9, .9, .95, .9, .9, .9, 1, 1.15, 1.35, 1.5, 1.65, 1.7, 1.65, 1.5, 1.2, .95, .9,
              .95, .9, .95, .9, .95, 1.2, 1.5, 1.65, 1.7, 1.65, 1.5, 1.35, 1.15, 1, .9, .9, .9, .95, .9, .9, 1.1, 1.55,
              1.7, 1.75]

        self.c_l = sp.interpolate.interp1d(aoa_tuple, CL, fill_value="extrapolate")
        self.c_d = sp.interpolate.interp1d(aoa_tuple, CD, fill_value="extrapolate")

        self.derivative_matrix = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 1],
                                           [-900*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), -1440*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), 0, 0, -180*self.c_b/(self.rho*self.A*self.L), 210*self.c_b/(self.rho*self.A*self.L), 0, 0],
                                           [-672*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), -1764*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), 0, 0, 210*self.c_b/(self.rho*self.A*self.L), -252*self.c_b/(self.rho*self.A*self.L), 0, 0],
                                           [0, 0, -900*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), -1440*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), 0, 0, -180*self.c_b/(self.rho*self.A*self.L), 210*self.c_b/(self.rho*self.A*self.L)],
                                           [0, 0, -672*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), -1764*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), 0, 0, 210*self.c_b/(self.rho*self.A*self.L), -252*self.c_b/(self.rho*self.A*self.L)]])

        self.derivative_vector_x = np.array([0, 0, 0, 0, -30/(self.rho*self.A*self.L), 42/self.rho*self.A*self.L, 0, 0])
        self.derivative_vector_y = np.array([0, 0, 0, 0, 0, 0, -30/(self.rho*self.A*self.L), 42/self.rho*self.A*self.L])
        #self.derivative_vector_moment = np.array([0, 0, 0, 0, 0, 0, 2/(self.rho*self.J*self.L), 2/(self.rho*self.J*self.L), 0, 0, 0, 0])

        #print("Start test")
        #A = np.array([[1, 2], [3, 4]])
        #b = np.array([1, 2])
        #c = np.array([3, 4])
        #print("A*b = ",A.dot(b))
        #print("A*b+c =",A.dot(b)+c)
        #print("End test")

        self.simulate()

    def simulate(self):
        #gamma_0 = np.array([0, 0, 0, 0, 0, 0, .1, .1, .1, .1, .1, .1])
        gamma_0 = np.array([0, 0, 0, 0, .01, .01, .01, .01])
        #dt = .01
        #t = np.arange(.1, step=dt)
        t = (0, 30)
        solution = solve_ivp(self.derivative, t, gamma_0, full_output=1, mxstep=1 )
        #print("HU: ",info.get('hu'))
        #print("NST: ",info.get('nst'))
        #print("NFE: ",info.get('nfe'))

        #TEST: Euler's method by hand
        #solution = np.ndarray(shape=(len(t), 12))
        #gamma = gamma_0
        #solution[0, :] = gamma
        #for i in range(1,len(t)):
        #    d_gamma_dt = self.derivative(gamma, t)
        #    gamma += d_gamma_dt.dot(dt)
        #    solution[i, :] = gamma
        #    print(gamma)

        #solution = np.array(sol)
        #print(solution)

        #theta = [row[0]+row[1] for row in solution]
        #w_x = [row[2]+row[3] for row in solution]
        #w_y = [row[4]+row[5] for row in solution]

        #theta = solution.y[0, :]+solution.y[1, :]
        #theta_prime = solution[:, 6]+solution[:, 7]
        w_x = solution.y[0, :]+solution.y[1, :]
        w_y = solution.y[2, :]+solution.y[3, :]

        #print(solution[0:100, 0])
        #print(solution[0:100, 1])
        #print(theta)
        #print(theta_prime)
        print(w_x)
        print(w_y)
        #print(w_y[300:400])

        #plt.plot(solution.t, theta, 'b', label="theta")
        #plt.plot(t, theta_prime, 'g', label="theta_prime")
        plt.plot(solution.t, w_x, 'g', label="w_x")
        plt.plot(solution.t, w_y, 'r', label="w_y")
        print("Iterations: ",self.iterations)

    def derivative(self, t, gamma): #dy/dt function for odeint
        self.iterations += 1
        #print("Time: ",t)
        [Fx, Fy] = self.aerodynamics(gamma)
        print(Fx, Fy)
        d_gamma_dt = self.derivative_matrix.dot(gamma) + self.derivative_vector_x.dot(Fx) + self.derivative_vector_y.dot(Fy)
        #print(d_gamma_dt)
        return d_gamma_dt

    def aerodynamics(self, gamma):
        #theta = gamma[0] + gamma[1]
        #theta = (theta+2*math.pi) % (2*math.pi)
        #print("Theta: ",theta)
        #x = gamma[2] + gamma[3]
        #y = gamma[4] + gamma[5]
        #theta_dot = gamma[6] + gamma[7]
        x_dot = gamma[4] + gamma[5]
        y_dot = gamma[6] + gamma[7]
        print("X_Dot: ", x_dot)
        print("Y_Dot: ", y_dot)
        #print("Theta_Dot: ", theta_dot)

        x1_dot = x_dot
        x2_dot = x_dot
        y1_dot = y_dot
        y2_dot = y_dot
        print("1: ",x1_dot, y1_dot)
        print("2: ",x2_dot, y2_dot)

        v_r1_2 = math.pow(self.U*math.cos(self.delta) - x1_dot, 2) + math.pow(self.U*math.sin(self.delta) - y1_dot, 2)
        v_r2_2 = math.pow(self.U*math.cos(self.delta) - x2_dot, 2) + math.pow(self.U*math.sin(self.delta) - y2_dot, 2)

        #alpha_1 = math.pi/3 - theta + math.atan2(self.U*math.sin(self.delta) - y1_dot, self.U*math.cos(self.delta) - x1_dot)
        alpha_1 = math.pi / 3 + math.atan2(self.U*math.sin(self.delta) - y1_dot, self.U*math.cos(self.delta) - x1_dot)
        #alpha_1 = (alpha_1+2*math.pi) % (2*math.pi)
        alpha_2 = math.pi/3 + math.atan2(self.U*math.sin(self.delta) - y2_dot, self.U * math.cos(self.delta) - x2_dot)
        print("Alpha: ",alpha_1, alpha_2)

        F_x1 = -.5*self.air_density*v_r1_2 * self.b*self.L * ( -self.c_l(alpha_1)*math.sin(alpha_1-math.pi/3) + self.c_d(alpha_1)*math.cos(alpha_1-math.pi/3) )
        F_y1 = -.5*self.air_density*v_r1_2 * self.b*self.L * ( self.c_l(alpha_1)*math.cos(alpha_1-math.pi/3) + self.c_d(alpha_1)*math.sin(alpha_1-math.pi/3) )
        F_x2 = -.5*self.air_density*v_r2_2 * self.b*self.L * ( -self.c_l(alpha_2)*math.sin(alpha_2-math.pi/3) + self.c_d(alpha_2)*math.cos(alpha_2-math.pi/3) )
        F_y2 = -.5*self.air_density*v_r2_2 * self.b*self.L * ( self.c_l(alpha_2)*math.cos(alpha_2-math.pi/3) + self.c_d(alpha_2)*math.sin(alpha_2-math.pi/3) )


        #F_x2 = 0
        #F_y2 = 0

        F_x = F_x1 + F_x2
        F_y = F_y1 + F_y2
        #M_z = 0.5*self.e * (math.sin(theta)*(F_x1 - F_x2) + math.cos(theta)*(F_y2 - F_y1))
        #qprint(F_x1, F_y1, "       ", F_x2, F_y2)
        #F_x = 0
        #F_y = 0
        return [F_x, F_y]
