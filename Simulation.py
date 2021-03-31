import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import time

#TESTSIM3 : Biaxial Bending-only Rayleigh-Ritz 2DOF ODE

class Simulation:
    def __init__(self):#, wind):
        #Todo: Align primary wind direction with tip bodies
        self.E = 68e6 #Aluminum Young's Modulus Todo: Replace with Material Class that holds these constants
        self.G = 25e6 #Aluminum Shear Modulus
        self.rho = 2.71e3 #kg/m^3
        self.air_density = 1.2 #kg/m^3
        self.c_t = .01
        self.c_b = .01
        self.zeta = .01

        self.r = .01 #1 cm radius for now Todo: Determine Geometric Settings, possibly customize
        self.b = .01
        self.e = .1
        self.L = .25 #Note: Tip bodies and shaft-beam have identical length

        #self.U = 10
        #self.delta = 120/180*math.pi
        #self.wind = wind

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

        M_matrix = np.array([[11/420, 643/20160], [643/20160, 437/11340]]).dot(self.rho*self.A*math.pow(self.L,5))
        K_matrix = np.array([[1/3, 5/12], [5/12, 8/15]]).dot(self.E*self.I*self.L)
        M_inverse = np.linalg.inv(M_matrix)
        self.omega_matrix = -1*abs(np.matmul(M_inverse,K_matrix))
        self.damp_matrix = -1*np.sqrt(-1*self.omega_matrix).dot(2*self.zeta)
        self.force_vector = M_inverse.dot(np.array([[self.L*self.L/3],[5/12*self.L*self.L]]))

        matrix_top = np.concatenate((np.zeros(shape=(4,4)), np.identity(n=4)), axis=1)
        matrix_x_rows = np.concatenate((self.omega_matrix, np.zeros(shape=(2,2)), self.damp_matrix, np.zeros(shape=(2,2))), axis=1)
        matrix_y_rows = np.concatenate((np.zeros(shape=(2,2)), self.omega_matrix, np.zeros(shape=(2,2)), self.damp_matrix), axis=1)

        self.derivative_matrix = np.concatenate((matrix_top, matrix_x_rows, matrix_y_rows), axis=0)
        self.derivative_vector_x = np.concatenate((np.zeros(shape=(1,4)),self.force_vector.T,np.zeros(shape=(1,2))), axis=1)[0]
        self.derivative_vector_y = np.concatenate((np.zeros(shape=(1,6)),self.force_vector.T), axis=1)[0]

        print(M_inverse)
        print(self.omega_matrix)
        print(np.linalg.eig(self.derivative_matrix)[0])
        print(self.damp_matrix)
        print(self.derivative_matrix)
        print(self.derivative_vector_x)

        self.old_derivative_matrix = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 1],
                                           [-900*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), -1440*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), 0, 0, -180*self.c_b/(self.rho*self.A*self.L), 210*self.c_b/(self.rho*self.A*self.L), 0, 0],
                                           [-672*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), -1764*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), 0, 0, 210*self.c_b/(self.rho*self.A*self.L), -252*self.c_b/(self.rho*self.A*self.L), 0, 0],
                                           [0, 0, -900*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), -1440*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), 0, 0, -180*self.c_b/(self.rho*self.A*self.L), 210*self.c_b/(self.rho*self.A*self.L)],
                                           [0, 0, -672*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), -1764*self.E*self.I/(self.rho*self.A*math.pow(self.L,4) ), 0, 0, 210*self.c_b/(self.rho*self.A*self.L), -252*self.c_b/(self.rho*self.A*self.L)]])
        #print(np.linalg.eig(self.old_derivative_matrix))

        #self.derivative_vector_x = np.array([0, 0, 0, 0, -30/(self.rho*self.A*self.L), 42/self.rho*self.A*self.L, 0, 0])
        #self.derivative_vector_y = np.array([0, 0, 0, 0, 0, 0, -30/(self.rho*self.A*self.L), 42/self.rho*self.A*self.L])
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
        #gamma_0 = np.array([.01, .01, .01, .01, 0, 0, 0, 0])
        #dt = .01
        #t = np.arange(.1, step=dt)
        t = (0, 30)
        start = time.time()
        #time_series = np.arange(0, 301)
        #discrete_wind_series = np.array([self.wind.update() for i in range(0,301)])
        #self.speed_series = sp.interpolate.interp1d(time_series, discrete_wind_series[:,0])
        #self.direction_series = sp.interpolate.interp1d(time_series, discrete_wind_series[:,1])
        #test_wind_series = np.array([[10,-180+360/300*current_second] for current_second in time_series])
        #self.test_speed_series = sp.interpolate.interp1d(time_series, test_wind_series[:,0])
        #self.test_direction_series = sp.interpolate.interp1d(time_series, test_wind_series[:,1])
        #plt.figure()
        #plt.plot(time_series, discrete_wind_series[:,0], 'b', label="Synthetic Speed")
        #plt.legend()
        #plt.figure()
        #plt.plot(time_series, discrete_wind_series[:,1], 'b', label="Synthetic Direction")
        #plt.legend()
        end = time.time()
        print("Synthetic Wind Series Created. Time Elapsed:",math.floor(end-start),"seconds")
        start = time.time()
        dt = .01
        time_series = np.arange(0,30+dt, step=dt)
        u_0 = np.array([0,0,0,0])
        u_dot_0 = np.array([.01,.01,.01,.01])
        tx = time_series
        #y = self.newmark(u_0,u_dot_0,time_series,dt)
        solution = solve_ivp(self.derivative, t, gamma_0, method='RK23')
        end = time.time()
        print("IVP Solved. Time Elapsed:",math.floor(end-start),"seconds")
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

        #w_x = math.pow(self.L,2)/3*solution.y[0, :] + 5/12*math.pow(self.L,2)*solution.y[1, :]
        #w_y = math.pow(self.L,2)/3*solution.y[2, :] + 5/12*math.pow(self.L,2)*solution.y[3, :]
        w_x = math.pow(self.L,2)/3*solution.y[0, :] + 5/12*math.pow(self.L,2)*solution.y[1, :]
        w_y = math.pow(self.L, 2) / 3 * solution.y[2, :] + 5 / 12 * math.pow(self.L, 2) * solution.y[3, :]

        #tip_moment_x = self.E*self.I*((2/self.L/self.L)*solution.y[0, :] + (6/self.L/self.L)*solution.y[1, :])
        #tip_moment_y = self.E*self.I*((2/self.L/self.L)*solution.y[2, :] + (6/self.L/self.L)*solution.y[3, :])
        #tip_shear_x = self.E*self.I*(6/self.L/self.L/self.L)*solution.y[1, :]
        #tip_shear_y = self.E*self.I*(6/self.L/self.L/self.L)*solution.y[3, :]
        #tip_max_bending_stress_x = self.r/self.I*tip_moment_x
        #tip_max_bending_stress_y = self.r/self.I*tip_moment_y

        #root_moment_x = self.E*self.I*(2/self.L/self.L)*solution.y[0, :]
        #root_moment_y = self.E*self.I*(2/self.L/self.L)*solution.y[2, :]
        #root_shear_x = tip_shear_x
        #root_shear_y = tip_shear_y
        #root_max_bending_stress_x = self.r/self.I*root_moment_x
        #root_max_bending_stress_y = self.r/self.I*root_moment_y

        #print(solution[0:100, 0])
        #print(solution[0:100, 1])
        #print(theta)
        #print(theta_prime)
        print(w_x)
        print(w_y)
        #print(w_y[300:400])

        #plt.plot(solution.t, theta, 'b', label="theta")
        #plt.plot(t, theta_prime, 'g', label="theta_prime")
        plt.figure()
        plt.plot(solution.t, w_x, 'g', label="w_x")
        plt.plot(solution.t, w_y, 'r', label="w_y")
        plt.legend()
        #plt.figure()
        #plt.plot(solution.t, tip_max_bending_stress_x, 'g', label="tip_bending_stress_x")
        #plt.figure()
        #plt.plot(solution.t, tip_max_bending_stress_y, 'r', label="tip_bending_stress_y")
        #plt.legend()
        #plt.figure()
        #plt.plot(solution.t, root_max_bending_stress_x, 'g', label="root_bending_stress_x")
        #plt.plot(solution.t, root_max_bending_stress_y, 'r', label="root_bending_stress_y")
        #plt.legend()
        #plt.figure()
        #plt.plot(solution.t, tip_shear_x, 'g', label="tip_shear_x")
        #plt.plot(solution.t, tip_shear_y, 'r', label="tip_shear_y")
        #plt.legend()
        #plt.figure()
        #plt.plot(solution.t, root_shear_x, 'g', label="root_shear_x")
        #plt.plot(solution.t, root_shear_y, 'r', label="root_shear_y")
        #plt.legend()
        print("Iterations: ",self.iterations)

    def derivative(self, t, gamma): #dy/dt function for odeint
        self.iterations += 1
        #print("Time: ",t)
        [Fx, Fy] = self.aerodynamics(t, gamma)
        #print(Fx, Fy)
        d_gamma_dt = self.derivative_matrix.dot(gamma) + self.derivative_vector_x.dot(Fx) + self.derivative_vector_y.dot(Fy)
        #print(gamma)
        #print(d_gamma_dt)
        return d_gamma_dt

    def newmark(self, u_0, u_dot_0, time, dt):
        full_omega_matrix_top = np.concatenate((self.omega_matrix, np.zeros(shape=(2, 2))), axis=1)
        full_omega_matrix_bottom = np.concatenate((np.zeros(shape=(2,2)),self.omega_matrix), axis=1)
        full_omega_matrix = np.concatenate((full_omega_matrix_top,full_omega_matrix_bottom),axis=0)
        full_damp_matrix_top = np.concatenate((self.damp_matrix, np.zeros(shape=(2, 2))), axis=1)
        full_damp_matrix_bottom = np.concatenate((np.zeros(shape=(2, 2)), self.damp_matrix), axis=1)
        full_damp_matrix = np.concatenate((full_damp_matrix_top, full_damp_matrix_bottom), axis=0)
        print(full_omega_matrix)
        print(full_damp_matrix)

        force_x = np.concatenate((self.force_vector,np.zeros(shape=(2,1))), axis=0)[:, 0]
        force_y = np.concatenate((np.zeros(shape=(2,1)),self.force_vector), axis=0)[:, 0]
        print(force_x)

        u = np.zeros(shape=(4,len(time)))
        u_dot = np.zeros(shape=(4,len(time)))
        u_ddot = np.zeros(shape=(4,len(time)))
        u[:, 0] = u_0.T
        u_dot[:, 0] = u_dot_0.T
        for i in range(1,len(time)):
            #beta = 0
            u_dot[:, i] = u_dot[:, i-1] + dt*u_ddot[:, i-1]
            u[:, i] = u[:, i-1] + dt*u_dot[:, i-1] + 0.5*dt*dt*u_ddot[:, i-1]
            [Fx,Fy] = self.newmark_aero(u_dot[:, i])
            print(Fx,Fy)
            u_ddot[:, i] = force_x*Fx + force_y*Fy - full_damp_matrix.dot(u_dot[:, i]) - full_omega_matrix.dot(u[:, i])
        return u

    def newmark_aero(self,u_dot):
        U = 1
        delta = 0/180*math.pi

        #x_dot = math.pow(self.L,2)/3*u_dot[0] + 5/12*math.pow(self.L,2)*u_dot[1]
        #y_dot = math.pow(self.L,2)/3*u_dot[2] + 5/12*math.pow(self.L, 2)*u_dot[3]

        #v_r2 = math.pow(U*math.cos(delta) - x_dot, 2) + math.pow(U*math.sin(delta) - y_dot, 2)
        #alpha = math.pi/3 + math.atan2(U*math.sin(delta) - y_dot, U*math.cos(delta) - x_dot)
        #print(alpha)

        #Fx = -self.air_density*v_r2 * self.b*self.L * ( -self.c_l(alpha)*math.sin(alpha-math.pi/3) + self.c_d(alpha)*math.cos(alpha-math.pi/3) )
        #Fy = -self.air_density*v_r2 * self.b*self.L * ( self.c_l(alpha)*math.cos(alpha-math.pi/3) + self.c_d(alpha)*math.sin(alpha-math.pi/3) )

        Fx = 0
        Fy = 0

        return [Fx,Fy]

    def aerodynamics(self, t, gamma):
        #U = self.speed_series(t)
        #delta = self.direction_series(t)/180*math.pi
        #Static Wind Test Case
        U = 20
        delta = 0/180*math.pi
        #Sweeping Wind Test Case
        #U = self.test_speed_series(t)
        #delta = self.test_direction_series(t)

        #theta = gamma[0] + gamma[1]
        #theta = (theta+2*math.pi) % (2*math.pi)
        #print("Theta: ",theta)
        #x = gamma[2] + gamma[3]
        #y = gamma[4] + gamma[5]
        #theta_dot = gamma[6] + gamma[7]

        #x_dot = gamma[4] + gamma[5]
        #y_dot = gamma[6] + gamma[7]

        x_dot = math.pow(self.L,2)/3*gamma[4] + 5/12*math.pow(self.L,2)*gamma[5]
        y_dot = math.pow(self.L,2)/3*gamma[6] + 5/12*math.pow(self.L,2)*gamma[7]

        #print("X_Dot: ", x_dot)
        #print("Y_Dot: ", y_dot)
        #print("Theta_Dot: ", theta_dot)

        x1_dot = x_dot
        x2_dot = x_dot
        y1_dot = y_dot
        y2_dot = y_dot
        #print("1: ",x1_dot, y1_dot)
        #print("2: ",x2_dot, y2_dot)

        v_r1_2 = math.pow(U*math.cos(delta) - x1_dot, 2) + math.pow(U*math.sin(delta) - y1_dot, 2)
        v_r2_2 = math.pow(U*math.cos(delta) - x2_dot, 2) + math.pow(U*math.sin(delta) - y2_dot, 2)

        #alpha_1 = math.pi/3 - theta + math.atan2(self.U*math.sin(self.delta) - y1_dot, self.U*math.cos(self.delta) - x1_dot)
        alpha_1 = math.pi / 3 + math.atan2(U*math.sin(delta) - y1_dot, U*math.cos(delta) - x1_dot)
        #alpha_1 = (alpha_1+2*math.pi) % (2*math.pi)
        alpha_2 = math.pi/3 + math.atan2(U*math.sin(delta) - y2_dot, U*math.cos(delta) - x2_dot)
        #print("Alpha: ",alpha_1)

        F_x1 = .5*self.air_density*v_r1_2 * self.b*self.L * ( -self.c_l(alpha_1)*math.sin(alpha_1-math.pi/3) + self.c_d(alpha_1)*math.cos(alpha_1-math.pi/3) )
        F_y1 = .5*self.air_density*v_r1_2 * self.b*self.L * ( self.c_l(alpha_1)*math.cos(alpha_1-math.pi/3) + self.c_d(alpha_1)*math.sin(alpha_1-math.pi/3) )
        F_x2 = .5*self.air_density*v_r2_2 * self.b*self.L * ( -self.c_l(alpha_2)*math.sin(alpha_2-math.pi/3) + self.c_d(alpha_2)*math.cos(alpha_2-math.pi/3) )
        F_y2 = .5*self.air_density*v_r2_2 * self.b*self.L * ( self.c_l(alpha_2)*math.cos(alpha_2-math.pi/3) + self.c_d(alpha_2)*math.sin(alpha_2-math.pi/3) )


        #F_x2 = 0
        #F_y2 = 0

        F_x = F_x1 + F_x2
        F_y = F_y1 + F_y2
        #M_z = 0.5*self.e * (math.sin(theta)*(F_x1 - F_x2) + math.cos(theta)*(F_y2 - F_y1))
        #qprint(F_x1, F_y1, "       ", F_x2, F_y2)
        #F_x = 0
        #F_y = 0
        return [F_x, F_y]
