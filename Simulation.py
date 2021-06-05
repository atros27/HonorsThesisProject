import math
import matplotlib.pyplot as plt
import numpy as np
import rainflow as rf
import scipy as sp
from scipy.integrate import solve_ivp
import time

#TESTSIM3 : Biaxial Bending-only Rayleigh-Ritz 2DOF ODE

class Simulation:
    def __init__(self):#, wind):
        #Todo: Align primary wind direction with tip bodies
        self.E = 68e9 #Aluminum Young's Modulus Todo: Replace with Material Class that holds these constants
        self.G = 25e9 #Aluminum Shear Modulus
        self.rho = 2.71e3 #kg/m^3
        self.air_density = 1.2 #kg/m^3
        self.c_t = .01 #Defunct
        self.c_b = .01 #Defunct
        self.zeta = .01
        alpha = 1
        beta = 0

        self.r = .003175 #.25 inch diameter for now Todo: Determine Geometric Settings, possibly customize
        self.b = .01
        self.e = .1 #Defunct: No twist, so moment arm is unnecessary
        self.L = 1#.25 #Note: Tip bodies and beam have identical length

        self.U = 2
        self.delta = 0/180*math.pi
        #self.wind = wind
        self.duration = 60#*60 #1 hour

        self.J = 0.5*math.pi*math.pow(self.r, 4)
        self.I = 0.25*math.pi*math.pow(self.r, 4)
        self.A = math.pi*math.pow(self.r, 2)

        self.endurance_limit = 68.95e6 #Endurance Limit in Pa
        self.endurance_crit = -6 #Log10 of inverse cycles at endurance. here there are 10^6 cycles at endurance
        self.fail_limit = 482.63e6 #Ultimate Tensile Strength in Pa
        self.miners_function = sp.interpolate.interp1d([self.endurance_limit, self.fail_limit], [self.endurance_crit, 0], fill_value="extrapolate")

        self.C = 8.87e-8
        self.m = 3.14
        a_per_width = np.array([0.1, 0.2, 0.3, 0.4, 0.5])*2*self.r
        faw = [1.044, 1.055, 1.125, 1.257, 1.5]
        self.f_factor = sp.interpolate.interp1d(a_per_width, faw, fill_value="extrapolate")

        self.iterations = 0
        self.miners_criteria = [0, 0, 0]

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

        c_l_grad = np.gradient(CL,aoa_tuple)
        c_d_grad = np.gradient(CD,aoa_tuple)
        self.c_l_slope = sp.interpolate.interp1d(aoa_tuple, c_l_grad, fill_value="extrapolate")
        self.c_d_slope = sp.interpolate.interp1d(aoa_tuple, c_d_grad, fill_value="extrapolate")

        #self.M_matrix = .5*np.array([[11/420, 643/20160], [643/20160, 437/11340]]).dot(self.rho*self.A*math.pow(self.L,5))
        #self.K_matrix = .5*np.array([[1/3, 5/12], [5/12, 8/15]]).dot(self.E*self.I*self.L)
        self.K_matrix = .5 * np.array([[1/3, 1/4], [1/4, 1/5]]).dot(self.E * self.I * self.L)
        self.M_matrix = .5 * np.array([[11/420, 173/20160], [173/20160, 13/810]]).dot(self.rho * self.A * math.pow(self.L, 5))
        self.M_inverse = np.linalg.inv(self.M_matrix)
        self.omega_matrix = -1*np.matmul(self.M_inverse,self.K_matrix)
        #self.damp_matrix = -1*np.multiply(np.sign(-1*self.omega_matrix),np.sqrt(abs(self.omega_matrix))).dot(2*self.zeta)
        #self.damp_matrix = -1*np.sqrt(abs(self.omega_matrix)).dot(2 * self.zeta)
        self.C_matrix = alpha*self.M_matrix + beta*self.K_matrix
        self.damp_matrix = -1*np.matmul(self.M_inverse,self.C_matrix)
        #self.force_vector = self.M_inverse.dot(np.array([[1/3],[5/12]])*self.L*self.L) #/(1/3+5/12))
        self.force_vector = self.M_inverse.dot(np.array([[1/8], [1/10]]) * self.L * self.L)

        matrix_top = np.concatenate((np.zeros(shape=(4,4)), np.identity(n=4)), axis=1)
        matrix_x_rows = np.concatenate((self.omega_matrix, np.zeros(shape=(2,2)), self.damp_matrix, np.zeros(shape=(2,2))), axis=1)
        matrix_y_rows = np.concatenate((np.zeros(shape=(2,2)), self.omega_matrix, np.zeros(shape=(2,2)), self.damp_matrix), axis=1)

        #jac_top = np.concatenate((np.zeros(shape=(4, 4)), np.identity(n=4), np.zeros(shape=(4, 2))), axis=1)
        #jac_x_rows = np.concatenate((self.omega_matrix, np.zeros(shape=(2, 2)), np.zeros(shape=(2,2)), np.zeros(shape=(2, 4))), axis=1)
        #jac_y_rows = np.concatenate((np.zeros(shape=(2, 2)), self.omega_matrix, np.zeros(shape=(2, 2)),np.zeros(shape=(2,2)), np.zeros(shape=(2, 2))), axis=1)
        #jac_moment_rows = np.zeros(shape=(2, 10))

        self.derivative_matrix = np.concatenate((matrix_top, matrix_x_rows, matrix_y_rows), axis=0)
        self.derivative_vector_x = np.concatenate((np.zeros(shape=(1,4)),self.force_vector.T,np.zeros(shape=(1,2))), axis=1)[0]
        self.derivative_vector_y = np.concatenate((np.zeros(shape=(1,6)),self.force_vector.T), axis=1)[0]
        #self.jac = np.concatenate((jac_top,jac_x_rows,jac_y_rows,jac_moment_rows), axis=0)

        print(self.M_matrix)
        print(self.K_matrix)
        print(self.M_inverse)
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
        #print(np.linalg.eig(self.derivative_matrix)[1])

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
        t0 = 0
        step_size = 1/4/40 #About one fourth of a period
        #gamma_0 = np.array([.01, .01, .01, .01, 0, 0, 0, 0])
        #dt = .01
        #t = np.arange(.1, step=dt)
        while max(self.miners_criteria)<1:
            t = (t0, t0+self.duration)
            start = time.time()
            time_series = np.arange(0, self.duration+1)
            #discrete_wind_series = np.array([self.wind.update() for i in range(0,self.duration+1)])
            #self.speed_series = sp.interpolate.interp1d(time_series, discrete_wind_series[:,0], fill_value="extrapolate")
            #self.direction_series = sp.interpolate.interp1d(time_series, discrete_wind_series[:,1], fill_value="extrapolate")

            #test_wind_series = np.array([[10,-180+360/300*current_second] for current_second in time_series])
            #self.test_speed_series = sp.interpolate.interp1d(time_series, test_wind_series[:,0])
            #self.test_direction_series = sp.interpolate.interp1d(time_series, test_wind_series[:,1])

            #plt.figure()
            #plt.plot(time_series, discrete_wind_series[:,0], 'b', label="Synthetic Speed")
            #plt.xlabel("Time [s]")
            #plt.ylabel("Speed [m/s]")
            #plt.legend()
            #plt.figure()
            #plt.plot(time_series, discrete_wind_series[:,1], 'b', label="Synthetic Direction")
            #plt.xlabel("Time [s]")
            #plt.ylabel("Angle [degrees]")
            #plt.legend()
            end = time.time()
            print("Synthetic Wind Series Created. Time Elapsed:",math.floor(end-start),"seconds")
            start = time.time()
            dt = 1
            #time_series = np.arange(0,self.duration+dt, step=dt)
            u_0 = np.array([[0,0,0,0]]).T
            u_dot_0 = np.array([[.01,.01,.01,.01]]).T
            #y = self.old_newmark(u_0,u_dot_0,time_series,dt,.5,.25)
            [time_series,y] = self.newmark(u_0,u_dot_0,self.duration,dt,1E-3,.9,.5,.25)
            #solution = solve_ivp(self.derivative, t, gamma_0, method='RK23')#, jac=self.jacobian)#, first_step=step_size)#, jac=self.jacobian)
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

            #w_x = math.pow(self.L,2)/3*y[0, :] + 5/12*math.pow(self.L,2)*y[1, :]
            #w_y = math.pow(self.L,2)/3*y[2, :] + 5/12*math.pow(self.L,2)*y[3, :]

            #w_x = math.pow(self.L,2)/3*solution.y[0, :] + 5/12*math.pow(self.L,2)*solution.y[1, :]
            #w_y = math.pow(self.L, 2) / 3 * solution.y[2, :] + 5 / 12 * math.pow(self.L, 2) * solution.y[3, :]
            #w_x = math.pow(self.L, 2) / 3 * solution.y[0, :] + 1 / 4 * math.pow(self.L, 2) * solution.y[1, :]
            #w_y = math.pow(self.L, 2) / 3 * solution.y[2, :] + 1 / 4 * math.pow(self.L, 2) * solution.y[3, :]

            #w_x = math.pow(self.L, 2) / 3 * y[0, :] + 5 / 12 * math.pow(self.L, 2) * y[1, :]
            #w_y = math.pow(self.L, 2) / 3 * y[2, :] + 5 / 12 * math.pow(self.L, 2) * y[3, :]
            w_x = math.pow(self.L, 2) / 3 * y[0, :] + 1 / 4 * math.pow(self.L, 2) * y[1, :]
            w_y = math.pow(self.L, 2) / 3 * y[2, :] + 1 / 4 * math.pow(self.L, 2) * y[3, :]

            #tip_moment_x = self.E*self.I*((2/self.L/self.L)*solution.y[0, :] + (6/self.L/self.L)*solution.y[1, :])
            #tip_moment_y = self.E*self.I*((2/self.L/self.L)*solution.y[2, :] + (6/self.L/self.L)*solution.y[3, :])
            #tip_shear_x = self.E*self.I*(6/self.L/self.L/self.L)*solution.y[1, :]
            #tip_shear_y = self.E*self.I*(6/self.L/self.L/self.L)*solution.y[3, :]
            #tip_max_bending_stress_x = self.r/self.I*tip_moment_x
            #tip_max_bending_stress_y = self.r/self.I*tip_moment_y

            #root_moment_x = self.E*self.I*(solution.y[0, :]+solution.y[1,:])
            #root_moment_y = self.E*self.I*(solution.y[2, :]+solution.y[3,:])

            root_moment_x = self.E*self.I*(y[0, :]+y[1,:])
            root_moment_y = self.E*self.I*(y[2, :]+y[3,:])

            #start = time.time()
            #root_moment_x = self.differentiate(solution.t,solution.y[8,:])
            #root_moment_y = self.differentiate(solution.t,solution.y[9,:])

            #Third Option: For t,gamma in t,gammas -> Moment = aerodynamics(t,gamma)*self.L
            #root_moment_x = np.zeros(shape=(len(solution.t),1))
            #root_moment_y = np.zeros(shape=(len(solution.t),1))
            #for i in range(0,len(solution.t)):
            #    [Fx,Fy] = self.aerodynamics(solution.t[i],solution.y[:,i])
            #    root_moment_x[i,0] = Fx*self.L
            #    root_moment_y[i,0] = Fy*self.L
            #end = time.time()
            #print("Moment Calculation Complete. Time Elapsed:",math.floor(end-start),"seconds")

            #root_shear_x = tip_shear_x
            #root_shear_y = tip_shear_y
            root_max_bending_stress_x = self.r/self.I*root_moment_x
            root_max_bending_stress_y = self.r/self.I*root_moment_y

            start = time.time()
            crit_point_1_bending_stress = -math.sqrt(3)/2*root_max_bending_stress_x + 0.5*root_max_bending_stress_y
            crit_point_2_bending_stress = root_max_bending_stress_y
            crit_point_3_bending_stress = math.sqrt(3)/2*root_max_bending_stress_x + 0.5*root_max_bending_stress_y
            rainflow_1 = [(rng,mean,count,i_start,i_end) for rng,mean,count,i_start,i_end in rf.extract_cycles(crit_point_1_bending_stress)]
            rainflow_2 = [(rng,mean,count,i_start,i_end) for rng,mean,count,i_start,i_end in rf.extract_cycles(crit_point_2_bending_stress)]
            rainflow_3 = [(rng,mean,count,i_start,i_end) for rng, mean, count, i_start, i_end in rf.extract_cycles(crit_point_3_bending_stress)]


            rainflow_1.sort(key=lambda row: row[3])
            rainflow_2.sort(key=lambda row: row[3])
            rainflow_3.sort(key=lambda row: row[3])

            end = time.time()
            print("Rainflow Counting Completed. Time Elapsed:",math.floor(end-start),"seconds")
            start = time.time()

            #rainflow_x_ampl = [obj[0] for obj in rainflow_x]
            #rainflow_y_ampl = [obj[0] for obj in rainflow_y]
            self.miners_criteria[0] += self.miners_rule(rainflow_1)
            self.miners_criteria[1] += self.miners_rule(rainflow_2)
            self.miners_criteria[2] += self.miners_rule(rainflow_3)
            #rainflow_x_mean = [obj[1] for obj in rainflow_x]
            #rainflow_x_count = [obj[2] for obj in rainflow_x]
            #rainflow_x_starts = [obj[3] for obj in rainflow_x]
            #rainflow_x_ends = [obj[4] for obj in rainflow_x]
            #print(rainflow_1)
            #print(rainflow_2)
            #print(rainflow_3)
            print("Final Miner's Criteria: ",self.miners_criteria)
            end = time.time()
            print("Fatigue Test (Miner's Rule) Completed. Time Elapsed:",math.floor(end-start),"seconds")

            #start = time.time()
            #cycle_list1 = range(0, len(rainflow_1))
            #cycle_list2 = range(0, len(rainflow_2))
            #cycle_list3 = range(0, len(rainflow_3))
            #a0 = 1e-6 #Initial Flaw: 1 Micron
            #crack1_solution = self.euler_method(a0, rainflow_1)
            #crack2_solution = self.euler_method(a0, rainflow_2)
            #crack3_solution = self.euler_method(a0, rainflow_3)
            #end = time.time()
            #print("Fatigue Test (SIF) Completed. Time Elapsed:",math.floor(end-start),"seconds")

            #print(rainflow_x_mean)
            #print(rainflow_x_count)
            #print(rainflow_x_starts)
            #print(rainflow_x_ends)

            #print(solution[0:100, 0])
            #print(solution[0:100, 1])
            #print(theta)
            #print(theta_prime)
            #print(w_x)
            #print(w_y)
            #print(w_y[300:400])

            #plt.plot(solution.t, theta, 'b', label="theta")
            #plt.plot(t, theta_prime, 'g', label="theta_prime")


            #Reset Loop
            #t0 = t[1]
            #gamma_0 = solution.y[:, -1]
            #step_size = solution.t[-1] - solution.t[-2]



            #peak_displ = max(max(abs(w_x)), max(abs(w_y)))
            #if peak_displ > self.L:
            #    print("Peak Displacement: ", peak_displ)
            #    print("Function unexpectedly diverged. Retrying...")
            #    continue  # Function blew up unexpectedly. Retry.

            #plt.figure()
            #plt.plot(solution.t, root_max_bending_stress_x / 1e6, 'g', label="root_bending_stress_x")
            #plt.title("Maximum Root Bending Stress [X-Moment]")
            #plt.xlabel("Time [s]")
            #plt.ylabel("Stress [MPa]")
            #plt.legend()
            #plt.figure()
            #plt.plot(solution.t, root_max_bending_stress_y / 1e6, 'r', label="root_bending_stress_y")
            #plt.title("Maximum Root Bending Stress [Y-Moment]")
            #plt.xlabel("Time [s]")
            #plt.ylabel("Stress [MPa]")
            #plt.legend()

            plt.show()

            #Cancel loop (for single-run tests)
            break

        #plt.figure()
        #plt.plot(solution.t, w_x, 'g', label="w_x")
        #plt.title("Tip Deflection in X-Direction")
        #plt.xlabel("Time [s]")
        #plt.ylabel("Displacement [m]")
        #plt.legend()
        #plt.figure()
        #plt.plot(solution.t, w_y, 'r', label="w_y")
        #plt.title("Tip Deflection in Y-Direction")
        #plt.xlabel("Time [s]")
        #plt.ylabel("Displacement [m]")
        #plt.legend()

        plt.figure()
        plt.plot(time_series, w_x, 'g', label="w_x")
        plt.title("Tip Deflection in X-Direction")
        plt.xlabel("Time [s]")
        plt.ylabel("Displacement [m]")
        plt.legend()
        plt.figure()
        plt.plot(time_series, w_y, 'r', label="w_y")
        plt.title("Tip Deflection in Y-Direction")
        plt.xlabel("Time [s]")
        plt.ylabel("Displacement [m]")
        plt.legend()

        #plt.figure()
        #plt.plot(solution.t, root_moment_x, 'g', label="moment_x")
        #plt.title("Root X-Moment")
        #plt.xlabel("Time [s]")
        #plt.ylabel("Moment [N-m]")
        #plt.legend()
        #plt.figure()
        #plt.plot(solution.t, root_moment_y, 'r', label="moment_y")
        #plt.title("Root Y-Moment")
        #plt.xlabel("Time [s]")
        #plt.ylabel("Moment [N-m]")
        #plt.legend()

        plt.figure()
        plt.plot(time_series, root_max_bending_stress_x / 1e6, 'g', label="root_bending_stress_x")
        plt.title("Maximum Root Bending Stress [X-Moment]")
        plt.xlabel("Time [s]")
        plt.ylabel("Stress [MPa]")
        plt.legend()
        plt.figure()
        plt.plot(time_series, root_max_bending_stress_y / 1e6, 'r', label="root_bending_stress_y")
        plt.title("Maximum Root Bending Stress [Y-Moment]")
        plt.xlabel("Time [s]")
        plt.ylabel("Stress [MPa]")
        plt.legend()

        plt.figure()
        plt.plot(time_series, y[2, :], 'r', label="gamma1_y")
        plt.title("Mode 1 [Y-Direction]")
        plt.xlabel("Time [s]")
        plt.legend()
        plt.figure()
        plt.plot(time_series, y[3,:], 'r', label="gamma2_y")
        plt.title("Mode 2 [Y-Direction]")
        plt.xlabel("Time [s]")
        plt.legend()

        #plt.figure()
        #plt.plot(time_series, y[3, :] / y[2, :], 'r', label="gamma2/gamma1_y")
        #plt.title("Mode Ratio [Y-Direction]")
        #plt.xlabel("Time [s]")
        #plt.legend()

        #plt.figure()
        #plt.plot(solution.t, root_max_bending_stress_x / 1e6, 'g', label="root_bending_stress_x")
        #plt.title("Maximum Root Bending Stress [X-Moment]")
        #plt.xlabel("Time [s]")
        #plt.ylabel("Stress [MPa]")
        #plt.legend()
        #plt.figure()
        #plt.plot(solution.t, root_max_bending_stress_y / 1e6, 'r', label="root_bending_stress_y")
        #plt.title("Maximum Root Bending Stress [Y-Moment]")
        #plt.xlabel("Time [s]")
        #plt.ylabel("Stress [MPa]")
        #plt.legend()
#
        #plt.figure()
        #plt.plot(solution.t, solution.y[2], 'g', label="gamma_3")
        #plt.title("Mode 1 Weight")
        #plt.xlabel("Time [s]")
        ##plt.ylabel("Stress [MPa]")
        #plt.legend()
        #plt.figure()
        #plt.plot(solution.t, solution.y[3], 'r', label="gamma_4")
        #plt.title("Mode 2 Weight")
        #plt.xlabel("Time [s]")
        ##plt.ylabel("Stress [MPa]")
        #plt.legend()

        #plt.figure()
        #plt.plot(solution.t, crit_point_1_bending_stress / 1e6, 'r', label="crit_point_1")
        #plt.title("Crit Point 1")
        #plt.xlabel("Time [s]")
        #plt.ylabel("Stress [MPa]")
        #plt.legend()

        #plt.figure()
        #plt.plot(solution.t, crit_point_2_bending_stress / 1e6, 'r', label="crit_point_2")
        #plt.title("Crit Point 2")
        #plt.xlabel("Time [s]")
        #plt.ylabel("Stress [MPa]")
        #plt.legend()
        #plt.figure()
        #plt.plot(solution.t, crit_point_3_bending_stress / 1e6, 'r', label="crit_point_3")
        #plt.title("Crit Point 3")
        #plt.xlabel("Time [s]")
        #plt.ylabel("Stress [MPa]")
        #plt.legend()

        #plt.figure()
        #plt.plot(cycle_list1, crack1_solution/self.r, 'b', label="crack_1")
        #plt.title("Crack 1")
        #plt.xlabel("Cycles")
        #plt.ylabel("Crack Fraction")
        #plt.legend()

        #plt.figure()
        #plt.plot(cycle_list2, crack2_solution/self.r, 'b', label="crack_2")
        #plt.title("Crack 2")
        #plt.xlabel("Cycles")
        #plt.ylabel("Crack Fraction")
        #plt.legend()
        #plt.figure()
        #plt.plot(cycle_list3, crack3_solution/self.r, 'b', label="crack_3")
        #plt.title("Crack 3")
        #plt.xlabel("Cycles")
        #plt.ylabel("Crack Fraction")
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
        #print("Aerodynamic Moment:", Fy*self.L)
        d_gamma_dt = self.derivative_matrix.dot(gamma) + self.derivative_vector_x.dot(Fx) + self.derivative_vector_y.dot(Fy)

        #print(gamma)
        #print(d_gamma_dt)
        return d_gamma_dt

    def newmark(self, u_0, u_dot_0, tf, dt_init, rel_error, shrink_factor, alpha, beta):

        #Initialize System Parameters
        M_top = np.concatenate((self.M_matrix, np.zeros(shape=(2,2))), axis=1)
        M_bot = np.concatenate((np.zeros(shape=(2,2)), self.M_matrix), axis=1)
        M = np.concatenate((M_top, M_bot), axis=0)
        print("M=",M)

        C_mat = self.C_matrix
        C_top = np.concatenate((C_mat,np.zeros(shape=(2,2))), axis=1)
        C_bot = np.concatenate((np.zeros(shape=(2,2)),C_mat), axis=1)
        C = np.concatenate((C_top,C_bot), axis=0)

        print("C = ",C)

        K_top = np.concatenate((self.K_matrix, np.zeros(shape=(2, 2))), axis=1)
        K_bot = np.concatenate((np.zeros(shape=(2, 2)), self.K_matrix), axis=1)
        K = np.concatenate((K_top, K_bot), axis=0)
        print("K=",K)

        #Initialize State Vectors
        time = np.zeros(shape=(1,1))
        #u = np.zeros(shape=(4,1))
        u = u_0
        u_dot_prev = u_dot_0
        u_ddot_prev = np.linalg.solve(M, self.newmark_aero(u_dot_prev[:])-C.dot(u_dot_prev)-K.dot(u))
        #i = 1
        dt = dt_init
        time = np.append(time, time[-1] + dt)
        print(u)
        print(u_dot_prev)
        print(u_ddot_prev)
        while time[-1] < tf:
            error_criterion = (rel_error+1)*np.ones(shape=(4,1))
            dt = dt/shrink_factor #Increase time step to prevent premature shortening
            #while all(x > abs_error for x in error_criterion[:,0]):
            while np.linalg.norm(error_criterion) > rel_error*np.linalg.norm(u_dot_prev):
                #Shorten time steps for each failed iteration
                dt = dt*shrink_factor

                #Update (or try to Update again)
                u_new = np.array([u[:,-1]]).T + u_dot_prev*dt + dt*dt/2*(1-2*beta)*u_ddot_prev
                u_dot = u_dot_prev + dt*(1-alpha)*u_ddot_prev
                #print("u:",u_new)
                #print("u_dot:",u_dot)
                #u_new = np.array([u_new[:,0]]).T #Normalize and remove repetitive entries
                #u_dot = np.array([u_dot[:,0]]).T
                #print(u_dot[:])
                f_i = self.newmark_aero(u_dot[:])
                matrix = M + alpha*dt*C + beta*dt*dt*K
                vector = f_i - C.dot(u_dot) - K.dot(u_new)
                u_ddot = np.linalg.solve(matrix, vector)

                #Eval Error Criterion
                error_criterion = abs((beta-1/6)*dt*dt*(u_ddot - u_ddot_prev))

            #Finalize new state
            u = np.append(u, u_new, axis=1)

            #Reset Names
            u_dot_prev = u_dot
            u_ddot_prev = u_ddot
            #i += 1
            time = np.append(time, time[-1] + dt)
            print(dt)
        time = time[0:len(time)-1]
        return [time,u]

    def old_newmark(self, u_0, u_dot_0, time, dt, alpha, beta):
        #full_omega_matrix_top = np.concatenate((self.omega_matrix, np.zeros(shape=(2, 2))), axis=1)
        #full_omega_matrix_bottom = np.concatenate((np.zeros(shape=(2,2)),self.omega_matrix), axis=1)
        #full_omega_matrix = np.concatenate((full_omega_matrix_top,full_omega_matrix_bottom),axis=0)
        #full_damp_matrix_top = np.concatenate((self.damp_matrix, np.zeros(shape=(2, 2))), axis=1)
        #full_damp_matrix_bottom = np.concatenate((np.zeros(shape=(2, 2)), self.damp_matrix), axis=1)
        #full_damp_matrix = np.concatenate((full_damp_matrix_top, full_damp_matrix_bottom), axis=0)
        #print(full_omega_matrix)
        #print(full_damp_matrix)

        #force_x = np.concatenate((self.force_vector,np.zeros(shape=(2,1))), axis=0)[:, 0]
        #force_y = np.concatenate((np.zeros(shape=(2,1)),self.force_vector), axis=0)[:, 0]
        #print(force_x)

        M_top = np.concatenate((self.M_matrix, np.zeros(shape=(2,2))), axis=1)
        M_bot = np.concatenate((np.zeros(shape=(2,2)), self.M_matrix), axis=1)
        M = np.concatenate((M_top, M_bot), axis=0)
        print("M=",M)

        #zeta = 10
        #C = M*zeta
        C_mat = self.C_matrix
        C_top = np.concatenate((C_mat,np.zeros(shape=(2,2))), axis=1)
        C_bot = np.concatenate((np.zeros(shape=(2,2)),C_mat), axis=1)
        C = np.concatenate((C_top,C_bot), axis=0)
        #C = self.C_matrix
        #C = np.zeros(shape=(4,4))

        print("C = ",C)
        #zeta = .01
        #C = np.array([[M[0,0]*zeta, 0, 0, 0],
        #              [0, M[1,1]*zeta, 0, 0],
        #              [0, 0, M[2,2]*zeta, 0],
        #              [0, 0, 0, M[3,3]*zeta]])

        K_top = np.concatenate((self.K_matrix, np.zeros(shape=(2, 2))), axis=1)
        K_bot = np.concatenate((np.zeros(shape=(2, 2)), self.K_matrix), axis=1)
        K = np.concatenate((K_top, K_bot), axis=0)
        print("K=",K)
        #C = -(0 * M + .02 / np.sqrt(abs(self.omega_matrix[1,1])) * K)
        #C = -self.C_matrix
        #print("C=",C)

        u = np.zeros(shape=(4,len(time)))
        #u_dot = np.zeros(shape=(4,1))
        #u_ddot = np.zeros(shape=(4,1))
        u[:, 0] = u_0.T
        u_dot_prev = u_dot_0.T
        u_ddot_prev = np.linalg.solve(M, self.newmark_aero(u_dot_0.T)-C.dot(u_dot_0.T)-K.dot(u_0.T))
        for i in range(1,len(time)):
            u[:,i] = u[:,i-1] + u_dot_prev*dt + dt*dt/2*(1-2*beta)*u_ddot_prev
            u_dot = u_dot_prev + dt*(1-alpha)*u_ddot_prev
            f_i = self.newmark_aero(u_dot)
            #print(f_i)
            matrix = M + alpha*dt*C + beta*dt*dt*K
            vector = f_i - C.dot(u_dot.T) - K.dot(u[:,i].T)
            u_ddot = np.linalg.solve(matrix, vector)

            u_dot_prev = u_dot
            u_ddot_prev = u_ddot
            #print(u_ddot[3])
        return u

    def newmark_aero(self,u_dot):
        U = 10
        delta = 0/180*math.pi

        phi_1 = math.pow(self.L,2)/3
        phi_2 = math.pow(self.L,2)/4

        force_1 = math.pow(self.L,2)/8
        force_2 = math.pow(self.L,2)/10

        x_dot = phi_1*u_dot[0,0] + phi_2*u_dot[1,0]
        y_dot = phi_1*u_dot[2,0] + phi_2*u_dot[3,0]

        #print(u_dot)
        #print(x_dot,y_dot)

        v_r2 = math.pow(U*math.cos(delta) - x_dot, 2) + math.pow(U*math.sin(delta) - y_dot, 2)
        alpha = math.pi/3 + math.atan2(U*math.sin(delta) - y_dot, U*math.cos(delta) - x_dot)
        #print(alpha/math.pi*180)

        Fx = self.air_density*v_r2 * self.b*self.L * ( -self.c_l(alpha)*math.sin(alpha-math.pi/3) + self.c_d(alpha)*math.cos(alpha-math.pi/3) )
        Fy = self.air_density*v_r2 * self.b*self.L * ( self.c_l(alpha)*math.cos(alpha-math.pi/3) + self.c_d(alpha)*math.sin(alpha-math.pi/3) )

        #Fx = 0
        #Fy = 0

        ans = np.array([[force_1*Fx,force_2*Fx,force_1*Fy,force_2*Fy]]).T
        #print(ans.T)
        return ans

    def aerodynamics(self, t, gamma):
        #U = self.speed_series(t)/10
        #delta = self.direction_series(t)/180*math.pi
        #Static Wind Test Case
        U = 5
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
        F_x = 0
        F_y = 0
        return [F_x, F_y]

    def miners_rule(self,cycles):
        miners_total = 0
        num=0
        for sigma in cycles:
            num += 1
            if(sigma[0] >= self.endurance_limit):
                #print(sigma[0])
                miners_total += math.pow(10, self.miners_function(sigma[0]))
            if(miners_total>=1):
                print(num,"cycles")
                break
        return miners_total

    def euler_method(self,a0,rainflow):
        crack = np.zeros(shape=(1,len(rainflow)))
        crack[0] = a0
        for i in range(1,len(rainflow)):
            crack[i] = crack[i-1] + self.crack_growth(crack[i-1],rainflow[i])*rainflow[i][2]
        return crack

    def crack_growth(self,a,rainflow): #Returns da/dN (Paris-Erdogan)
        K_max = self.stress_intensity_factor(a, rainflow[1] + rainflow[0])
        K_min = self.stress_intensity_factor(a, rainflow[1] - rainflow[0]) #mean stress minus stress amplitude
        dadN = self.C*math.pow(K_max-K_min, self.m)
        return dadN

    def stress_intensity_factor(self,a,sigma): #SIF for a beam undergoing bending
        f_a_w = self.f_factor(a)[0]
        print("faw:",f_a_w,"sigma:",sigma)
        K = f_a_w * sigma * math.sqrt(math.pi*a)
        return K

    def differentiate(self,t,x_dot):
        x = np.zeros(shape=(len(x_dot),1))
        for i in range(0,len(x_dot)-2):
            x[i,0] = (x_dot[i+1] - x_dot[i])/(t[i+1] - t[i])
        return x

    def jacobian(self, t, gamma):
        #U = self.speed_series(t)/10
        #delta = self.direction_series(t)/180*math.pi
        U = 5
        delta = 0

        x_dot = math.pow(self.L,2)/3*gamma[4] + 5/12*math.pow(self.L,2)*gamma[5]
        y_dot = math.pow(self.L,2)/3*gamma[6] + 5/12*math.pow(self.L,2)*gamma[7]
        phi_1 = math.pow(self.L,2)/3
        phi_2 = 5/12*math.pow(self.L,2)
        phi_3 = phi_1
        phi_4 = phi_2

        v_r_2 = math.pow(U*math.cos(delta) - x_dot, 2) + math.pow(U*math.sin(delta) - y_dot, 2)
        d_vr2_d_xdot = 2*x_dot - 2*U*math.cos(delta)
        d_vr2_d_ydot = 2*y_dot - 2*U*math.sin(delta)

        alpha = math.pi / 3 + math.atan2(U*math.sin(delta) - y_dot, U*math.cos(delta) - x_dot)
        d_alpha_d_xdot = (U*math.sin(delta) - y_dot)/v_r_2
        d_alpha_d_ydot = (x_dot - U*math.cos(delta))/v_r_2

        F_x = self.air_density*v_r_2 * self.b*self.L * ( -self.c_l(alpha)*math.sin(alpha-math.pi/3) + self.c_d(alpha)*math.cos(alpha-math.pi/3) )
        F_y = self.air_density*v_r_2 * self.b*self.L * ( self.c_l(alpha)*math.cos(alpha-math.pi/3) + self.c_d(alpha)*math.sin(alpha-math.pi/3) )

        d_Fx_d_vr2 = F_x/v_r_2
        d_Fx_d_alpha = self.air_density*self.b*self.L*(-self.c_l_slope(alpha)*math.sin(alpha-math.pi/3) - self.c_l(alpha)*math.cos(alpha-math.pi/3) + self.c_d_slope(alpha)*math.cos(alpha-math.pi/3) - self.c_d(alpha)*math.sin(alpha-math.pi/3))
        d_Fy_d_vr2 = F_y/v_r_2
        d_Fy_d_alpha = self.air_density*self.b*self.L*(self.c_l_slope(alpha)*math.cos(alpha-math.pi/3) - self.c_l(alpha)*math.sin(alpha-math.pi/3) + self.c_d_slope(alpha)*math.sin(alpha-math.pi/3) + self.c_d(alpha)*math.cos(alpha-math.pi/3))

        d_Fx_d_xdot = d_Fx_d_vr2*d_vr2_d_xdot + d_Fx_d_alpha*d_alpha_d_xdot
        d_Fx_d_ydot = d_Fx_d_vr2*d_vr2_d_ydot + d_Fx_d_alpha*d_alpha_d_ydot
        d_Fy_d_xdot = d_Fy_d_vr2*d_vr2_d_xdot + d_Fy_d_alpha*d_alpha_d_xdot
        d_Fy_d_ydot = d_Fy_d_vr2*d_vr2_d_ydot + d_Fy_d_alpha*d_alpha_d_ydot

        jac_force_main = np.array([[phi_1*phi_1*d_Fx_d_xdot, phi_1*phi_2*d_Fx_d_xdot, phi_1*phi_1*d_Fx_d_ydot, phi_1*phi_2*d_Fx_d_ydot],
                                   [phi_1*phi_2*d_Fx_d_xdot, phi_2*phi_2*d_Fx_d_xdot, phi_1*phi_2*d_Fx_d_ydot, phi_2*phi_2*d_Fx_d_ydot],
                                   [phi_1*phi_1*d_Fy_d_xdot, phi_1*phi_2*d_Fy_d_xdot, phi_1*phi_1*d_Fy_d_ydot, phi_1*phi_2*d_Fy_d_ydot],
                                   [phi_1*phi_1*d_Fy_d_xdot, phi_1*phi_2*d_Fy_d_xdot, phi_1*phi_1*d_Fy_d_ydot, phi_1*phi_2*d_Fy_d_ydot]])
        jac_force_top = np.zeros(shape=(4,8))
        jac_force_mid = np.concatenate((np.zeros(shape=(4,4)),jac_force_main), axis=1)
        jac_force = np.concatenate((jac_force_top,jac_force_mid), axis=0)

        jac = self.derivative_matrix + jac_force[0]
        #print(jac.shape)
        #jac = np.reshape(jac,(10,10))
        #print(jac.shape)
        return jac