
import math

class Strip:
    def __init__(self, inner_mat, t_inner, piezo_mat, t_piezo, width, length):
        self.inner_material = inner_mat
        self.t_inner = t_inner
        self.piezo_material = piezo_mat
        self.t_piezo = t_piezo
        self.width = width
        self.length = length

        self.dEI = 1/3*piezo_mat.E*t_piezo*math.pow(width,3) + 1/6*inner_mat.E*t_inner*math.pow(width,3) \
        + 2*piezo_mat.E*(t_piezo*width*math.pow(.5*width+.5*t_piezo, 2) + 1/12*width*math.pow(t_piezo,3) ) \
        + 2*inner_mat.E*(t_inner*width*math.pow(.5*width+t_piezo+.5*t_inner, 2) + 1/12*width*math.pow(t_inner,3) ) \
        + 2*piezo_mat.E*(t_piezo*width*math.pow(.5*width+3/2*t_piezo+t_inner, 2) + 1/12*width*math.pow(t_piezo,3) )

        self.dRhoA = 8*piezo_mat.rho*width*t_piezo + 4*inner_mat.rho*width*t_inner