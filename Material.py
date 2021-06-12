

class Material:
    def __init__(self,name,isPiezo,*args):
        #Format: Name, Type, E, rho, F_tu, EndurancePoint(S,log10_N), **d31**
        self.name = name
        self.type = type
        self.E = args[0]
        self.rho = args[1]
        self.Ftu = args[2]
        self.endurance_point = args[3] #2-tuple of form (S_endurance, N_endurance)

        if isPiezo:
            self.d31 = args[4]
