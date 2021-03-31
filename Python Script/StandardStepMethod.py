import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Secant(f, x0, x1, target = 0, eps = 0.0001, maxiter = 1000, *args):
#  print("   ***  Numerical Analysis - Secant Method  ***   ")
#  print("--------------------------------------------------")
  xlama = x0
  x = x1
  y = target
  error = 1
  i = 0
  while error > eps and i < maxiter:
    xbaru = x + (y - f(x, *args))*(xlama - x)/(f(xlama, *args) - f(x, *args))
    error = np.abs(xbaru - x)/np.abs(xbaru)
    xlama = x
    x = xbaru
    i += 1
#    print('Iterasi ke {}, nilai x = {:.3f} dengan error {:.3f}%'.format(i, x, error*100))
#  print("--------------------------------------------------")
#  print("Nilai X adalah {:.3f} dengan error {:.3f}%".format(x, error*100))
  return x

def Sf_manning(n, v, R): #energy slope based on Manning's Equation
    """
    n = Manning Coeficient [=]
    v = velocity [m/s]
    R = Hydraulic Radius [m]
    """
    a = (n**2)*(v**2)
    b = R**(4/3)
    return a/b

class channel():   
    def __init__(self, location, base_width, left_slope, right_slope, manning, slope):
        self.x = location
        self.B = base_width
        self.m1 = left_slope
        self.m2 = right_slope
        self.n = manning
        self.S0 = slope
    
    def area(self, y):
        return (self.B + y*((self.m1+self.m2)/2))*y

    def wetP(self, y):
        return self.B + y*np.sqrt(1+self.m1**2) + y*np.sqrt(1+self.m2**2)
    
    def hydraulicRadius(self, y):
        return self.area(y)/self.wetP(y)
    
    def delta_Yn(self, y, discharge):
        return ((self.area(y)**(5/3))/(self.wetP(y)**(2/3))) - (discharge*self.n/(self.S0**(1/2)))
    
    def delta_Yc(self, y, discharge):
        g = 9.81
        return (g/discharge**2) - ((self.B + (self.m1 + self.m2)*y)/self.area(y)**3)

    def Yn(self, discharge):
        return Secant(self.delta_Yn, 1, 2, 0, 0.000001, 1000, discharge)

    def Yc(self, discharge):
        return Secant(self.delta_Yc, 1, 2, 0 , 0.000001, 1000, discharge)


class StandardStep():
    def __init__(self, discharge):
        self.Q = discharge #[m3/s]
    
    def velocity(self, y, channel):
        return self.Q/channel.area(y)
    
    def Sf(self, y, channel):
        return Sf_manning(channel.n, self.velocity(y, channel), channel.hydraulicRadius(y))
    
    def v_head(self, y, channel):
        g = 9.81 #gravity
        return self.velocity(y, channel)**2/(2*g)

    def total_head(self, y, channel):
        return y + self.v_head(y, channel)

    def Sf_avg(self, y, channel):
        return (self.Sf(y, channel) + self.df.iloc[-1, 0])/2

    def dSf(self, y, channel):
        return channel.S0 - self.Sf_avg(y, channel)

    def dH(self, y, channel):
        return self.total_head(y, channel) - self.df.iloc[-1, 1]

    def dx(self, y, channel):
        return self.dH(y, channel)/self.dSf(y, channel)

    def analyze(self, channel_array, y_bound, y_base):
        self.df = pd.DataFrame(
            {
                "Energy Slope"       : [self.Sf(y_bound, channel_array[0])],
                "Spesific Energy"    : [self.total_head(y_bound, channel_array[0])],
                "X Position"         : [channel_array[0].x],
                "Stream Depth (Y)"   : [y_bound],
                "Normal Depth (Yn)"  : [channel_array[0].Yn(self.Q)],
                "Critical Depth (Yc)": [channel_array[0].Yc(self.Q)],
                "Base Depth"         : [y_base]
            }
        )
        for index, channel in enumerate(channel_array[1:], start=1):
            target_dx = channel.x - self.df.iloc[-1, 2]
            y0 = self.df.iloc[-1, 3]
            y1 = (1+np.exp(-10))*y0
            y  = Secant(self.dx, y0, y1, target_dx, 0.0001, 1000, channel)
            new_state = {
                "Energy Slope"       : self.Sf(y, channel),
                "Spesific Energy"    : self.total_head(y, channel),
                "X Position"         : channel.x,
                "Stream Depth (Y)"   : y,
                "Normal Depth (Yn)"  : channel.Yn(self.Q),
                "Critical Depth (Yc)": channel.Yc(self.Q),
                "Base Depth"         : -target_dx*channel.S0 + self.df.iloc[-1, -1]
            }
            self.df = self.df.append(new_state, ignore_index=True)
        self.df['Y Position']  = self.df['Base Depth'] + self.df['Stream Depth (Y)']
        self.df['Yn Position'] = self.df['Base Depth'] + self.df['Normal Depth (Yn)']
        self.df['Yc Position'] = self.df['Base Depth'] + self.df['Critical Depth (Yc)']


def Calc_NormalCriticalDepth(discharge = None, base_width = None, left_slope = None, right_slope = None, manning = None, slope = None):
    print("            ***  Normal & Critical Depth - Water Profile Analysis  ***             ")
    print("                      coded by: Arif Yunando S - 2017410211                        ")
    print("  Applied Hydraulics - Civil Engineering Dept. - Parahyangan Catholic University   ")
    print("-----------------------------------------------------------------------------------") 
    if discharge == None:
        discharge   = float(input("     Discharge Rate           : "))
    else:
        print("     Discharge Rate           : {}".format(discharge))

    if base_width == None:
        base_width  = float(input("     Channel Base Width       : "))
    else:
        print("     Channel Base Width       : {}".format(base_width))

    if left_slope == None:
        left_slope  = float(input("     Left Slope               : "))
    else:
        print("     Left Slope               : {}".format(left_slope))

    if right_slope == None:
        right_slope = float(input("     Right Slope              : "))
    else:
        print("     Right Slope              : {}".format(right_slope))

    if manning == None:
        manning     = float(input("     Manning Coefficient      : "))
    else:
        print("     Manning Coefficient      : {}".format(manning))

    if slope == None:
        slope       = float(input("     Channel Slope            : "))
    else:
        print("     Channel Slope            : {}".format(slope))
    print("-----------------------------------------------------------------------------------") 
    section = channel(0, base_width, left_slope, right_slope, manning, slope)
    Yc, Yn = section.Yc(discharge), section.Yn(discharge)
    print("     Normal Depth (Yn)        : {:.3f}".format(Yn))
    print("     Critical Depth (Yc)      : {:.3f}".format(Yc))
    print("-----------------------------------------------------------------------------------") 
    return Yc, Yn



def uniformChannel(Discharge = None, Slope = None, Base = None, Manning = None, M1 = None, M2 = None, Boundary_Depth = None, Initial_x = None, delta_x = None, Final_x = None, BaseDepthAtBoundary = None, Direction = None):
    print("            ***  Standard Step Method - Water Profile Analysis  ***             ")
    print("                     coded by: Arif Yunando S - 2017410211                      ")
    print(" Applied Hydraulics - Civil Engineering Dept. - Parahyangan Catholic University ")
    print("--------------------------------------------------------------------------------")
    if Discharge == None:
        Q   = float(input("     Discharge Rate           : "))
    else:
        Q = Discharge
        print("     Discharge Rate           : {}".format(Q))

    if Slope == None:
        S   = float(input("     Channel Slope            : "))
    else:
        S = Slope
        print("     Channel Slope            : {}".format(S))
        
    if Base == None:
        B   = float(input("     Channel Base Width       : "))
    else:
        B = Base
        print("     Channel Base Width       : {}".format(B))

    if Manning == None:
        n   = float(input("     Manning Coefficient      : "))
    else:
        n = Manning
        print("     Manning Coefficient      : {}".format(n))

    if M1 == None:
        m1  = float(input("     Left Slope               : "))
    else:
        m1 = M1
        print("     Left Slope               : {}".format(m1))    

    if M2 == None:
        m2  = float(input("     Right Slope              : "))
    else:
        m2 = M2
        print("     Right Slope              : {}".format(m2))

    if Boundary_Depth == None:
        y   = float(input("     Water Depth at Boundary  : "))
    else:
        y = Boundary_Depth
        print("     Water Depth at Boundary  : {}".format(y))

    if Initial_x == None:
        x   = float(input("     Initial X Position       : ")) 
    else:
        x = Initial_x
        print("     Initial X Position       : {}".format(x)) 
 
    if delta_x == None:
        dx  = float(input("     Delta X Position         : ")) 
    else:
        dx = delta_x
        print("     Delta X Position         : {}".format(dx))
 
    if Final_x == None:
        x_f = float(input("     Final X position         : ")) 
    else:
        x_f = Final_x
        print("     Final X position         : {}".format(x_f))
         
    if BaseDepthAtBoundary == None:
        base= float(input("     Base Position at Boundary: ")) 
    else:
        base = BaseDepthAtBoundary
        print("     Base Position at Boundary: {}".format(base))
 
    if Direction == None:    
        dire = float(input("     Direction [1 (Upstream)/-1 (Downstream)] : ")) 
    else:
        dire = Direction
        string = "Upstream" if dire == 1 else "Downstream"
        print("     Direction                : {}".format(string))
    print("--------------------------------------------------------------------------------")

    channel_array = []
    for i in range(int((x_f - x)/dx)):
        #(self, location, base_width, left_slope, right_slope, manning, slope)
        channel_array.append(channel(x, B, m1, m2, n, S))
        if dire == 1: 
            x -= dx
        else:
            x += dx
    
    check = StandardStep(Q)
    check.analyze(channel_array, y, base)
    return check.df

def createPlot(result, fig_width = 15, fig_heigth = 5):
    result = result if isinstance(result, list) else [result]
    plt.figure(figsize=(fig_width, fig_heigth), dpi=80)
    for index, df_hasil in enumerate(result, start=1):
        df_hasil.set_index('X Position')['Base Depth'].plot(label = 'Base Depth', c='black')
        df_hasil.set_index('X Position')['Y Position'].plot(label = 'Water Profile {}'.format('#' + str(index)), c='navy')
        df_hasil.set_index('X Position')['Yn Position'].plot(label = 'Yn Position {}'.format('#' + str(index)), ls='-.', c='royalblue')
        df_hasil.set_index('X Position')['Yc Position'].plot(label = 'Yc Position {}'.format('#' + str(index)), ls='--', c='Crimson')
    plt.legend()