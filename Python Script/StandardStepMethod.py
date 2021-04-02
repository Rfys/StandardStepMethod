import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as poly

def Secant(f, x0, x1, target = 0, eps = 0.0001, maxiter = 1000, *args):
#  print("   ***  Numerical Analysis - Secant Method  ***   ")
#  print("--------------------------------------------------")
  xlama  = x0
  x      = x1
  y      = target
  error  = 1
  i      = 0
  while error > eps and i < maxiter:
    xbaru = x + (y - f(x, *args))*(xlama - x)/(f(xlama, *args) - f(x, *args))
    error = np.abs(xbaru - x)/np.abs(xbaru)
    xlama = x
    x     = xbaru
    i    += 1
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
        self.x  = location
        self.B  = base_width
        self.m1 = left_slope
        self.m2 = right_slope
        self.n  = manning
        self.S0 = slope
    
    def area(self, y):
        return (self.B + y*((self.m1+self.m2)/2))*y
    
    def centroid(self, y): #measure from water surface
        a = 3*self.B*y + (self.m1 + self.m2)*y**2
        b = 6*self.B   + (self.m1 + self.m2)*y*3
        return a/b

    def wetP(self, y):
        return self.B + y*np.sqrt(1+self.m1**2) + y*np.sqrt(1+self.m2**2)
    
    def hydraulicRadius(self, y):
        return self.area(y)/self.wetP(y)
    
    def froude(self, discharge):
        g  = 9.81
        yn = self.Yn(discharge)
        return (discharge/self.area(yn))/np.sqrt(g*yn)
    
    def momentum(self, y, discharge):
        g = 9.81
        return self.area(y)*self.centroid(y) + discharge**2/(g*self.area(y))

    def energy(self, y, discharge):
        g = 9.81
        return y + (discharge/self.area(y))**2/(2*g)

    def delta_Momentum(self, y1, y2, discharge):
        M1 = self.momentum(y1, discharge)
        M2 = self.momentum(y2, discharge)
        return abs(M1 - M2)
    
    def delta_Energy(self, y1, y2, discharge):
        E1 = self.energy(y1, discharge)
        E2 = self.energy(y2, discharge)
        return abs(E1 - E2)

    def delta_Yn(self, y, discharge):
        return ((self.area(y)**(5/3))/(self.wetP(y)**(2/3))) - (discharge*self.n/(self.S0**(1/2)))
    
    def delta_Yc(self, y, discharge):
        g = 9.81
        return (g/discharge**2) - ((self.B + (self.m1 + self.m2)*y)/self.area(y)**3)
    
    def alternate(self, y, discharge):
        x = 0.1*y if y > self.Yc(discharge) else 2*y
        return Secant(self.delta_Energy, x, (1+np.exp(-10))*x, 0, 0.00001, 1000, y, discharge)
    
    def conjugate(self, y, discharge):
        x = 0.1*y if y > self.Yc(discharge) else 2*self.Yc(discharge)
        return Secant(self.delta_Momentum, x, (1+np.exp(-10))*x, 0, 0.00001, 1000, y, discharge)

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
                "Conjugate Depth"    : [channel_array[0].conjugate(y_bound, self.Q)],
                "Alternate Depth"    : [channel_array[0].alternate(y_bound, self.Q)],
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
                "Critical Depth (Yc)": channel.Yc(self.Q),
                "Normal Depth (Yn)"  : channel.Yn(self.Q),
                "Alternate Depth"    : channel.alternate(y, self.Q),
                "Conjugate Depth"    : channel.conjugate(y, self.Q),
                "Base Depth"         : - target_dx*channel.S0 + self.df.iloc[-1, -1]
            }
            self.df = self.df.append(new_state, ignore_index=True)
        self.df['Y Position'        ] = self.df['Base Depth'] + self.df['Stream Depth (Y)'   ]
        self.df['Yc Position'       ] = self.df['Base Depth'] + self.df['Critical Depth (Yc)']
        self.df['Yn Position'       ] = self.df['Base Depth'] + self.df['Normal Depth (Yn)'  ]
        self.df['Alternate Position'] = self.df['Base Depth'] + self.df['Alternate Depth'    ]
        self.df['Conjugate Position'] = self.df['Base Depth'] + self.df['Conjugate Depth'    ]


"""
Wrapper Functions :)
"""


def Calc_NormalCriticalDepth(discharge = None, base_width = None, left_slope = None, right_slope = None, manning = None, slope = None):
    print("            ***  Normal & Critical Depth - Water Profile Analysis  ***             ")
    print("                      coded by: Arif Yunando S - 2017410211                        ")
    print("  Applied Hydraulics - Civil Engineering Dept. - Parahyangan Catholic University   ")
    print("-----------------------------------------------------------------------------------")
    script = [
        "     Discharge Rate           :",
        "     Channel Base Width       :",
        "     Left Slope               :",
        "     Right Slope              :",
        "     Manning Coefficient      :",
        "     Channel Slope            :"
    ]
    discharge   = float(input(script[0])) if discharge   == None else discharge
    base_width  = float(input(script[1])) if base_width  == None else base_width 
    left_slope  = float(input(script[2])) if left_slope  == None else left_slope
    right_slope = float(input(script[3])) if right_slope == None else right_slope
    manning     = float(input(script[4])) if manning     == None else manning
    slope       = float(input(script[5])) if slope       == None else slope
    print(script[0], discharge)
    print(script[1], base_width)
    print(script[2], left_slope)
    print(script[3], right_slope)
    print(script[4], manning)
    print(script[5], slope)
    print("-----------------------------------------------------------------------------------") 
    section = channel(0, base_width, left_slope, right_slope, manning, slope)
    Yc, Yn, Fr  = section.Yc(discharge), section.Yn(discharge), section.froude(discharge)
    jn_sal   = "Saluran Landai" if Fr < 1 else "Saluran Curam"
    print("     Normal Depth   (Yn)      : {:.3f}".format(Yn))
    print("     Critical Depth (Yc)      : {:.3f}".format(Yc))
    print("     Froude Number  (Fr)      : {:.3f}".format(Fr))
    print("     Jenis Saluran            : {}".format(jn_sal))
    print("-----------------------------------------------------------------------------------") 
    return Yc, Yn



def uniformChannel(Discharge = None, Slope = None, Base = None, Manning = None, M1 = None, M2 = None, Boundary_Depth = None, Initial_x = None, delta_x = None, Distance = None, BaseDepthAtBoundary = None, Direction = None):
    print("            ***  Standard Step Method - Water Profile Analysis  ***             ")
    print("                     coded by: Arif Yunando S - 2017410211                      ")
    print(" Applied Hydraulics - Civil Engineering Dept. - Parahyangan Catholic University ")
    print("--------------------------------------------------------------------------------")
    script = [
        "     Discharge Rate           :",
        "     Channel Slope            :",
        "     Channel Base Width       :",
        "     Manning Coefficient      :",
        "     Left Slope               :",
        "     Right Slope              :",
        "     Water Depth at Boundary  :",
        "     Initial X Position       :",
        "     Delta X                  :",
        "     Distance                 :",
        "     Base Position at Boundary:",
        "     Direction [1 (Upstream)/-1 (Downstream)] :"
    ]
    Slope               = float(input(script[1]))  if Slope               == None else Slope
    Discharge           = float(input(script[0]))  if Discharge           == None else Discharge
    Base                = float(input(script[2]))  if Base                == None else Base
    Manning             = float(input(script[3]))  if Manning             == None else Manning
    M1                  = float(input(script[4]))  if M1                  == None else M1
    M2                  = float(input(script[5]))  if M2                  == None else M2
    Boundary_Depth      = float(input(script[6]))  if Boundary_Depth      == None else Boundary_Depth 
    Initial_x           = float(input(script[7]))  if Initial_x           == None else Initial_x 
    delta_x             = float(input(script[8]))  if delta_x             == None else delta_x 
    Distance            = float(input(script[9]))  if Distance            == None else Distance 
    BaseDepthAtBoundary = float(input(script[10])) if BaseDepthAtBoundary == None else BaseDepthAtBoundary 
    Direction           = float(input(script[11])) if Direction           == None else Direction 
    string              = "Upstream"               if Direction           == 1    else "Downstream"
    print(script[1], Slope)
    print(script[0], Discharge)
    print(script[2], Base)
    print(script[3], Manning)
    print(script[4], M1) 
    print(script[5], M2)
    print(script[6], Boundary_Depth)
    print(script[7], Initial_x)
    print(script[8], delta_x)
    print(script[9], Distance)
    print(script[10],BaseDepthAtBoundary)
    print(script[11],string)
    print("--------------------------------------------------------------------------------")
    section = channel(Initial_x, Base, M1, M2, Manning, Slope)
    Yc, Yn, Fr  = section.Yc(Discharge), section.Yn(Discharge), section.froude(Discharge)
    jn_sal   = "Saluran Landai" if Fr < 1 else "Saluran Curam"
    print("     Normal Depth   (Yn)      : {:.3f}".format(Yn))
    print("     Critical Depth (Yc)      : {:.3f}".format(Yc))
    print("     Froude Number  (Fr)      : {:.3f}".format(Fr))
    print("     Jenis Saluran            : {}".format(jn_sal))
    print("-----------------------------------------------------------------------------------") 

    channel_array = []
    iteration = Distance/delta_x
    for i in range(int(round(iteration, 0)) + 1):
        channel_array.append(channel(Initial_x, Base, M1, M2, Manning, Slope))
        Initial_x = Initial_x - delta_x if Direction == 1 else Initial_x + delta_x
    
    check = StandardStep(Discharge)
    check.analyze(channel_array, Boundary_Depth, BaseDepthAtBoundary)
    return check.df

def checkHydraulicJump(Upstream, Downstream):
    df1 = Upstream[Upstream['Conjugate Depth'] > 0]
    df1 = df1[df1['Normal Depth (Yn)'] > 0]
    df2 = Downstream[Downstream['Conjugate Depth'] > 0]
    df2 = df2[df2['Normal Depth (Yn)'] > 0]
    x1 = df1['X Position'].tolist()
    y1 = df1['Y Position'].tolist()
    x2 = df2['X Position'].tolist()
    y2 = df2['Y Position'].tolist()
    y3 = df2['Conjugate Position'].tolist()
    coef1 = poly.polyfit(x1, y1, 10)
    coef2 = poly.polyfit(x2, y2, 10)
    coef3 = poly.polyfit(x2, y3, 10)
    p1 = poly.Polynomial(coef1)
    p2 = poly.Polynomial(coef2)
    p3 = poly.Polynomial(coef3)
    def delta(x):
        return p1(x) - p3(x)
    xJump = Secant(delta, 1, 2)
    print("               ***  Hydraulic Jump - Water Profile Analysis  ***                ")
    print("                     coded by: Arif Yunando S - 2017410211                      ")
    print(" Applied Hydraulics - Civil Engineering Dept. - Parahyangan Catholic University ")
    print("--------------------------------------------------------------------------------")
    print("                  Loncat Air terjadi pada posisi X = {:.3f}".format(xJump))
    df = Upstream[Upstream['X Position'] < xJump].append(Downstream[Downstream['X Position'] > xJump], ignore_index=True)
    return df

def createPlot(result, fig_width = 15, fig_heigth = 5):
    result = result if isinstance(result, list) else [result]
    plt.figure(figsize=(fig_width, fig_heigth), dpi=200)
    df = result[0].set_index('X Position').sort_index()
    df['Base Depth' ].plot(label = 'Base Depth',       c='black')
    df['Yn Position'].plot(label = 'Yn Profile',       c='royalblue', ls='-.')
    df['Yc Position'].plot(label = 'Yc Profile',       c='Crimson',   ls='--')
    df['Y Position' ].plot(label = 'Water Profile #1', c='navy')
    if len(result) > 1:
        for index, df_hasil in enumerate(result[1:], start=2):
            df = df_hasil.set_index('X Position').sort_index()
            df['Base Depth' ].plot(c='black',     label='')
            df['Yn Position'].plot(c='royalblue', label='', ls='-.')
            df['Yc Position'].plot(c='Crimson',   label='', ls='--')
            df['Y Position' ].plot(label = 'Water Profile {}'.format('#' + str(index)))
    plt.legend()

def conjugatePlot(result):
    result.set_index('X Position')['Conjugate Position'].plot(label = 'Conjugate', c='grey', ls = ':')

def alternatePlot(result):
    result.set_index('X Position')['Alternate Position'].plot(label = 'Alternate', c='red',  ls = ':')