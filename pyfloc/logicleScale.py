import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import data
from FlowCytometryTools import FCMeasurement

class LogicleScale(object):
    def __init__(self,data):
        """
        data: np.array
        y: risultato della logicle (decade)
        """
        self.S = data
        self.y = []
        self.p = None
        self.W = None
        self.r = None
        self.T = None
        self.M = None
        self.A = None 
        self.r = None
    def __call__(self):
        self.calculate_T_M_A_r()
        self.calculate_p_W()
        self.calculate_y()
    def calculate_T_M_A_r(self, percentile = 5.0):
        if np.sum(self.S < 0) >0:
            self.r = 1.0*np.percentile(self.S[self.S < 0], percentile)
        else:
            self.r = -10        
        self.T = np.max(self.S)
        #self.A = np.log10(np.abs(self.r)) verificare prima di cancellare
        self.A = 0
        self.M = np.log10(self.T)
    def calculate_p_W(self):
        self.W = (self.M - np.log10(self.T/abs(self.r)))/2.0 
        W_prov = 0
        p_prov = 0.51
        while W_prov < self.W:
            p_prov = p_prov + 0.5 
            W_prov = 2*p_prov*np.log10(p_prov)/(p_prov+1)
        P = np.linspace(p_prov-1, p_prov+1, num=1000)
        W1 = 2*P*np.log10(P)/(P+1)
        p_interp = interp1d(W1,P)
        self.p = float(p_interp(self.W))
    def calculate_y(self, T = None, M = None, A = None, p = None, W = None):
        if T is None:
            T = self.T
        if M is None:
            M = self.M
        if A is None:
            A = self.A
        if p is None:
            p = self.p
        if W is None:
            W = self.W
        y = np.linspace(-7,7,100000)
        S_prov = self.calculate_S(y=y, T=T, M=M, A=A, p=p, W=W)
        y_interp = interp1d(S_prov, y)
        self.y = y_interp(self.S)
        return self.y
    def plot_data(self, X = None, Y = None, xlabel = None, ylabel = None):
        if X is None:
            X = self.S
        if Y is None:
            Y = self.y
        fig = plt.figure()
        plt.scatter(X,Y, s = 0.1)
        plt.title('W = {}'.format(self.W))
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        return fig
    def show(self, pdf = None):
        fig_Sy = self.plot_data(self.S,self.y, 'data', 'decade')
        fig_yS = self.plot_data(self.y,self.S, 'decade', 'data')
        if pdf is not None:
            pdf.savefig(fig_Sy)
            pdf.savefig(fig_yS)
            pdf.close()
        else:
            plt.show()
    def calculate_S(self, y = None, T = None, M = None, A = None, p = None, W = None):
        if y is None:
            y = self.y
        if T is None:
            T = self.T
        if M is None:
            M = self.M
        if A is None:
            A = self.A
        if p is None:
            p = self.p
        if W is None:
            W = self.W
        y_pos = np.where(y>=W+A)[0]
        y_neg = np.where(y<W+A)[0]
        S = np.zeros(np.shape(y)) 
        S[y_pos] = T*(10**(-(M-W-A)))*(10**(y[y_pos]-W-A)-(p**2)*10**(-(y[y_pos]-W-A)/p)+(p**2)-1)
        S[y_neg] = -T*(10**(-(M-W-A)))*(10**(W+A-y[y_neg])-(p**2)*10**(-(W+A-y[y_neg])/p)+(p**2)-1)
        return S

if __name__ == '__main__':
    pdf = PdfPages('test_logicle.pdf')
    file_name = '/home/marco/Documents/T315/ILN_9_8_047.fcs'
    E = data.Experiment(file_name = file_name, mode = 'all')
    print (E)
    C = data.Collection()
    C.add_experiment(experiment = E, condition = 'case')
    C.compensate()
    database = C.get_data_features(['BV711-A', 'SSC-A'])
    
    data = database[:,0]
    data = data[np.isfinite(data)] - 4000
    print ("Data go from ", np.min(data), " to ", np.max(data))
    
    C = LogicleScale(data)
    C()
    print ("Creating figures...")
    C.show(pdf = pdf)
    #C.show()
