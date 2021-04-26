import ipywidgets as widgets
import skrf as rf
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display, clear_output

from skrf.media import Coaxial




class Section(widgets.VBox):
    CONDUCTIVITIES = {
        'Steel': 1.32e6, # 1.45e6 in Wikipedia
        'Copper': 5.8e7, #1/1.682e-8, # 5.8e7 # Annealed copper.  5.96e7 for pure Cu in Wikipedia
        'Copper_200deg': 34578645,  # Copper 200deg 
        'Silver': 6.3e7, #1/1.59e-8, #6.3e7 # Wikipedia
        'Aluminium': 1/2.65e-8, #3.8e7 # Wikipedia
    }
        
    def __init__(self, kind: str = 'Line', config: dict = None):
        '''
        Section object
        
        Parameters
        ----------
        kind : str
            'Line', 'Short' or 'Tee'
        config : dict
            Dictionnary containing the Section properties (optional)
            Keys are 'Dint', 'Dout', 'L', 'R' and 'sigma'
        '''
        width = '200px'
        self.kind = kind

        self.w_kind = widgets.Dropdown(
            value=self.kind,
            placeholder='Choose kind',
            options=['Line', 'Short', 'Tee'],
            description='Type',
            disabled=False,
            layout=widgets.Layout(width=width)
        )

        self.output = widgets.Output()
        # define the various widgets
        self.w_Dint = widgets.BoundedFloatText(description='Dint', value=140, min=20, max=300, 
                                             layout=widgets.Layout(width=width))
        self.w_Dout= widgets.BoundedFloatText(description='Dout', value=230, min=100, max=400, 
                                            layout=widgets.Layout(width=width))
        self.w_L = widgets.BoundedFloatText(description='L', min=0, max=3000, value=100, 
                                     layout=widgets.Layout(width=width))
        self.w_sigma = widgets.Dropdown(
            value = 'Copper',
            options = ['Copper', 'Silver', 'Aluminium', 'Steel'],
            description = 'Metal',
            disabled = False,
            layout=widgets.Layout(width=width)
        )
        self.w_R = widgets.FloatText(description='R', value=1e-3, 
                                     layout=widgets.Layout(width=width))

        # set widget properties if passed
        if config:
            for key in config.keys():
                if key == 'Dint':
                    self.w_Dint.value = config[key]
                elif key == 'Dout':
                    self.w_Dout.value = config[key]
                elif key in ('L', 'Lengh'):
                    self.w_L.value = config[key]
                elif key == 'R':
                    self.w_R.value = config[key]
                elif key == 'sigma':
                    self.w_sigma.value = config[key]
        
        super().__init__(children=self.UI, 
                         layout=widgets.Layout(border='solid', max_width=width))
        
        self.w_kind.observe(self.__on_kind_change, 'value')

        
    def __on_kind_change(self, change):
        self.kind = change['new']
        self.__instantiate_ui()

    @property
    def UI(self):
        # instantiate the new kind widget 
        if self.kind == 'Line':
            UI = [self.w_kind, self.w_Dint, self.w_Dout, self.w_L, self.w_sigma]
        elif self.kind == 'Short':
            UI = [self.w_kind, self.w_Dint, self.w_Dout, self.w_R]
        elif self.kind == 'Tee':
            UI = [self.w_kind, self.w_Dint, self.w_Dout, self.w_sigma]
        return UI
    
    def __instantiate_ui(self):
        self.children = self.UI
        display(self)

    def to_dict(self):
        '''
        Return a dictionnary version of the Section configuration
        '''
        if self.kind == 'Line':
            return {
                'kind': self.kind,
                'Dint': self.w_Dint.value,
                'Dout': self.w_Dout.value,
                'Length': self.w_L.value,
                'sigma': self.w_sigma.value,
            }
        elif self.kind == 'Short':
            return {
                'kind': self.kind,
                'Dint': self.w_Dint.value,
                'Dout': self.w_Dout.value,
                'R': self.w_R.value
            }
        elif self.kind == 'Tee':
            return {
                'kind': self.kind,
                'Dint': self.w_Dint.value,
                'Dout': self.w_Dout.value,
                'sigma': self.w_sigma.value,
            }

    

    @classmethod
    def sigma_2_value(cls, sigma: str) -> float:
        '''
        Return the electrical conductivity of the given metal
        
        Parameter
        ---------
        sigma: str
            Metal type
        
        Return
        ------
        sigma_value: float
            Electrical Conductivity in S/m
        '''
        try: 
            return cls.CONDUCTIVITIES[sigma]
        except KeyError as e:
            raise ValueError('Uncorrect sigma description')

    def media(self, frequency):
        '''
        Return the Section skrf Media
        
        Parameter
        ---------
        frequency: skrf.Frequency
            Frequency to evaluate the Network on
        
        Return
        ------
        media : skrf.Media
            Coaxial Media of the Section
        '''
        return Coaxial(frequency=frequency, 
                        Dint=self.w_Dint.value*1e-3, 
                        Dout=self.w_Dout.value*1e-3, 
                        epsilon_r=1, 
                        sigma=self.sigma_2_value(self.w_sigma.value)
                       )
        
        
    def to_network(self, frequency):
        '''
        Return the Section scikit-rf Network
        
        Parameters
        ----------
        frequency: skrf.Frequency
            Frequency to evaluate the Network on
        
        Return
        ------
        ntwk: skrf.Network
            Section's resulting Network

        '''
        media = self.media(frequency)
        
        if self.kind == 'Line':
            return media.line(self.w_L.value, unit='mm', name=f'Line_{self.w_L.value}')
        elif self.kind == 'Tee':
            return media.tee(name='tee')
        elif self.kind == 'Short':
            return media.resistor(self.w_R.value, name=f'Short_R_{self.w_R.value}') ** media.short()
        else:
            raise ValueError('Incorrect kind type: ', self.kind)


#########################################################################
class ResonatorBuilder(widgets.VBox):
    def __init__(self):
        # initial configuration
        self._config = [Section('Short'), 
                        Section('Line', config={'L': 2700}), 
                        Section('Tee'), 
                        Section('Line', config={'L': 2750}), 
                        Section('Short')]
        
        self._config = [  # T-Resonator SSA-50
            Section('Short', config={'Dint': 128, 'Dout': 219, 'R': 3.5e-3}),
            Section('Line', config={'Dint': 128, 'Dout': 219, 'L': 33, 'sigma':'Copper'}),
            Section('Line', config={'Dint': 168, 'Dout': 230, 'L': 1100, 'sigma':'Copper'}),
            Section('Line', config={'Dint': 140, 'Dout': 230, 'L': 1021, 'sigma':'Copper'}),
            Section('Line', config={'Dint': 100, 'Dout': 230, 'L': 100, 'sigma':'Silver'}),
            Section('Line', config={'Dint': 140, 'Dout': 230, 'L': 114, 'sigma':'Silver'}),
            Section('Tee',  config={'Dint': 140, 'Dout': 230, 'sigma':'Silver'}),
            Section('Line', config={'Dint': 140, 'Dout': 230, 'L': 728, 'sigma':'Silver'}),
            Section('Line', config={'Dint': 100, 'Dout': 230, 'L': 100, 'sigma':'Silver'}),
            Section('Line', config={'Dint': 140, 'Dout': 230, 'L': 1512, 'sigma':'Steel'}),
            Section('Line', config={'Dint': 140, 'Dout': 219, 'L': 32, 'sigma':'Steel'}),
            Section('Short', config={'Dint': 140, 'Dout': 219, 'R': 7e-3}),
            ]      
        
        # define + and - buttons
        self.w_add_left = widgets.Button(icon="plus-square", layout=widgets.Layout(width='30px'))
        self.w_add_right = widgets.Button(icon="plus-square", layout=widgets.Layout(width='30px'))
        self.w_del_left = widgets.Button(icon="minus-square", layout=widgets.Layout(width='30px'))
        self.w_del_right = widgets.Button(icon="minus-square", layout=widgets.Layout(width='30px'))
        # define callbacks for + and - buttons
        self.w_add_left.on_click(self.add_section_to_the_left)
        self.w_add_right.on_click(self.add_section_to_the_right)
        self.w_del_left.on_click(self.del_section_to_the_left)
        self.w_del_right.on_click(self.del_section_to_the_right)
        
        self.output = widgets.Output()
        
        self.w_plot = widgets.Button(description='plot')
        self.w_plot.on_click(self.update_plot)
        self.__update_display()
        super().__init__(children=self.UI)
        
        self.fig, self.ax = plt.subplots()
        

    @property
    def UI(self):
        return [
            widgets.HBox([
                widgets.VBox([self.w_add_left, self.w_del_left]), 
                *self.config, 
                widgets.VBox([self.w_add_right, self.w_del_right]),
            ]),
            self.w_plot,
            self.output,
        ]
        
    def add_section_to_the_left(self, change):
        self.config.insert(1, Section())
        self.__update_display()

    def add_section_to_the_right(self, change):
        self.config.insert(-2, Section())
        self.__update_display()
            
    def del_section_to_the_left(self, change):
        self.config.pop(0)
        self.__update_display()
            
    def del_section_to_the_right(self, change):
        self.config.pop()
        self.__update_display()
            
    def __update_display(self):
        self.children = self.UI
        # update callback to trigger plot update when one change a widget
        for section in self.config:
            for child in section.children:
                child.observe(self.update_plot)

    @property
    def config(self) -> list:
        '''
        Get the UI configuration 
        '''
        return self._config

    @config.setter
    def config(self, cfg: list):
        '''
        Set the UI configuration
        '''
        self._config = cfg
        
    @property
    def configuration(self) -> list:
        '''
        Returns a simplified version of the configuration 
        '''
        configuration = [section.to_dict() for section in self.config]
        return configuration

    def medias_list(self, frequency) -> list:
        '''
        Return the list of Media which compose tje resonator
        
        Parameter
        ---------
        frequency : skrf.Frequency
            Frequency to evaluate the Network on
            
        Return
        ------
        medias: list of skrf.Media
            List of the Medias which compose the resonator        
        '''
        medias = [section.media(frequency) for section in self.config]
        return medias
    
    def networks_list(self, frequency) -> list:
        '''
        Return the list of Networks which compose the resonator
        
        Parameter
        ---------
        frequency : skrf.Frequency
            Frequency to evaluate the Network on
            
        Return
        ------
        networks: list of skrf.Network
            List of the Networks which compose the resonator
        '''
        networks = [section.to_network(frequency) for section in self.config]
        return networks

    def is_valid(self) -> bool:
        '''
        Check if the resonator configuration is valid
        
        Check if:
            - there is only one Tee
            - there are two Shorts
            - Short are at both ends
            - There is at least two Lines
            
        Return
        ------
        is_valid : bool
            True if the resonator is valid, False otherwise
        '''
        cfg = self.configuration
        is_valid = False
        kinds = []
        for section in cfg:
            kinds.append(section['kind'])
    
        if (kinds.count('Tee') == 1) and (kinds[0] == 'Short') and (kinds[-1] == 'Short') and (kinds.count('Short') == 2) and (kinds.count('Line') >= 2):
            is_valid = True
        
        return is_valid
    
    def to_network(self, frequency):
        '''
        Return the resonator Network for a given frequency
        
        Parameter
        ---------
        frequency : skrf.Frequency
            Frequency to evaluate the Network on
            
        Return
        ------
        network : krf.Network
            Resonator Network        
        '''
        
        if self.is_valid():
            # Calculate all Networks
            networks = self.networks_list(frequency)
                
            # traverse the list until we found the Tee
            for (idx, section) in enumerate(networks):
                if section.name == 'tee':
                    idx_tee = idx
            # split the resonator in two branches
            networks_branch_left = networks[:idx_tee]
            tee = networks[idx_tee]
            networks_branch_right = networks[idx_tee+1:]
            # caccade left and right branches
            with self.output:
                print(tee)
            branch_left = rf.cascade_list(networks_branch_left[-1::-1])  # cascade in reverse order to keep the short at the end
            branch_right = rf.cascade_list(networks_branch_right)
            # connect the tee
            resonator = rf.connect(rf.connect(branch_left, 0, tee, 1), 1, branch_right, 0)
            resonator.name = 'Resonator'
            return resonator
        else:
            with self.output:
                print('Resonator is not valid. Check its consistency !')
#             raise ValueError('Resonator is not valid. Check its consistency !')
            
    def update_plot(self, change, frequency=rf.Frequency(62, 63, unit='MHz', npoints=1001)):
        with self.output:
            if self.is_valid():
                plt.clf()  # avoid superposing curves
                ntwk = self.to_network(frequency)

                ax = plt.gca()
                ax = plt.subplot(3, 1, 1)
                ax.plot(ntwk.frequency.f_scaled, ntwk.s_db.squeeze())
                # find min and display
                f_match = ntwk.frequency.f_scaled[np.argmin(ntwk.s_mag)]
                ax.set_title(f'Match Freq: {f_match:0.2f} MHz')
                ax.set_ylim(-30, 1)
                
                f0 = 62.64
                frequency = rf.Frequency(f0, f0, unit='MHz', npoints=1)
                Pin = 50e3
                L_left, L_right, V_left, V_right, I_left, I_right = self.voltage_current(frequency, Pin)

                # Voltages                
                ax2 = plt.subplot(3, 1, 2)
                ax2.plot(-L_left, np.abs(V_left))
                ax2.plot(L_right, np.flipud(np.abs(V_right)))
                # ax2.set_ylim(0, 100)
                ax2.set_ylabel('V [kV]')
                # Currents  
                ax3 = plt.subplot(3, 1, 3)
                ax3.plot(-L_left, np.abs(I_left))
                ax3.plot(L_right, np.flipud(np.abs(I_right)))
                # ax3.set_ylim(0, 10)
                ax3.set_ylabel('I [kA]')                
                
                
                
    def voltage_current_branch(self, frequency, Zin, Zbranch, TL_indexes, R, P_in):
        '''
        Zin : float
            Input Impedance at the Tee input
        Zbranch : float
            Short Impedance at the end of the branch
        TL_indexes : list of int
            List of the indices
        R : float 
            Characteristic Impedance of the feeder
        P_in : float
            input power in [W]
            
        '''
        # spatial sampling 
        dl = 1e-3  # m 
        # Input voltage from input power and feeder impedance
        Vin = np.sqrt(P_in*2*R) # forward voltage
        rho_in = (Zin - R)/(Zin + R) # reflection coefficient
    
        # arrays of initial voltage and current for each line section
        V0, I0 = [], []
        # arrays along L
        V, I, Z, L_full = [], [], [], []
    
        # Going from T to short 
        V0.append( Vin*(1 + rho_in) ) # total voltage
        I0.append( V0[0]/Zbranch)
    
        # For each transmission line section,
        # propagates the V,I,Z from the last section values to the length of current section
        config = self.configuration
        medias = self.medias_list
        for TL_index in TL_indexes:
            # characteristic impedance and propagation constant of the current line
            Zc = medias(frequency)[TL_index].z0.squeeze().real
            gamma = medias(frequency)[TL_index].gamma.squeeze()
            with self.output:
                print(TL_index, Zc, gamma)
            _L = np.arange(start=0, stop=config[TL_index]['Length']*1e-3, step=dl)         
            for l in _L:
                _V, _I = transfer_matrix(-l, V0[-1], I0[-1], Zc, gamma)
                V.append(_V)
                I.append(_I)
                Z.append(_V/_I)
            # cumulate the length array    
            if L_full:
                L_full.append(_L + L_full[-1][-1])
            else:
                L_full.append(_L) # 1st step
            # section intersection values
            V0.append( V[-1] )
            I0.append( I[-1] )
    
        # convert list into arrays
        V = np.asarray(V)
        I = np.asarray(I)
        Z = np.asarray(Z)
        L = np.concatenate(np.asarray([np.asarray(_L) for _L in L_full]))
    
        return L, V, I, Z

    def voltage_current(self, frequency, P_in):
        '''
        

        Parameters
        ----------
        frequency : TYPE
            DESCRIPTION.
        P_in : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if self.is_valid():
            networks = self.networks_list(frequency)
            config = self.configuration
            
            # traverse the list until we found the Tee
            for (idx, section) in enumerate(networks):
                if section.name == 'tee':
                    idx_tee = idx
            
            # split the resonator in two branches (without the shorts)
            indexes_branch_left = list(range(1, idx_tee))[-1::-1]
            indexes_branch_right = list(range(idx_tee+1, len(networks)-1))
            
            ntwk = self.to_network(frequency)
            Zin = ntwk.z.squeeze()
            R = ntwk.z0.squeeze()
                        
            Zbranch_left = config[0]['R']
            Zbranch_right = config[-1]['R']
            
            
            L_left, V_left, I_left, Z_left = self.voltage_current_branch(frequency, Zin=Zin, Zbranch=Zbranch_left, 
                                                                       TL_indexes=indexes_branch_left, R=R, P_in=P_in)
            L_right, V_right, I_right, Z_right = self.voltage_current_branch(frequency, Zin=Zin, Zbranch=Zbranch_right, 
                                                                       TL_indexes=indexes_branch_right, R=R, P_in=P_in)
            
            return L_left, L_right, V_left, V_right, I_left, I_right
        else:
            print('Resonator configuration is not valid')
            return None

"""
Transmission Line helper functions
"""

def ZL_2_Zin(L,Z0,gamma,ZL):
    """
    Returns the input impedance seen through a lossy transmission line of
    characteristic impedance Z0 and complex wavenumber gamma=alpha+j*beta
             
    Zin = ZL_2_Zin(L,Z0,gamma,ZL)
    
    Args
    ----
    L : length [m] of the transmission line
    Z0: characteristic impedance of the transmission line
    gamma: complex wavenumber associated to the transmission line
    ZL: Load impedance

    Returns
    -------
    Zin: input impedance
    """
    
    assert L > 0
    assert Z0 > 0
    
    Zin = Z0*(ZL + Z0*np.tanh(gamma*L))/(Z0 + ZL*np.tanh(gamma*L))
    return Zin

def transfer_matrix(L,V0,I0,Z0,gamma):
    """
    Returns the voltage and the current at a distance L from an
    initial voltage V0 and current I0 on a transmission line which
    propagation constant is gamma.
     
    VL, IL = transfer_matrix(L,V0,I0,Z0,gamma)
     
    L is positive from the load toward the generator
      
    Args
    -----
    L  : transmission line length [m]
    V0: initial voltage [V]
    I0: initial current [A]
    Z0 : characteristic impedance of the transmission line
    gamma: =alpha+j*beta propagation constant of the transmission line
    
    Returns
    --------
    VL: voltage at length L
    IL: current at length L
    """
    if Z0 <= 0:
        raise ValueError
        
    transfer_matrix = np.array([[np.cosh(gamma*L), Z0*np.sinh(gamma*L)], 
                                [np.sinh(gamma*L)/Z0, np.cosh(gamma*L)]])
    U = np.array([V0,I0])
    A = transfer_matrix @ U           
    VL = A[0]
    IL = A[1]
    return VL, IL

def V0f_2_VL(L, V0f, gamma, reflection_coefficient):
    """
    Propagation of the voltage at a distance L from the forward
    voltage and reflection coefficient
     
    VL = V0f_2_VL(L, V0f, gamma, reflectionCoefficient)
     
    Args
    ----
    L : Transmission Line Length [m]
    V0f : forward voltage [V]
    gamma : Transmission Line Complex Propagatioon Constant [1]
    reflectionCoefficient : complex reflection coefficient [1]
    
    Returns
    --------
    VL : (total) voltage at length L 
    """ 
    assert L > 0
    assert gamma > 0
    assert reflection_coefficient > 0
    
    VL = V0f*(np.exp(-gamma*L) + reflection_coefficient*np.exp(+gamma*L))
    return VL       
    
    