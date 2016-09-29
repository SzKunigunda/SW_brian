# -*- coding: iso-8859-1 -*-
"""
Example of making a NEURON cell model compatible with PyNN

Szabolcs Kali, HAS, 16th November 2011

based on example by

Andrew Davison, UNIC, CNRS, 29th March 2010
"""

from neuron import h, hclass
from pyNN.neuron import *
from pyNN.neuron.cells import *

class sIN3(StandardIF):
    """
    PyNN-compatible cell class created by subclassing StandardIF
    from pyNN.neuron.cells.
    """
    conductance_based = True
    
    def __init__(self, **parameters):
        StandardIF.__init__(self, syn_type=parameters['syn_type'], syn_shape=parameters['syn_shape'],
                            c_m=parameters['c_m'], tau_m=parameters['tau_m'], v_rest=parameters['v_rest'],
			    v_thresh=parameters['v_thresh'], t_refrac=parameters['t_refrac'], i_offset=parameters['i_offset'], v_reset=parameters['v_reset'],
                            tau_e=parameters['tau_e'], tau_i=parameters['tau_i'], e_e=parameters['e_e'], e_i=parameters['e_i'])
        
	syn_type="conductance"
	syn_shape="alpha"
	synapse_model = StandardIF.synapse_models[syn_type][syn_shape]
	self.slisyn = synapse_model(0.5, sec=self)
	self.slisyn.tau = parameters['tau_sli']
	self.slisyn.e = parameters['e_sli']
		
	self.parameter_names.extend(['tau_sli', 'e_sli'])
	
        for name in self.parameter_names:
	    setattr(self,name,parameters[name])

    @property
    def slow_inhibitory(self):
	return self.slisyn

    def _get_tau_sli(self):
	return self.slisyn.tau
    def _set_tau_sli(self, value):
	self.slisyn.tau = value
    tau_sli = property(fget=_get_tau_sli, fset=_set_tau_sli)

    def _get_e_sli(self):
	return self.slisyn.e
    def _set_e_sli(self, value):
	self.slisyn.e = value
    e_sli = property(fget=_get_e_sli, fset=_set_e_sli)

class sIN3_Type(NativeCellType):
    default_parameters = {'syn_type':"conductance", 'syn_shape':"exp",
                          'c_m':1.0, 'tau_m':20.0, 'v_rest':-65.0, 'v_thresh':-55.0,
			  't_refrac':2.0, 'i_offset':0.0, 'v_reset':-60.0,
                          'tau_e':2.0, 'tau_i':2.0, 'tau_sli':100.0,
			  'e_e':0.0, 'e_i':-70.0, 'e_sli':-70.0}
    default_initial_values = {'v': -65.0}
    recordable = ['spikes', 'v']
    model = sIN3
