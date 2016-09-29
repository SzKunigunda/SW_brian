# -*- coding: iso-8859-1 -*-
"""
Conversion of the Brunel network implemented in nest-1.0.13/examples/brunel.sli
to use PyNN.

Brunel N (2000) Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons.
J Comput Neurosci 8:183-208

Andrew Davison, UNIC, CNRS
May 2006

$Id: brunel.py 705 2010-02-03 16:06:52Z apdavison $

"""

from pyNN.utility import get_script_args, Timer
import numpy as np
import matplotlib.pyplot as plt
simulator_name = "brian"
exec("from %s import *" %simulator_name)
#simulator_name = get_script_args(1)[0]  
#exec("from pyNN.%s import *" % simulator_name)
#from sIN_4pyNN8e import sIN3_Type
from pyNN.random import NumpyRNG, RandomDistribution

# seed for random generator(s) used during simulation
kernelseed  = 43210987      

timer = Timer()

run_name = "ca3net8n_spw44cll_brian"

dt      = 0.1    # the resolution in ms
simtime = 200000.0 # Simulation time in ms
maxdelay   = 4.0    # synaptic delay in ms

epsilon_Pyr = 0.1  # connection probability
epsilon_Bas = 0.25
epsilon_Sli = 0.1

N_Pyr_all = 400 #7200
N_Bas =50 #400
N_Sli =50 #400
N_neurons = N_Pyr_all + N_Bas + N_Sli

NP = 10   # number of excitatory subpopulations
N_Pyr = int(N_Pyr_all / NP) # number of pyramidal cells in each subpopulation

# N_rec      = 50 # record from 50 neurons

C_Exc  = int(epsilon_Pyr*N_Pyr_all) # number of excitatory synapses per neuron
C_Inh  = int(epsilon_Bas*N_Bas) # number of fast inhibitory synapses per neuron
C_SlowInh = int(epsilon_Sli*N_Sli) # number of slow inhibitory synapses per neuron
#C_tot = C_Inh+C_Exc      # total number of synapses per neuron

# number of synapses---just so we know
N_syn = (C_Exc+1)*N_neurons + C_Inh*(N_Pyr+N_Bas) + C_SlowInh*N_Pyr

# Initialize the parameters of the integrate and fire neurons

z = 1*nS #?
gL_Pyr = 3.3333*uS
tauMem_Pyr = 60.0*ms
Cm_Pyr = tauMem_Pyr * gL_Pyr
Vrest_Pyr = -70.0*mV
reset_Pyr = -60.0*mV
theta_Pyr = -50.0*mV
tref_Pyr = 5.0*ms

# Adaptation parameters for pyr cells
a_Pyr = -0.8*nS  # nS    Subthreshold adaptation conductance
b_Pyr = 0.04*nA  # nA    Spike-triggered adaptation
delta_T_Pyr = 2.0*mV  # Slope factor
tau_w_Pyr = 300*ms  # Adaptation time constant
v_spike_Pyr = theta_Pyr + 10 * delta_T_Pyr

gL_Bas = 7.1429*uS
tauMem_Bas = 14.0*ms
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -70.0*mV
reset_Bas = -55.0*mV
theta_Bas  = -45.0*mV
tref_Bas = 2.0*ms

gL_Sli = 4.0*uS
tauMem_Sli = 40.0*ms
Cm_Sli = tauMem_Sli * gL_Sli
Vrest_Sli = -70.0*mV
reset_Sli = -55.0*mV
theta_Sli  = -45.0*mV
tref_Sli = 2.0*ms

# Initialize the synaptic parameters
J_PyrExc  = 0.15
J_PyrInh  = 4.0
J_BasExc  = 1.5
J_BasInh  = 2.0
J_PyrSlowInh = 3.0
J_SliExc = 0.1
J_PyrExt = 0.3
J_PyrMF = 5.0
J_BasExt = 0.3
J_SliExt = 0.3

# Potentiated synapses
dpot = 10.0       # degree of potentiation
J_PyrPot = (1 + dpot) * J_PyrExc

# Synaptic reversal potentials
E_Exc = 0.0*mV
E_Inh = -70.0*mV

# Synaptic time constants
tauSyn_PyrExc = 10.0*ms
tauSyn_PyrInh = 3.0*ms
tauSyn_BasExc = 3.0*ms
tauSyn_BasInh = 1.5*ms
tauSyn_SliExc = 5.0*ms
tauSyn_PyrSlowInh = 30.0*ms 

# Synaptic delays
delay_PyrExc = 3.0*ms
delay_PyrInh = 1.5*ms
delay_BasExc = 3.0*ms
delay_BasInh = 1.5*ms
delay_SliExc = 3.0*ms
delay_PyrSlowInh = 2.0*ms

p_rate_pyr = 1.0e-8*Hz
p_rate_mf = 0.8*Hz      # 3.0
p_rate_bas = 1.0e-8*Hz
p_rate_sli = 1.0e-8*Hz



# cell equations

eqs_adexp = '''
dvm/dt = (-gL_Pyr*(vm-Vrest_Pyr) + gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr)-w - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Pyr : volt
dw/dt = (a_Pyr*(vm- Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = -g_ampa/tauSyn_PyrExc : 1
dg_gaba/dt = -g_gaba/tauSyn_PyrInh : 1
'''

reset_adexp = '''
vm = reset_Pyr
w += b_Pyr
'''

eqs_bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas) - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Bas :volt
dg_ampa/dt = -g_ampa/tauSyn_BasExc : 1
dg_gaba/dt = -g_gaba/tauSyn_BasInh : 1
'''

eqs_sli = '''
dvm/dt = (-gL_Sli*(vm-Vrest_Sli) - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Sli :volt
dg_ampa/dt = -g_ampa/tauSyn_SliExc : 1
dg_gaba/dt = -g_gaba/tauSyn_PyrSlowInh : 1
'''

def myresetfunc(P, spikes):
    P.vm[spikes] = reset_Pyr   # reset voltage
    P.w[spikes] += b_Pyr  # low pass filter of spikes (adaptation mechanism)


SCR = SimpleCustomRefractoriness(myresetfunc, tref_Pyr, state='vm')

# === Build the network ========================================================

# clear all existing network elements and set resolution and limits on delays.
# For NEST, limits must be set BEFORE connecting any elements

##extra = {'threads' : 2}
#extra = {}

#rank = setup(timestep=dt, max_delay=maxdelay, **extra)
#num = num_processes()
#import socket
#host_name = socket.gethostname()
#print "Host #%d is on %s" % (rank+1, host_name)

#if extra.has_key('threads'):
#    print "%d Initialising the simulator with %d threads..." %(rank, extra['threads'])
#else:
#    print "%d Initialising the simulator with single thread..." %(rank)

## Small function to display information only on node 1
#def nprint(s):
#    if (rank == 0):
#        print s

#timer.start() # start timer on construction    

#print "%d Setting up random number generator" %rank
#rng = NumpyRNG(kernelseed)

#vrest_sd = 5
#vrest_distr = RandomDistribution('uniform', [Vrest_Pyr - np.sqrt(3)*vrest_sd, Vrest_Pyr + np.sqrt(3)*vrest_sd], rng)

Pyr_net = []

for sp in range(NP):
    print "Creating pyramidal cell population %d  with %d neurons." % (sp, N_Pyr)
    Pyr_subnet = NeuronGroup(N_Pyr, model=eqs_adexp, threshold=v_spike_Pyr, reset=SCR)
    Pyr_subnet.vm = Vrest_Pyr
    Pyr_subnet.g_ampa = 0
    Pyr_subnet.g_gaba = 0

    Pyr_net.append(Pyr_subnet)

print "Creating basket cell population with %d neurons." % (N_Bas)

Bas_net = NeuronGroup(N_Bas, model=eqs_bas, threshold=theta_Bas, reset=reset_Bas, refractory=tref_Bas)

Bas_net.vm  = Vrest_Bas
Bas_net.g_ampa = 0
Bas_net.g_gaba = 0

print "Creating slow inhibitory population with %d neurons." % (N_Sli)
Sli_net = NeuronGroup(N_Sli, model=eqs_sli, threshold=theta_Sli, reset=reset_Sli, refractory=tref_Sli)



print "Creating external inputs"
Pyr_input = []
for sp in range(NP):
    print "Creating pyramidal cell external input %d with rate %g spikes/s." % (sp, p_rate_pyr)
    PP = PoissonGroup(N_Pyr, p_rate_pyr)
    Pyr_input.append(PP)

Pyr_input_MF = []

for sp in range(NP):
    print "Creating pyramidal cell external input %d with rate %g spikes/s." % (sp, p_rate_mf)
    MF = PoissonGroup(N_Pyr, p_rate_mf)
    Pyr_input_MF.append(MF)



print "Creating basket cell external input with rate %g spikes/s." % (p_rate_bas)
BP = PoissonGroup(N_Bas, p_rate_bas)

print "Creating slow inhibitory cell external input with rate %g spikes/s." % (p_rate_sli)
PS = PoissonGroup(N_Sli, p_rate_sli) 


# Creating external input - is this correct on multiple threads?

## Record spikes
#for sp in range(NP):
#    print "%d Setting up recording in excitatory population %d." % (rank, sp)
#    Pyr_net[sp].record()
#    Pyr_net[sp][[0, 1]].record_v()

#print "%d Setting up recording in basket cell population." % rank
#Bas_net.record()
#Bas_net[[0, 1]].record_v()

#print "%d Setting up recording in slow inhibitory population." % rank
#Sli_net.record()
#Sli_net[[0, 1]].record_v()

print 'Connecting the network'

Pyr_to_Pyr = []
Bas_to_Pyr = []
Sli_to_Pyr = []
Ext_to_Pyr = []
MF_to_Pyr = []

for post in range(NP):
    P2P_post = []
    Pyr_to_Pyr.append(P2P_post)
    for pre in range(NP):
	print "Connecting pyramidal cell subpopulations %d and %d." % (pre, post)
	if pre == post:
	    P2P_sub = Connection(Pyr_net[pre], Pyr_net[post], 'g_ampa',weight=J_PyrPot, sparsness=epsilon_Pyr,delay=delay_PyrExc)
	else:
	    P2P_sub = Connection(Pyr_net[pre], Pyr_net[post], 'g_ampa', weight=J_PyrExc, sparsness=epsilon_Pyr,delay=delay_PyrExc)
	print "Pyr --> Pyr\t\t"#, len(P2P_sub), "connections"
	Pyr_to_Pyr[post].append(P2P_sub)
    B2P_sub = Connection(Bas_net, Pyr_net[post], 'g_gaba', weight=J_PyrInh, sparsness=epsilon_Bas,delay=delay_PyrInh)
    print "Bas --> Pyr\t\t"#, len(B2P_sub), "connections"
    Bas_to_Pyr.append(B2P_sub)
    S2P_sub = Connection(Sli_net, Pyr_net[post], 'g_ampa', weight=J_PyrSlowInh, sparsness=epsilon_Sli,delay=delay_PyrSlowInh) # g_ampa??
    print "SlowInh --> Pyr\t\t"#, len(S2P_sub), "connections"
    Sli_to_Pyr.append(S2P_sub)
    E2P_sub = Connection(Pyr_input[post], Pyr_net[post], 'g_ampa', weight=J_PyrExt,delay=dt)
    E2P_sub.connect_one_to_one(Pyr_input[post],Pyr_net[post]) #?
    print "Ext --> Pyr\t"#, len(E2P_sub), "connections"
    Ext_to_Pyr.append(E2P_sub)
    MF2P_sub = Connection(Pyr_input_MF[post], Pyr_net[post], 'g_ampa',weight=J_PyrMF, delay=dt)
    MF2P_sub.connect_one_to_one(Pyr_input_MF[post], Pyr_net[post])
    print "MF --> Pyr\t"#, len(MF2P_sub), "connections"
    MF_to_Pyr.append(MF2P_sub) #? elmenteni melyiket kell??

Pyr_to_Bas = []

print "Connecting basket cell population."
for pre in range(NP):
    P2B_sub = Connection(Pyr_net[pre], Bas_net, 'g_ampa', weight=J_BasExc,sparsness=epsilon_Pyr, delay=delay_BasExc)
    print "Pyr --> Bas\t\t"#, len(P2B_sub), "connections"
    Pyr_to_Bas.append(P2B_sub)
Bas_to_Bas = Connection(Bas_net, Bas_net, 'g_gaba',weight=J_BasInh,sparsness=epsilon_Bas,delay=delay_BasInh)
print "Bas --> Bas\t\t"#, len(Bas_to_Bas), "connections"
Ext_to_Bas = Connection(BP, Bas_net,'g_ampa', weight=J_BasExt,delay=dt)
Ext_to_Bas.connect_one_to_one(BP, Bas_net)
print "Ext --> Bas\t"#, len(Ext_to_Bas), "connections"

Pyr_to_Sli = []

print "Connecting slow inhibitory population."
for pre in range(NP):
    P2S_sub = Connection(Pyr_net[pre], Sli_net, 'g_ampa', weight=J_SliExc,sparsness=epsilon_Pyr, delay=delay_SliExc)
    print "Pyr --> SlowInh\t\t"#, len(P2S_sub), "connections"
    Pyr_to_Sli.append(P2S_sub)
Ext_to_Sli = Connection(PS, Sli_net, 'g_ampa',weight=J_SliExt, delay=dt)
Ext_to_Sli.connect_one_to_one(PS, Sli_net)
print "Ext --> SlowInh\t"#, len(Ext_to_Sli), "connections"

#read out time used for building
buildCPUTime = timer.elapsedTime()
#=== Run simulation ===========================================================

# #run, measure computer time
#timer.start()  start timer on construction
#print "Running simulation for %g ms." % (simtime)
#run(simtime)
#simCPUTime = timer.elapsedTime()


# Monitors

sme = []
for sp in range(NP):
     print "Monitoring excitatory population %d" %sp
     sme.append(SpikeMonitor(Pyr_net[sp]))

print "Monitoring basket cell population."
smi= SpikeMonitor(Bas_net)


print "Monitoring slow inhibitory population." 
sms=SpikeMonitor(Sli_net)

print "Writing data to file." 

filename_spike_pyr = []
filename_v_pyr = []
for sp in range(NP):
    filename_spike_sub = "Results/%s_pyr%d_np_%s.ras" % (run_name, sp, simulator_name) # output file for excit. population
    filename_spike_pyr.append(filename_spike_sub)
    filename_v_sub = "Results/%s_pyr%d_np_%s.v" % (run_name, sp, simulator_name) # output file for membrane potential traces
    filename_v_pyr.append(filename_v_sub)
filename_spike_bas  = "Results/%s_bas_num_%s.ras" % (run_name, simulator_name) # output file for basket cell population  
filename_spike_sli  = "Results/%s_sli_np_%s.ras" % (run_name, simulator_name) # output file for slow inhibitory population
filename_v_bas = "Results/%s_bas_np_%s.v"   % (run_name, simulator_name) # output file for membrane potential traces
filename_v_sli = "Results/%s_sli_num_%s.v"   % (run_name, simulator_name) # output file for membrane potential traces

# write data to file
for sp in range(NP):
    np.savez(filename_spike_pyr[sp], spikes=sme[sp].spikes, spiketimes=sme[sp].spiketimes.values())
       
np.savez(filename_spike_bas,spikes=smi.spikes,spiketimes=smi.spiketimes.values())
np.savez(filename_spike_sli,spikes=sms.spikes,spiketimes=sms.spiketimes.values())


popre_all = []
for sp in range(NP):
    popre_all.append(PopulationRateMonitor(Pyr_net[sp], bin=1*ms))
popri = PopulationRateMonitor(Bas_net, bin=1*ms)
poprs= PopulationRateMonitor(Sli_net, bin=1*ms)



# write a short report
print("\n--- CA3 Network Simulation ---")
print("Number of Neurons  : %d" % N_neurons)
print("Number of Synapses : %d" % N_syn)
print("Small-amplitude Pyr input firing rate  : %g" % p_rate_pyr)
print("Large-amplitude Pyr input firing rate  : %g" % p_rate_mf)
print("Bas input firing rate  : %g" % p_rate_bas)
print("SlowInh input firing rate  : %g" % p_rate_sli)
print("Pyr -> Pyr weight  : %g" % J_PyrExc)
print("Pyr -> Pyr potentiated weight  : %g" % J_PyrPot)
print("Pyr -> Bas weight  : %g" % J_BasExc)
print("Bas -> Pyr weight  : %g" % J_PyrInh)
print("Bas -> Bas weight  : %g" % J_BasInh)
print("Pyr -> SlowInh weight  : %g" % J_SliExc)
print("SlowInh -> Pyr weight  : %g" % J_PyrSlowInh)
#print("Pyramidal cell rate    : %g Hz" % Pyr_rate)
#print("Basket cell rate    : %g Hz" % Bas_rate)
#print("Slow inhibitory rate    : %g Hz" % Sli_rate)
print("Build time         : %g s" % buildCPUTime)   
#print("Simulation time    : %g s" % simCPUTime)
  
# === Clean up and quit ========================================================
