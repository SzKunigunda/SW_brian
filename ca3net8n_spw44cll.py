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

simulator_name = "neuron"
#simulator_name = get_script_args(1)[0]  
exec("from pyNN.%s import *" % simulator_name)
from sIN_4pyNN8e import sIN3_Type

from pyNN.random import NumpyRNG, RandomDistribution

# seed for random generator(s) used during simulation
kernelseed  = 43210987      

timer = Timer()

run_name = "ca3net8n_spw44cll"

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

gL_Pyr = 3.3333
tauMem_Pyr = 60.0
Cm_Pyr = tauMem_Pyr * gL_Pyr
Vrest_Pyr = -70.0
reset_Pyr = -60.0
theta_Pyr = -50.0
tref_Pyr = 5.0

gL_Bas = 7.1429
tauMem_Bas = 14.0
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -70.0
reset_Bas = -55.0
theta_Bas  = -45.0
tref_Bas = 2.0

gL_Sli = 4.0
tauMem_Sli = 40.0
Cm_Sli = tauMem_Sli * gL_Sli
Vrest_Sli = -70.0
reset_Sli = -55.0
theta_Sli  = -45.0
tref_Sli = 2.0

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
E_Exc = 0.0
E_Inh = -70.0

# Synaptic time constants
tauSyn_PyrExc = 10.0
tauSyn_PyrInh = 3.0
tauSyn_BasExc = 3.0
tauSyn_BasInh = 1.5
tauSyn_SliExc = 5.0
tauSyn_PyrSlowInh = 30.0

# Synaptic delays
delay_PyrExc = 3.0
delay_PyrInh = 1.5
delay_BasExc = 3.0
delay_BasInh = 1.5
delay_SliExc = 3.0
delay_PyrSlowInh = 2.0

p_rate_pyr = 1.0e-8
p_rate_mf = 0.8      # 3.0
p_rate_bas = 1.0e-8
p_rate_sli = 1.0e-8

cell_params_Pyr= {"c_m":        Cm_Pyr,
                    "tau_m":        tauMem_Pyr,
                    "v_rest":        Vrest_Pyr,
                    "e_e":       E_Exc,
                    "e_i":       E_Inh,
		    "e_sli":     E_Inh,
                    "tau_e": tauSyn_PyrExc,
                    "tau_i": tauSyn_PyrInh,
		    "tau_sli":  tauSyn_PyrSlowInh,
                    "t_refrac":      tref_Pyr,
                    "v_reset":    reset_Pyr,
                    "v_thresh":       theta_Pyr}

cell_params_Bas= {"cm":        Cm_Bas,
                    "tau_m":        tauMem_Bas,
                    "v_rest":        Vrest_Bas,
                    "e_rev_E":       E_Exc,
                    "e_rev_I":       E_Inh,
                    "tau_syn_E": tauSyn_BasExc,
                    "tau_syn_I": tauSyn_BasInh,
                    "tau_refrac":      tref_Bas,
                    "v_reset":    reset_Bas,
                    "v_thresh":       theta_Bas}

cell_params_Sli= {"cm":        Cm_Sli,
                    "tau_m":        tauMem_Sli,
                    "v_rest":        Vrest_Sli,
                    "e_rev_E":       E_Exc,
                    "e_rev_I":       E_Inh,
                    "tau_syn_E": tauSyn_SliExc,
                    "tau_syn_I": tauSyn_BasInh,
                    "tau_refrac":      tref_Sli,
                    "v_reset":    reset_Sli,
                    "v_thresh":       theta_Sli}
 
# === Build the network ========================================================

# clear all existing network elements and set resolution and limits on delays.
# For NEST, limits must be set BEFORE connecting any elements

#extra = {'threads' : 2}
extra = {}

rank = setup(timestep=dt, max_delay=maxdelay, **extra)
np = num_processes()
import socket
host_name = socket.gethostname()
print "Host #%d is on %s" % (rank+1, host_name)

if extra.has_key('threads'):
    print "%d Initialising the simulator with %d threads..." %(rank, extra['threads'])
else:
    print "%d Initialising the simulator with single thread..." %(rank)

# Small function to display information only on node 1
def nprint(s):
    if (rank == 0):
        print s

timer.start() # start timer on construction    

print "%d Setting up random number generator" %rank
rng = NumpyRNG(kernelseed)

vrest_sd = 5
vrest_distr = RandomDistribution('uniform', [Vrest_Pyr - numpy.sqrt(3)*vrest_sd, Vrest_Pyr + numpy.sqrt(3)*vrest_sd], rng)

Pyr_net = []

for sp in range(NP):
    print "%d Creating pyramidal cell population %d  with %d neurons." % (rank, sp, N_Pyr)
    Pyr_subnet = Population((N_Pyr,),sIN3_Type,cell_params_Pyr)
    for cell in Pyr_subnet:
	vrest_rval = vrest_distr.next(1)
	cell.set_parameters(v_rest=vrest_rval)
    Pyr_net.append(Pyr_subnet)

print "%d Creating basket cell population with %d neurons." % (rank, N_Bas)
Bas_net = Population((N_Bas,),IF_cond_exp,cell_params_Bas)

print "%d Creating slow inhibitory population with %d neurons." % (rank, N_Sli)
Sli_net = Population((N_Sli,),IF_cond_exp,cell_params_Sli)

#print "%d Initialising membrane potential to random values between %g mV and %g mV." % (rank, U0, theta)
#uniformDistr = RandomDistribution('uniform', [U0,theta], rng)
#E_net.randomInit(uniformDistr)
#I_net.randomInit(uniformDistr)

# Creating external input - is this correct on multiple threads?
Pyr_input = []

for sp in range(NP):
    print "%d Creating pyramidal cell external input %d with rate %g spikes/s." % (rank, sp, p_rate_pyr)
    pyr_poisson = Population((N_Pyr,), SpikeSourcePoisson, {'rate': p_rate_pyr})
    Pyr_input.append(pyr_poisson)

Pyr_input_MF = []

for sp in range(NP):
    print "%d Creating pyramidal cell external input %d with rate %g spikes/s." % (rank, sp, p_rate_mf)
    pyr_poisson = Population((N_Pyr,), SpikeSourcePoisson, {'rate': p_rate_mf})
    Pyr_input_MF.append(pyr_poisson)

print "%d Creating basket cell external input with rate %g spikes/s." % (rank, p_rate_bas)
bas_poisson = Population((N_Bas,), SpikeSourcePoisson, {'rate': p_rate_bas})

print "%d Creating slow inhibitory cell external input with rate %g spikes/s." % (rank, p_rate_sli)
sli_poisson = Population((N_Sli,), SpikeSourcePoisson, {'rate': p_rate_sli})

# Record spikes
for sp in range(NP):
    print "%d Setting up recording in excitatory population %d." % (rank, sp)
    Pyr_net[sp].record()
    Pyr_net[sp][[0, 1]].record_v()

print "%d Setting up recording in basket cell population." % rank
Bas_net.record()
Bas_net[[0, 1]].record_v()

print "%d Setting up recording in slow inhibitory population." % rank
Sli_net.record()
Sli_net[[0, 1]].record_v()

PyrExc_Connector = FixedProbabilityConnector(epsilon_Pyr, weights=J_PyrExc, delays=delay_PyrExc)
PyrPot_Connector = FixedProbabilityConnector(epsilon_Pyr, weights=J_PyrPot, delays=delay_PyrExc)
PyrInh_Connector = FixedProbabilityConnector(epsilon_Bas, weights=J_PyrInh, delays=delay_PyrInh)
BasExc_Connector = FixedProbabilityConnector(epsilon_Pyr, weights=J_BasExc, delays=delay_BasExc)
BasInh_Connector = FixedProbabilityConnector(epsilon_Bas, weights=J_BasInh, delays=delay_BasInh)
SliExc_Connector = FixedProbabilityConnector(epsilon_Pyr, weights=J_SliExc, delays=delay_SliExc)
PyrSlowInh_Connector = FixedProbabilityConnector(epsilon_Sli, weights=J_PyrSlowInh, delays=delay_PyrSlowInh)
PyrExt_Connector = OneToOneConnector(weights=J_PyrExt, delays=dt)
PyrMF_Connector = OneToOneConnector(weights=J_PyrMF, delays=dt)
BasExt_Connector = OneToOneConnector(weights=J_BasExt, delays=dt)
SliExt_Connector = OneToOneConnector(weights=J_SliExt, delays=dt)

Pyr_to_Pyr = []
Bas_to_Pyr = []
Sli_to_Pyr = []
Ext_to_Pyr = []
MF_to_Pyr = []

for post in range(NP):
    P2P_post = []
    Pyr_to_Pyr.append(P2P_post)
    for pre in range(NP):
	print "%d Connecting pyramidal cell subpopulations %d and %d." % (rank, pre, post)
	if pre == post:
	    P2P_sub = Projection(Pyr_net[pre], Pyr_net[post], PyrPot_Connector, rng=rng, target="excitatory")
	else:
	    P2P_sub = Projection(Pyr_net[pre], Pyr_net[post], PyrExc_Connector, rng=rng, target="excitatory")
	print "Pyr --> Pyr\t\t", len(P2P_sub), "connections"
	Pyr_to_Pyr[post].append(P2P_sub)
    B2P_sub = Projection(Bas_net, Pyr_net[post], PyrInh_Connector, rng=rng, target="inhibitory")
    print "Bas --> Pyr\t\t", len(B2P_sub), "connections"
    Bas_to_Pyr.append(B2P_sub)
    S2P_sub = Projection(Sli_net, Pyr_net[post], PyrSlowInh_Connector, rng=rng, target="slow_inhibitory")
    print "SlowInh --> Pyr\t\t", len(S2P_sub), "connections"
    Sli_to_Pyr.append(S2P_sub)
    E2P_sub = Projection(Pyr_input[post], Pyr_net[post], PyrExt_Connector, target="excitatory")
    print "Ext --> Pyr\t", len(E2P_sub), "connections"
    Ext_to_Pyr.append(E2P_sub)
    MF2P_sub = Projection(Pyr_input_MF[post], Pyr_net[post], PyrMF_Connector, target="excitatory")
    print "MF --> Pyr\t", len(MF2P_sub), "connections"
    MF_to_Pyr.append(MF2P_sub)

Pyr_to_Bas = []

print "%d Connecting basket cell population." % (rank)
for pre in range(NP):
    P2B_sub = Projection(Pyr_net[pre], Bas_net, BasExc_Connector, rng=rng, target="excitatory")
    print "Pyr --> Bas\t\t", len(P2B_sub), "connections"
    Pyr_to_Bas.append(P2B_sub)
Bas_to_Bas = Projection(Bas_net, Bas_net, BasInh_Connector, rng=rng, target="inhibitory")
print "Bas --> Bas\t\t", len(Bas_to_Bas), "connections"
Ext_to_Bas = Projection(bas_poisson, Bas_net, BasExt_Connector, target="excitatory")
print "Ext --> Bas\t", len(Ext_to_Bas), "connections"

Pyr_to_Sli = []

print "%d Connecting slow inhibitory population." % (rank)
for pre in range(NP):
    P2S_sub = Projection(Pyr_net[pre], Sli_net, SliExc_Connector, rng=rng, target="excitatory")
    print "Pyr --> SlowInh\t\t", len(P2S_sub), "connections"
    Pyr_to_Sli.append(P2S_sub)
Ext_to_Sli = Projection(sli_poisson, Sli_net, SliExt_Connector, target="excitatory")
print "Ext --> SlowInh\t", len(Ext_to_Sli), "connections"

# read out time used for building
buildCPUTime = timer.elapsedTime()
# === Run simulation ===========================================================

# run, measure computer time
timer.start() # start timer on construction
print "%d Running simulation for %g ms." % (rank, simtime)
run(simtime)
simCPUTime = timer.elapsedTime()

print "%d Writing data to file." % rank

filename_spike_pyr = []
filename_v_pyr = []
for sp in range(NP):
    filename_spike_sub = "Results/%s_pyr%d_np%d_%s.ras" % (run_name, sp, np, simulator_name) # output file for excit. population
    filename_spike_pyr.append(filename_spike_sub)
    filename_v_sub = "Results/%s_pyr%d_np%d_%s.v" % (run_name, sp, np, simulator_name) # output file for membrane potential traces
    filename_v_pyr.append(filename_v_sub)
filename_spike_bas  = "Results/%s_bas_np%d_%s.ras" % (run_name, np, simulator_name) # output file for basket cell population  
filename_spike_sli  = "Results/%s_sli_np%d_%s.ras" % (run_name, np, simulator_name) # output file for slow inhibitory population
filename_v_bas = "Results/%s_bas_np%d_%s.v"   % (run_name, np, simulator_name) # output file for membrane potential traces
filename_v_sli = "Results/%s_sli_np%d_%s.v"   % (run_name, np, simulator_name) # output file for membrane potential traces

# write data to file
for sp in range(NP):
    Pyr_net[sp].printSpikes(filename_spike_pyr[sp])
    #Pyr_net[sp].print_v(filename_v_pyr[sp])
Bas_net.printSpikes(filename_spike_bas)
#Bas_net.print_v(filename_v_bas)
Sli_net.printSpikes(filename_spike_sli)
#Sli_net.print_v(filename_v_sli)

Pyr_rate = 0
for sp in range(NP):
    Pyr_rate_sub = Pyr_net[sp].meanSpikeCount()*1000.0/simtime
    Pyr_rate = Pyr_rate + Pyr_rate_sub / NP
Bas_rate = Bas_net.meanSpikeCount()*1000.0/simtime
Sli_rate = Sli_net.meanSpikeCount()*1000.0/simtime

# write a short report
nprint("\n--- CA3 Network Simulation ---")
nprint("Nodes              : %d" % np)
nprint("Number of Neurons  : %d" % N_neurons)
nprint("Number of Synapses : %d" % N_syn)
nprint("Small-amplitude Pyr input firing rate  : %g" % p_rate_pyr)
nprint("Large-amplitude Pyr input firing rate  : %g" % p_rate_mf)
nprint("Bas input firing rate  : %g" % p_rate_bas)
nprint("SlowInh input firing rate  : %g" % p_rate_sli)
nprint("Pyr -> Pyr weight  : %g" % J_PyrExc)
nprint("Pyr -> Pyr potentiated weight  : %g" % J_PyrPot)
nprint("Pyr -> Bas weight  : %g" % J_BasExc)
nprint("Bas -> Pyr weight  : %g" % J_PyrInh)
nprint("Bas -> Bas weight  : %g" % J_BasInh)
nprint("Pyr -> SlowInh weight  : %g" % J_SliExc)
nprint("SlowInh -> Pyr weight  : %g" % J_PyrSlowInh)
nprint("Pyramidal cell rate    : %g Hz" % Pyr_rate)
nprint("Basket cell rate    : %g Hz" % Bas_rate)
nprint("Slow inhibitory rate    : %g Hz" % Sli_rate)
nprint("Build time         : %g s" % buildCPUTime)   
nprint("Simulation time    : %g s" % simCPUTime)
  
# === Clean up and quit ========================================================

end()
