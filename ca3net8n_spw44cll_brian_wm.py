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
import numpy as np# -*- coding: iso-8859-1 -*-
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
#from pyNN.random import NumpyRNG, RandomDistribution
from detect_oscillations_15_50 import replay, ripple, gamma


# seed for random generator(s) used during simulation
kernelseed  = 43210987      

timer = Timer()
run_name = "ca3net8n_spw44cll_brian"


X = np.zeros((19,1))
dt      = 0.1    # the resolution in ms
simtime = 1000.0 # Simulation time in ms #10000
maxdelay   = 4.0    # synaptic delay in ms

epsilon_Pyr = 0.1  # connection probability
epsilon_Bas = 0.25
epsilon_Sli = 0.1

N_Pyr_all =3000 #3600 #7200
N_Bas =200 #200  #400
N_Sli = 200 #200 #400
N_neurons = N_Pyr_all + N_Bas + N_Sli

NP = 1   # number of excitatory subpopulations
N_Pyr = int(N_Pyr_all / NP) # number of pyramidal cells in each subpopulation

# N_rec      = 50 # record from 50 neurons

C_Exc  = int(epsilon_Pyr*N_Pyr_all) # number of excitatory synapses per neuron
C_Inh  = int(epsilon_Bas*N_Bas) # number of fast inhibitory synapses per neuron
C_SlowInh = int(epsilon_Sli*N_Sli) # number of slow inhibitory synapses per neuron
#C_tot = C_Inh+C_Exc      # total number of synapses per neuron

# number of synapses---just so we know
N_syn = (C_Exc+1)*N_neurons + C_Inh*(N_Pyr+N_Bas) + C_SlowInh*N_Pyr

# Initialize the parameters of the integrate and fire neurons

z = 1*uS # it gives unit to synaptic conductances (g_) #nS
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
J_PyrSlowInh = 5.0 #3.0 # slow->pyr
J_SliExc = 0.5  #0.1 # pyr->slow
J_PyrExt = 0.3
J_PyrMF = 5.0
J_BasExt = 0.3
J_SliExt = 0.3

# Potentiated synapses
dpot = 15.0 #10.0       # degree of potentiation
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
tauSyn_SliInh = 3.0*ms
tauSyn_PyrSlowInh = 30.0*ms 


# Synaptic delays
delay_PyrExc = 3.0*ms
delay_PyrInh = 1.5*ms
delay_BasExc = 3.0*ms
delay_BasInh = 1.5*ms
delay_SliExc = 3.0*ms
delay_PyrSlowInh = 2.0*ms

p_rate_pyr = 0.0*Hz  #1.0e-8*Hz
p_rate_mf = 3.0*Hz #5.0 #0.8
p_rate_bas = 0.0*Hz  #1.0e-8*Hz
p_rate_sli = 0.0*Hz  #1.0e-8*Hz

# cell equations

eqs_adexp = '''
dvm/dt = (-gL_Pyr*(vm-Vrest_Pyr) + gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr)-w - (g_ampa*z*(vm-E_Exc) + g_gaba_fast*z*(vm-E_Inh)+ g_gaba_slow*z*(vm-E_Inh)))/Cm_Pyr : volt
dw/dt = (a_Pyr*(vm- Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = -g_ampa/tauSyn_PyrExc : 1
dg_gaba_fast/dt = -g_gaba_fast/tauSyn_PyrInh : 1
dg_gaba_slow/dt = -g_gaba_slow/tauSyn_PyrSlowInh : 1
'''

# peldak ami alapjan lehet modositani
#input1 = PoissonGroup(200, rates=lambda t: (t < 200 * ms and 2000 * Hz) or 0 * Hz)
#eqs=Equations(’dx/dt=-x/tau : volt’,tau=10*ms)
# Pyr_subnet.Vrest_Pyr = rand(N_Pyr)*(upper-lower) + lower

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
dg_gaba/dt = -g_gaba/tauSyn_SliInh : 1
'''

def myresetfunc(P, spikes):
    P.vm[spikes] = reset_Pyr   # reset voltage
    P.w[spikes] += b_Pyr  # low pass filter of spikes (adaptation mechanism)


SCR = SimpleCustomRefractoriness(myresetfunc, tref_Pyr, state='vm')

# === Build the network ========================================================
#print "%d Setting up random number generator" %rank
#rng = NumpyRNG(kernelseed)


Pyr_net = []
vrest_sd = 5*mV
upper = Vrest_Pyr + np.sqrt(3)*vrest_sd
lower = Vrest_Pyr - np.sqrt(3)*vrest_sd

for sp in range(NP):
    print "Creating pyramidal cell population %d  with %d neurons." % (sp, N_Pyr)
    Pyr_subnet = NeuronGroup(N_Pyr, model=eqs_adexp, threshold=v_spike_Pyr, reset=SCR)
    Pyr_subnet.Vrest_Pyr = rand(N_Pyr)*(upper-lower) + lower # changing pyramidal resting potential to random values
    Pyr_subnet.w = 0
    Pyr_subnet.g_ampa = 0
    Pyr_subnet.g_gaba_fast = 0
    Pyr_subnet.g_gaba_slow = 0
    Pyr_subnet.vm = Vrest_Pyr

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
 
# egymastol fuggetlen Poisson bementek-e? van ilyen Brian példa- Brunel halozat

print 'Connecting the network'

Pyr_to_Pyr = []
Bas_to_Pyr = []
Sli_to_Pyr = []
Ext_to_Pyr = []
MF_to_Pyr = []


f = file('wmxR3000.txt', 'r')
Wee = [line.split() for line in f]

f.close()

for i in range(N_Pyr_all):
    Wee[i][:] = [float(x) * 1.e9 for x in Wee[i]]


for post in range(NP):
    print "Pyr --> Pyr\t\t"
    Pyr_to_Pyr = Connection(Pyr_net[post], Pyr_net[post], 'g_ampa', delay=delay_PyrExc)
    Pyr_to_Pyr.connect(Pyr_net[post], Pyr_net[post], Wee)

    B2P_sub = Connection(Bas_net, Pyr_net[post], 'g_gaba_fast', weight=J_PyrInh, sparseness=epsilon_Bas,delay=delay_PyrInh)
    print "Bas --> Pyr\t\t"
    Bas_to_Pyr.append(B2P_sub)
    S2P_sub = Connection(Sli_net, Pyr_net[post], 'g_gaba_slow', weight=J_PyrSlowInh, sparseness=epsilon_Sli,delay=delay_PyrSlowInh)
    print "SlowInh --> Pyr\t\t"
    Sli_to_Pyr.append(S2P_sub)
    E2P_sub = IdentityConnection(Pyr_input[post], Pyr_net[post], 'g_ampa', weight=J_PyrExt)
    print "Ext --> Pyr\t"
    Ext_to_Pyr.append(E2P_sub)
    MF2P_sub = IdentityConnection(Pyr_input_MF[post], Pyr_net[post], 'g_ampa',weight=J_PyrMF)
    print "MF --> Pyr\t"
    MF_to_Pyr.append(MF2P_sub)

Pyr_to_Bas = []

print "Connecting basket cell population."
for pre in range(NP):
    P2B_sub = Connection(Pyr_net[pre], Bas_net, 'g_ampa', weight=J_BasExc,sparseness=epsilon_Pyr, delay=delay_BasExc)
    print "Pyr --> Bas\t\t"
    Pyr_to_Bas.append(P2B_sub)
Bas_to_Bas = Connection(Bas_net, Bas_net, 'g_gaba',weight=J_BasInh,sparseness=epsilon_Bas,delay=delay_BasInh)
print "Bas --> Bas\t\t"
Ext_to_Bas = IdentityConnection(BP, Bas_net,'g_ampa', weight=J_BasExt)
print "Ext --> Bas\t"

Pyr_to_Sli = []

print "Connecting slow inhibitory population."
for pre in range(NP):
    P2S_sub = Connection(Pyr_net[pre], Sli_net, 'g_ampa', weight=J_SliExc,sparseness=epsilon_Pyr, delay=delay_SliExc)
    print "Pyr --> SlowInh\t\t"
    Pyr_to_Sli.append(P2S_sub)
Ext_to_Sli = IdentityConnection(PS, Sli_net, 'g_ampa',weight=J_SliExt)
print "Ext --> SlowInh\t"

print 'Connections done'

#read out time used for building
buildCPUTime = timer.elapsedTime()

# Monitors
sme = [] # Spike Monitor Exitatory
for sp in range(NP):
     print "Monitoring excitatory population %d" %sp
     sme.append(SpikeMonitor(Pyr_net[sp]))

print "Monitoring basket cell population."
smi= SpikeMonitor(Bas_net)

print "Monitoring slow inhibitory population." 
sms=SpikeMonitor(Sli_net)

popre_all = []
for sp in range(NP):
    popre_all.append(PopulationRateMonitor(Pyr_net[sp], bin=1*ms))
popri = PopulationRateMonitor(Bas_net, bin=1*ms)
poprs= PopulationRateMonitor(Sli_net, bin=1*ms)

pyr_satemon = StateMonitor(Pyr_net[0], 'vm', record=[0,1])
bas_statemon = StateMonitor(Bas_net, 'vm', record=[0,1])
sli_statemon = StateMonitor(Sli_net,'vm',record=[0,1,2,3,4,5])

#=== Run simulation ===========================================================
print "Running simulation for %g ms." % (simtime)
run(simtime*ms, report='text') 




# get the average spike rate of piramidals

pyr_rate_array=[]
for sp in range(NP):
    pyr_rate_array.append(popre_all[sp].rate)

Pyr_avg_rate = np.mean(pyr_rate_array, axis=0)

# get the average spike rate of baskets

Bas_avg_rate = popri.rate



# detect oscillations

meanEr, rEAC, maxEAC, tMaxEAC, maxEACR, tMaxEACR, fE, PxxE, avgRippleFE, ripplePE = ripple(Pyr_avg_rate, 1000)
avgGammaFE, gammaPE = gamma(fE, PxxE)

meanIr, rIAC, maxIAC, tMaxIAC, maxIACR, tMaxIACR, fI, PxxI, avgRippleFI, ripplePI = ripple(Bas_avg_rate, 1000)
avgGammaFI, gammaPI = gamma(fI, PxxI)


# saving the results into a matrix

X = [p_rate_mf,
       meanEr, maxEAC, tMaxEAC, maxEACR, tMaxEACR,
       meanIr, maxIAC, tMaxIAC, maxIACR, tMaxIACR,
       avgRippleFE, ripplePE, avgGammaFE, gammaPE,
       avgRippleFI, ripplePI, avgGammaFI, gammaPI]




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
print("Build time         : %g s" % buildCPUTime)   

# === Plots ========================================================

#membrane potentials

#figure0 = plot(pyr_statemon.times / ms, pyr_statemon[0] / mV)
#xlabel('Time (in ms)')
#ylabel('Membrane potential (in mV)')
#title('Membrane potential for pyr neuron 0')


#figure1 = plot(pyr_statemon.times / ms, pyr_statemon[1] / mV)
#xlabel('Time (in ms)')
#ylabel('Membrane potential (in mV)')
#title('Membrane potential for pyr neuron 1')


#figure2= plot(bas_statemon.times / ms, bas_statemon[0] / mV)
#xlabel('Time (in ms)')
#ylabel('Membrane potential (in mV)')
#title('Membrane potential for bas neuron 0')


#figure3 = plot(bas_statemon.times / ms, bas_statemon[1] / mV)
#xlabel('Time (in ms)')
#ylabel('Membrane potential (in mV)')
#title('Membrane potential for bas neuron 1')


# Pyr population
logic1 = isinstance(PxxE, float) # checks weather it is nan value
print logic1
logic2 = isinstance(PxxI, float)
print logic2
fig1 = plt.figure(figsize=(14, 20))

if not(logic1):
	ax = fig1.add_subplot(4, 1, 1)
	ax.plot(np.linspace(0, 1000, len(Pyr_avg_rate)), Pyr_avg_rate, 'b-')
	ax.set_title('Pyramidal population rate | Mf exitation = %.2f Hz' % p_rate_mf)
	ax.set_xlabel('Time [ms]')
	ax.set_ylabel('Pyr. rate [Hz]')

	PxxEPlot = 10 * np.log10(PxxE / max(PxxE))

	fE = np.asarray(fE)
	rippleS = np.where(145 < fE)[0][0]
	rippleE = np.where(fE < 250)[0][-1]
	gammaS = np.where(15 < fE)[0][0]
	gammaE = np.where(fE < 50)[0][-1]
	fE.tolist()

	PxxRipple = PxxE[rippleS:rippleE]
	PxxGamma = PxxE[gammaS:gammaE]

	fRipple = fE[rippleS:rippleE]
	fGamma = fE[gammaS:gammaE]

	PxxRipplePlot = 10 * np.log10(PxxRipple / max(PxxE))
	PxxGammaPlot = 10 * np.log10(PxxGamma / max(PxxE))

	ax1 = fig1.add_subplot(4, 1, 2)
	ax1.plot(fE, PxxEPlot, 'b-', marker='o', linewidth=1.5)
	ax1.plot(fRipple, PxxRipplePlot, 'r-', marker='o', linewidth=2)
	ax1.plot(fGamma, PxxGammaPlot, 'k-', marker='o', linewidth=2)
	ax1.set_title('Power Spectrum Density')
	ax1.set_xlim([0, 500])
	ax1.set_xlabel('Frequency [Hz]')
	ax1.set_ylabel('PSD [dB]')
	
	ax6 = fig1.add_subplot(4, 1, 3)
	ax6.plot(np.linspace(0, 1000, len(Bas_avg_rate)), Bas_avg_rate, 'b-')
	ax6.set_title('Bas. population rate | Mf exitation = %.2f Hz' % p_rate_mf)
	ax6.set_xlabel('Time [ms]')
	ax6.set_ylabel('Bas. rate [Hz]')

if not(logic2):
	PxxIPlot = 10 * np.log10(PxxI / max(PxxI))

	fI = np.asarray(fI)
	rippleS = np.where(145 < fI)[0][0]
	rippleE = np.where(fI < 250)[0][-1]
	gammaS = np.where(15 < fI)[0][0]
	gammaE = np.where(fI < 50)[0][-1]
	fI.tolist()

	PxxRipple = PxxI[rippleS:rippleE]
	PxxGamma = PxxI[gammaS:gammaE]

	fRipple = fI[rippleS:rippleE]
	fGamma = fI[gammaS:gammaE]

	PxxRipplePlot = 10 * np.log10(PxxRipple / max(PxxI))
	PxxGammaPlot = 10 * np.log10(PxxGamma / max(PxxI))

	ax7 = fig1.add_subplot(4, 1, 4)
	ax7.plot(fI, PxxIPlot, 'b-', marker='o', linewidth=1.5)
	ax7.plot(fRipple, PxxRipplePlot, 'r-', marker='o', linewidth=2)
	ax7.plot(fGamma, PxxGammaPlot, 'k-', marker='o', linewidth=2)
	ax7.set_title('Power Spectrum Density')
	ax7.set_xlim([0, 500])
	ax7.set_xlabel('Frequency [Hz]')
	ax7.set_ylabel('PSD [dB]')

#fig1.tight_layout()
fig1.savefig('gammaPSD_mfexitation%.2fhz_norec.png' %p_rate_mf)


fig2 = plt.figure(figsize=(10, 8))

ax = fig2.add_subplot(3,1,1)
ax.plot(np.linspace(0, simtime, len(Pyr_avg_rate)), Pyr_avg_rate, 'g-')
ax.set_title('Pyr. pop avg rate | Mf exitation = %.2f Hz' % p_rate_mf)


ax = fig2.add_subplot(3,1,2)
ax.plot(np.linspace(0, simtime, len(popri.rate)), popri.rate, 'g-')
ax.set_title('Bas. population rate')


ax = fig2.add_subplot(3,1,3)
ax.plot(np.linspace(0, simtime, len(poprs.rate)), poprs.rate, 'g-')
ax.set_title('Slow inh. population rate')
ax.set_xlabel('Time [ms]')

fig2.savefig('avg_rate%.2fmf_%.2fslowpyr_%.2fpyrslow_%ddpot.png' % (p_rate_mf,J_PyrSlowInh,J_SliExc,dpot))

fig3 = plt.figure(figsize=(10,8))
raster_plot(sme[0], spacebetweengroups=1, title='Pyr raster plot', newfigure=False)

fig4 = plt.figure(figsize=(10,8))
raster_plot(smi, spacebetweengroups=1, title='Bas. raster plot', newfigure=False)


fig3.savefig('rasterplot%dpyr%.2fMF.png' % (N_Pyr_all,p_rate_mf))
fig4.savefig('rasterplot%dbas%.2fMF.png' % (N_Pyr_all,p_rate_mf))



figure5= plot(sli_statemon.times/ms, sli_statemon[0]/ mV)
xlabel('Time [ms]')
ylabel('Membrane potential [mV]')
title('Membrane potential for sli neuron 0')
figure6= plot(sli_statemon.times/ms, sli_statemon[2]/ mV)
xlabel('Time [ms]')
ylabel('Membrane potential [mV]')
title('Membrane potential for sli neuron 2')
figure7= plot(sli_statemon.times/ms, sli_statemon[3]/ mV)
xlabel('Time [ms]')
ylabel('Membrane potential [mV]')
title('Membrane potential for sli neuron')



## raster plot and rate pyr (higher resolution)
#fig4 = plt.figure(figsize=(10, 8))

#spikes = sme.spikes
#spikingNeurons = [i[0] for i in spikes]
#spikeTimes = [i[1] for i in spikes]

#tmp = np.asarray(spikeTimes)
#ROI = np.where(tmp > 9.9)[0].tolist()

#rasterX = np.asarray(spikeTimes)[ROI] * 1000
#rasterY = np.asarray(spikingNeurons)[ROI]

#if rasterY.min()-50 > 0:
#ymin = rasterY.min()-50
#else:
#ymin = 0

#if rasterY.max()+50 < 4000:
#ymax = rasterY.max()+50
#else:
#ymax = 4000

#ax = fig4.add_subplot(2, 1, 1)
#ax.scatter(rasterX, rasterY, c='blue', marker='.', lw=0)
#ax.set_title('Raster plot (last 100 ms)')
#ax.set_xlim([9900, 10000])
#ax.set_xlabel('Time [ms]')
#ax.set_ylim([ymin, ymax])
#ax.set_ylabel('Neuron number')

#ax2 = fig4.add_subplot(2, 1, 2)
#ax2.plot(np.linspace(9900, 10000, len(popre.rate[9900:10000])), popre.rate[9900:10000], 'b-', linewidth=1.5)
#ax2.set_title('Pyr. population rate (last 100 ms)')
#ax2.set_xlabel('Time [ms]')

#fig4.tight_layout()

#figName = os.path.join(SWBasePath, 'figures', str(multiplier)+'*pyr_rate.png')
#fig4.savefig(figName)

## raster plot and rate bas (higher resolution)
#fig5 = plt.figure(figsize=(10, 8))

#spikes = smi.spikes
#spikingNeurons = [i[0] for i in spikes]
#spikeTimes = [i[1] for i in spikes]

#tmp = np.asarray(spikeTimes)
#ROI = np.where(tmp > 9.9)[0].tolist()

#rasterX = np.asarray(spikeTimes)[ROI] * 1000
#rasterY = np.asarray(spikingNeurons)[ROI]

#if rasterY.min()-50 > 0:
#ymin = rasterY.min()-50
#else:
#ymin = 0

#if rasterY.max()+50 < 1000:
#ymax = rasterY.max()+50
#else:
#ymax = 1000

#ax = fig5.add_subplot(2, 1, 1)
#ax.scatter(rasterX, rasterY, c='green', marker='.', lw=0)
#ax.set_title('Bas. raster plot (last 100 ms)')
#ax.set_xlim([9900, 10000])
#ax.set_xlabel('Time [ms]')
#ax.set_ylim([ymin, ymax])
#ax.set_ylabel('Neuron number')

#ax2 = fig5.add_subplot(2, 1, 2)
#ax2.plot(np.linspace(9900, 10000, len(popri.rate[9900:10000])), popri.rate[9900:10000], 'g-', linewidth=1.5)
#ax2.set_title('Bas. population rate (last 100 ms)')
#ax2.set_xlabel('Time [ms]')

#fig5.tight_layout()

#figName = os.path.join(SWBasePath, 'figures', str(multiplier)+'*bas_rate.png')
#fig5.savefig(figName)

plt.close('all')
