Please cite "Y. Burak and I. R. Fiete. Accurate path integration in continuous attractor network models of grid cells. PLoS Comp. Biol. 5(2) (2009)" If this code is used. 

======================
== ASSOCIATED FILES ==
======================
GC_Dynamics2009c.zip
-- gc_dynamcs.sh
-- gc_dynamics.c
-- r.dat
-- trajectory



=================
== DESCRIPTION ==
=================

The associated code models periodic and aperiodic continuous attractor networks of Grid Cell under a variety of simulation conditions outlined below.
 
==================
== INSTALLATION ==
==================

1) Unzip GC_Dynamics2009c.zip with any archive utility (eg. http://www.7-zip.org/download.html)

*** If you just want to run the simulation, skip to that section now ***

2) To compile GC_Dynamcs.c you must have fftw3 installed. It is available at http://www.fftw.org/download.html.  There is not much support for windows so it is recommended that you use unix/osx. These instructions are for osx.

3) Once downloaded and unziped, open a terminal window and cd to the unzipped directory.

4) The code uses single precision, so when installing fftw, use the argument --enable-float. Typing the following in terminal should be sufficient for installation:

./configure --enable-float && make
make install

This installs the libs in /usr/lib/
and the include in /usr/local/include/

4) cd to the directory containing gc_dynamics.c and compile with the following command:

gcc -std=gnu99 -o gc_dynamics gc_dynamics.c -I/usr/local/lib
-I/usr/lib -lfftw3f -lm



============================
== RUNNING THE SIMULATION ==
============================

1) To run the simulation in OSX or unix, open a terminal window and cd to the directory you unzipped GC_Dynamics2009c.zip into

2) Type 'sh gc_dynamics_periodic.txt' or 'sh gc_dynamics_aperiodic.txt' to begin a simulation. Various files will be created during the simulation. Pop files are the initial population activity. sn_#_# files are single neuron recordings. The first # references when the recording took place and the second # references the neuron. Neuron locations are saved in sn_legend. For example, to view a SN recording, open matlab and load all sn_#_# files for the same neuron. Add them together and surface plot the sum: load sn_100000_001, load sn_200000_001, load sn_300000_001, surf(sn_100000_001 + sn_200000_001 + sn_300000_001)


3) If you would like to alter the parameters of the simulation, open gc_dynamics_periodic.txt in a text editor. All parameters are listed below.  The variable type (boolean) means that the -argument only needs to be present to take effect. All other require values; for example changing gc_dynamics_periodic to:

./ento5 -n 128 -randomr -spike -spikename SpikeFiles -periodic

will run a periodic simulation with 123 neurons, random initial population activity and record spiking to a file called 'SpikeFiles' the remaining values will be set to their defaults.

Changing parameters may severely degrade performance or destabilize the population activity.


Simulation Parameters:
	-dt	 	(decimal) Simulation time step in milliseconds.
	-niter	 	(integer) Number of iterations.

Network Parameters:
	-n	 	(integer) Number of neurons.
	-tau 	 	(integer) Neuron time constant.
	-clip		(decimal) Clipping in nonlinear transfer function

Spiking Parameters:
	-spike		(boolean) Use Spiking		
	-spikeout 	(boolean) Record Spike Output
	-spikename 	(string)  Name of Spike output files
	
Weight Parameters  (Equation 3):
	-wamp		(decimal) weight amplitude
	-abar		(decimal)
	-alphabar	(decimal)
	-bs		(decimal) Lambda from Equation (3)
	-vgain		(decimal) velocity gain

Boundry Parameters:
	-climit		(integer) Radius for gaussian falloff
	-slimit		(integer) Edge length for square falloff
	-scaleweights 	(boolean) Scale weights together with input scaling
	-falloff	(decimal) Gaussian Falloff factor ~exp(-falloff(*populationSize^2/distanceFromCenter^2))
	-sqfall		(boolean) Square Falloff
	-periodic	(boolean) use periodic boundaries

Initialization Parameters:
	-rfile 		(string)  Initial population activity
	-randomr 	(bool)    Randomizes Initial population activity
	-initnoise	(decimal) Order of initial noise (ex. .001)
	-nflow 		(integer) Number of iterations to run prior to simulation
	-flow_theta	(decimal) Direction of initial flow in [0,2*pi])
	
Trajectory Parameters:
	-d 		(integer) Size of enclosure
	-det 		(boolean) Deterministic circle trajectory
	-smooth 	(boolean) Use smooth turns in trajectory
	-sharp 		(boolean) use sharp turns in trajectory
	-af 		(decimal) Determines acceleration in units of v/tau
	-Nminstay 	(decimal) Minimum time to stay at one location
	-Nmaxstay 	(decimal) Maximum time to stay at one location
	-initth 	(decimal) Direction of initial trajectory in [0,2*pi])
	-initv 		(decimal) initial velocity
	-initpos	(decimal) initial position
	-trj_file 	(string)  Trajectory Filename
	-external 	(boolean) Specifies if trajectory file will be loaded 
	-vstep 		(boolean) 
	-vstepclear	(boolean) 

Velocity Gradient Parameters:
	-gradv 		(decimal,decimal) dvx, dvy

Single Neuron Recording Parameters:
	-nrec		(integer) Number of recordings
	-snradius	(integer) Allowable recording radius
	-ndumps		(integer) Number of recording saves
	-nclears	(integer) Number of clears

Population Recording Parameters:
	-ndumpp		(integer) Number of times population will be saved
	-popfile	(string) Filename for population recording

Tracking Parameters:
	-ntrack		(integer) Tracking Diameter
	-initcm		(decimal, decimal)  Initial tracking location
	-trackfile	(string) Filename for tracking recordings



