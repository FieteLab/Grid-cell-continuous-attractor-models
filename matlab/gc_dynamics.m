function [spikes] = gc_dynamics(filename,dt, useSpiking,simulate_non_periodic)
%---------------------------------------------------------------
% GC Dynamics 
%---------------------------------------------------------------
%
% This code is in reference to Burk Y, Fiete I (2009) 
% Accurate Path Integration in Continuous Attractor 
% Network Models of Grid Cells. 
% PLoS Computational Biology 5: e1000291
% 
% gc_dynamics initializes the network parameters and then calls
% one of two functions, gc_periodic or gc_non_periodic to run the
% simulation. If 'useSpiking' is enabled these functions will return a cell
% array containing the activity of the population.
%
%----------------------------------------------------------------
close all;

%---------------------
% Initalize Variables
%---------------------


% Timestep in ms - If you are loading your own data, you must change dt to
% your recording time step.
if dt == 0 
    dt = 0.5;   
end

%----Warning------

% The following are the parameters used in the associated paper. Altering them will
% likely lead to an unsucessful simulation.

% Number of neurons
n = 2^7; 

% Neuron time-constant (in ms)
tau = 5;

% Envelope and Weight Matrix Parameters
lambda = 13; % Equation (3)
beta = 3/lambda^2; % Equation (3)
alphabar = 1.05; % alphabar = gamma/beta from Equation (3)
abar = 1; % a should be <= alphabar^2. Equation (3)
wtphase = 2; % wtphase is 'l' from Equation (2)
alpha  = 1; % The velocity gain from Equation (4)

%---------------
% RUN SIMULATION
%---------------
if simulate_non_periodic == 0
    [spikes] = gc_periodic(filename,n,tau,dt,beta,alphabar,abar,wtphase,alpha, useSpiking);
else
    [spikes] = gc_non_periodic(filename,n,tau,dt,beta,alphabar,abar,wtphase,alpha, useSpiking);
end