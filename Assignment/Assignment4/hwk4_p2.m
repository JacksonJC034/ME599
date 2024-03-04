% system parameters
Rm = 3.5; % Ohms, Armature resistance  
Kb = 0.1; % Vs/rad, Motor back EMF constant
Kt = 0.65*Kb; % Nm/A, Motor torque constant
% Lm;  H, Armature inductance; assume 0    
mp = 0.04;% kg, Spotlight(pendulum) mass  
Lp = 0.1; %m, Spotlight(pendulum) length 
Ip = (1/2)*mp*(Lp)^2;% Moment of inertia of pendulum about center of mass
Bp = 0.000008; %Nms/rad (viscous friction coefficient)
g = 9.81; %m/s^2 acceleration due to gravity