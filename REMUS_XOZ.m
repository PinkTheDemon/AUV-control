function [F1, G1, F2, G2] = REMUS_XOZ(x)
% Simulation algorithm for REMUS This code is based in:
% Prestero, T., 2001. Verification of a six-degree of freedom simulation 
% model for the REMUS autonomous underwater vehicle (Doctoral dissertation, 
% Massachusetts Institute of Technology and Woods Hole Oceanographic 
% Institution).         

% TERMS
% -------------------------------------------------------------------------
%STATE VECTOR:
%
% x = (u v w p q r xpos ypos zpos phi theta psi) ,
% Body-referenced Coordinates
% u  = Surge velocity [m/s]
% v  = Sway velocity  [m/s]
% w  = Heave velocity [m/s]
% p  = Roll rate      [rad/s]
% q  = Pitch rate     [rad/s]
% r  = Yaw rate       [rad/s]
%
% Earth-fixed coordinates
% xpos  = Position in x-direction [m]
% ypos  = Position in y-direction [m]
% zpos  = Position in z-direction [m]
% phi   = Roll angle              [rad]
% theta = Pitch angle             [rad]
% psi   = Yaw angle               [rad]
%
%INPUT VECTOR
% ui = [n delta_s delta_r]'
%  Control Fin Angles
%   n       = shaft speed            [RPM]
%   delta_s = angle of stern planes  [rad]
%   delta_r = angle of rudder planes [rad]

% Get state variables
u     = x(1); 
% v     = 0; %x(2) 
w     = x(2); 
% p     = 0; %x(4)
q     = x(3); 
% r     = 0; %x(6)
% xpos  = x(7);
% ypos  = 0; %x(8)
% zpos  = x(4);
% phi   = 0; %x(10)
theta = x(5); 
% psi   = 0; %x(12)

% 45 degree maximum rudder angle
% delta_max = 45 * pi/180;
% 
% % Check control inputs (useful later)
% if abs(delta_s) > delta_max && delta_s < delta_max
%     delta_s = sign(delta_s) * delta_max;
% end 
% 
% if abs(delta_r) > delta_max 
%     delta_r = sign(delta_r) * delta_max;
% end

% Initialize elements of coordinate system transform matrix
% -------------------------------------------------------------------------
c2 = cos(theta); 
s2 = sin(theta); 

% ----------------------------------------
% Vehicle Parameters and Coefficients
% ----------------------------------------
W = 2.99e2; % Weight (N)
B = 3.1e2; % Bouyancy (N)%% Note buoyanci incorrect simulation fail with this value

g = 9.81; % Force of gravity 
m = W/g; % Mass of vehicle

Zww   = -1.31e2;  % Cross-flow drag 
Zqq   = -6.32e-1; % Cross-flow drag
Zuw   = -2.86e1;  % Body lift force and fin lift
Zuq   = -5.22;    % Added mass cross-term and fin lift

% Center of Gravity wrt Origin at CB
xg = 0;
zg = 1.96e-2;

% Control Fin Coefficients
% Zuuds = -9.64; % Fin Lift Force

% Center of Buoyancy wrt Origin at Vehicle Nose
xb = -6.11e-1;
% zb =  0;

% Cross flow drag and added mass terms
Mww   =  3.18; % Cross-flow drag
Mqq   = -1.88e2; % Cross-flow drag
Muq   = -2; % Added mass cross term and fin lift
Muw   =  2.40e1; % Body and fin lift and munk moment
% Muuds = -6.15; % Fin lift moment

% Moments of Inertia wrt Origin at CB
Iyy  = 3.45;  

% Non-linear Moments Coefficients
Mwdot = -1.93;    % Added mass
Mqdot = -4.88;    % Added mass
Zqdot = -1.93;    % Added mass
Zwdot = -3.55e1;  % Added mass

% Set total forces from equations of motion
% x = (u v w p q r xpos ypos zpos phi theta psi)
% u = const; v, p, r, y, phi, psi = 0;
% c1, c3 = 1; s1, s3 = 0

% -------------------------------------------------------------------------
% REMUS XOZ 3DOF model in 19.Depth control, input num = 2
f1 = (W-B)*c2 + Zww*w*abs(w) + Zqq*q*abs(q)+ Zuw*u*w + (Zuq+m)*u*q + ...
     (m*zg)*q^2;
g1 = 1;
% Z = f1 + delta1;

f2 = -zg*W*s2 + xb*B*c2 + Mww*w*abs(w) + Mqq*q*abs(q) - (m*zg)*w*q + ...
     Muq*u*q + Muw*u*w;
g2 = 1;
% M = f2 + delta2;
% -------------------------------------------------------------------------

% Accelerations Matrix (Prestero Thesis page 46)
Amat = [(m - Zwdot)       -Zqdot ;
        -Mwdot      (Iyy - Mqdot)];

% Inverse Mass Matrix
Minv = inv(Amat);

F1 = Minv(1,1)*f1 + Minv(1,2)*f2;
G1 = [Minv(1,1)*g1 Minv(1,2)*g2];
F2 = Minv(2,1)*f1 + Minv(2,2)*f2;
G2 = [Minv(2,1)*g1 Minv(2,2)*g2];





