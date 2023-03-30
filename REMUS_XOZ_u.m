function [F1, G1, F2, G2, F3, G3] = REMUS_XOZ_u(x)
% Simulation algorithm for REMUS This code is based in:
% Prestero, T., 2001. Verification of a six-degree of freedom simulation 
% model for the REMUS autonomous underwater vehicle (Doctoral dissertation, 
% Massachusetts Institute of Technology and Woods Hole Oceanographic 
% Institution).         
            
%              _____F1__________________________________
%          |  /           ||                       ||   \
%          |==      F3    ||       Remus           ||    | 
%          |  \___________||_______________________||___/
%                   F2
% V=[u,v,w,p,q,r,x,y,z,phi,psi,theta];


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
%   n= shaft speed RPM
%   delta_s = angle of stern planes  [rad]
%   delta_r = angle of rudder planes [rad]

% Get state variables
u     = x(1); 
% v     = x(2); % 0
w     = x(2); 
% p     = x(4); % 0
q     = x(3); 
% r     = x(6); % 0
% xpos  = x(7); % 不在此处考虑
% ypos  = x(8); % 0
% zpos  = x(4); % 不在此处考虑
% phi   = x(10); % 0
theta = x(5); 
% psi   = x(12); % 0
% if mu(1) < 0
%     mu(1) = 0;
% end

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
% ---------------------------------------- --------------------------------
c2 = cos(theta); 
s2 = sin(theta); 

% ----------------------------------------
% Vehicle Parameters and Coefficients
% ----------------------------------------
W = 2.99e2; % Weight (N)
B = 3.1e2; % Bouyancy (N)%% Note buoyanci incorrect simulation fail with this value

g = 9.81; % Force of gravity 
m = W/g; % Mass of vehicle

Xuu   = -1.62;    % Axial Drag
Xwq   = -3.55e1;  % Added mass cross-term
Xqq   = -1.93;    % Added mass cross-term
% Xvr   =  3.55e1;  % Added mass cross-term
% Xrr   = -1.93;    % Added mass cross-term
% Yvv   = -1.31e3;  % Cross-flow drag
% Yrr   =  6.32e-1; % Cross-flow drag
% Yuv   = -2.86e1;  % Body lift force and fin lift
% Ywp   =  3.55e1;  % Added mass cross-term
% Yur   =  5.22;    % Added mass cross-term and fin lift
% Ypq   =  1.93;    % Added mass cross-term
Zww   = -1.31e2;  % Cross-flow drag 
Zqq   = -6.32e-1; % Cross-flow drag
Zuw   = -2.86e1;  % Body lift force and fin lift
Zuq   = -5.22;    % Added mass cross-term and fin lift
% Zvp   = -3.55e1;  % Added mass cross-term
% Zrp   =  1.93;    % Added mass cross-term

% Center of Gravity wrt Origin at CB
% xg = 0;
% yg = 0;
zg = 1.96e-2;

% Control Fin Coefficients
% Yuudr =  9.64;
% Nuudr = -6.15;
Zuuds = -9.64; % Fin Lift Force

% Center of Buoyancy wrt Origin at Vehicle Nose
xb = -6.11e-1;
% yb =  0;
% zb =  0;
% Propeller Terms
% Xprop =  1.569759e-4*n*abs(n);
% Kpp   = -1.3e-1;  % Rolling resistance
% Kprop = -2.242e-05*n*abs(n);%-5.43e-1; % Propeller Torque
% Kpdot = -7.04e-2; % Added mass

% Cross flow drag and added mass terms
Mww   =  3.18; % Cross-flow drag
Mqq   = -1.88e2; % Cross-flow drag
% Mrp   =  4.86; % Added mass cross-term
Muq   = -2; % Added mass cross term and fin lift
Muw   =  2.40e1; % Body and fin lift and munk moment
% Mwdot = -1.93; % Added mass
% Mvp   = -1.93; % Added mass cross term
Muuds = -6.15; % Fin lift moment
% Nvv   = -3.18; % Cross-flow drag
% Nrr   = -9.40e1; % Cross-flow drag
% Nuv   = -2.40e1; % Body and fin lift and munk moment
% Npq   = -4.86; % Added mass cross-term

% Moments of Inertia wrt Origin at CB
% Ixx  = 1.77e-1;   
Iyy  = 3.45;  
% Izz  = 3.45;

% Nwp = -1.93; % Added mass cross-term
% Nur = -2.00; % Added mass cross term and fin lift

% Non-linear Moments Coefficients
Xudot = -9.30e-1; % Added mass
% Yvdot = -3.55e1;  % Added mass
% Nvdot =  1.93;    % Added mass
Mwdot = -1.93;    % Added mass
Mqdot = -4.88;    % Added mass
Zqdot = -1.93;    % Added mass
Zwdot = -3.55e1;  % Added mass
% Yrdot =  1.93;    % Added mass
% Nrdot = -4.88;    % Added mass

% Set total forces from equations of motion
% -------------------------------------------------------------------------
f1 = -(W-B)*s2 + Xuu*u*abs(u) + (Xwq-m)*w*q + Xqq*q^2;
g1 = 1;
% h1 = 0;

f2 = (W-B)*c2 + Zww*w*abs(w) + Zqq*q*abs(q)+ Zuw*u*w + (Zuq+m)*u*q + ...
     (m*zg)*q^2;
% g2 = 0;
h2 = Zuuds*u^2;

f3 = -zg*W*s2 + xb*B*c2 + Mww*w*abs(w) + Mqq*q*abs(q) - (m*zg)*w*q + ... 
      Muq*u*q + Muw*u*w;
% g3 = 0;
h3 = Muuds*u^2;
% -------------------------------------------------------------------------


% Accelerations Matrix (Prestero Thesis page 46)
Amat = [(m - Xudot) 0            m*zg        ;
        0           (m - Zwdot) -Zqdot       ;
        m*zg        -Mwdot      (Iyy - Mqdot)];

% Inverse Mass Matrix
Minv = inv(Amat);

F1 = Minv(1,1)*f1 + Minv(1,2)*f2 + Minv(1,3)*f3;
F2 = Minv(2,1)*f1 + Minv(2,2)*f2 + Minv(2,3)*f3;
F3 = Minv(3,1)*f1 + Minv(3,2)*f2 + Minv(3,3)*f3;
G1 = [Minv(1,1)*g1  Minv(1,2)*h2 + Minv(1,3)*h3];
G2 = [Minv(2,1)*g1  Minv(2,2)*h2 + Minv(2,3)*h3];
G3 = [Minv(3,1)*g1  Minv(3,2)*h2 + Minv(3,3)*h3];

