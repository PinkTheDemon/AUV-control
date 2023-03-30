function [M,C,D] = remus100_3DOF(u,v,r)
% [M,C,D] = remus100_3DOF(U,L,B,T,Cb,R66,xg,T_surge) computes the system matrices 
% Outputs: 3x3 model matrices M and N in surge, sway and yaw
%    M nu + N(U) nu = tau,     where N(U) = C(U) + D
% 
% corresponding to the linear maneuvering model
% 
%  (m - Xudot) udot - Xu u                            = (1-t) T
%  (m - Yvdot) vdot + (m - Yrdot)  rdot - Yv v - Yr r = Yd delta
%  (m - Yvdot) vdot + (Iz - Nrdot) rdot - Nv v - Nr r = Nd delta
%
% Note that the coefficients Yv, Yr, Nv and Nr in the N(U) matrix includes 
% linear damping D and the linearized Coriolis and centripetal matrix C(U).
%
% Inputs:
%
% Rigid body parameters
rho = 1025;                 % density of water
m = 185;                    % mass=185kg
Iz = 50;      % moment of inerta about the CO
% Nondimenisonal hydrodynamic derivatives in surge
Xu = 70;  
Xudot = -30;
Yv=100;
Yvdot=-80;
Nr=50;
Nrdot=-30;
m11=m-Xudot;
m22=m-Yvdot;
m33=Iz-Nrdot;
Xuu=100;
Yvv=200;
Nrr=100;

MRB = [ m   0    0           % rigid-body inertia matrix
        0   m    0
        0   0    Iz ];

MA = [ -Xudot   0       0
      0        -Yvdot   0
      0         0       -Nrdot ];


C=[0          0            -m22*v
   0          0             m11*u
   m22*v      -m11*u         0      ];



D=[Xu+Xuu*abs(u)     0                  0
    0                Yv+Yvv*abs(v)      0
    0                0                  Nr+Nrr*abs(r)];
 
M = MRB + MA;       % system inertia matrix

 
 
