% 1、--分析内动态
% %  --两delta输入的情况没有内动态？可能是有6个内动态，但是大部分是比较重复的
% 2、-按照真实模型，不考虑u恒定，这样会多出一个输入量n，是否能够得到更好的控制？
% 3、--探究Q与控制效果的关系，是否可以通过内动态的相关设计来找到P？（仔细读14.RES-CLF）
% 4、epsilon虽然起到调节收敛速度的作用，但对于性能的影响在文献中没有讨论，还需要进一步分析或测试
% 5、--epsilon的引入还存在问题
% 6、q为什么会高频变化？

%% main program
clear ; clc ; close ;

% system constant value
Tf = 20;   % 仿真总时间
Ts = 0.01; % 单步时间(s)
N  = Tf/Ts;

%% controller
% u = const; v, p, r, y, phi, psi = 0;
% given current state x = [w, q, z, theta]
% generate control law delta_s

% state vector init
x     = zeros(5,1);
x(1)  = 5;
x(4)  = -5;
% x(5)  = 0.15;
x_ses = zeros(5,N);
x_pos = zeros(1,N);
xpos  = 0;
V_ses = zeros(1,N);
Y_ses = zeros(4,N);

% control vector init
mu = [0;0];

% input-output linearization
r = 2; % 相对度为2
m = 2; % 输出的维数为2
F = cat(2, zeros(m*r, m), cat(1, eye(m*r-m), zeros(m, m*r-m))); % mr×mr
G = cat(1, zeros(m*r-m, m), eye(m)); % mr×m

% CLF系数矩阵Peps计算
epsilon = 0.15; % 取值范围0-1，越小收敛越快
Q = 1*eye(m*r); % 任意正定矩阵,mr×mr
R = 1*eye(2);
Meps = kron([1/epsilon 0;0 1], eye(m));
P = are(F, G/R*G.', Q); % Riccati方程的解
Peps = Meps * P * Meps;
% Peps = P;

% CBF极点配置计算Kb
Fb = [0 1;0 0];
Gb = [0;1];
poles = [-1+5i, -1-5i];
Kb = place(Fb, Gb, poles); % 极点配置

% -------------------------------------------------------------------------
% % try yalmip
% Y = sdpvar(m*r,m*r);
% L = sdpvar(m,m*r,'full');
% F = [Y >= 0];
% F = [F, [(-F*Y-G*L+(-F*Y-G*L).') Y L.';Y inv(Q) zeros(m*r,m);L zeros(m,m*r) inv(R)] >= 0];
% optimize(F,-trace(Y));
% K = value(L)/(value(Y));
% -------------------------------------------------------------------------

for i = 1:N
    u     = x(1);
    w     = x(2);
    q     = x(3);
    z     = x(4);
    theta = x(5);
    s2    = sin(theta);
    c2    = cos(theta);
    zdot  = w*c2 - u*s2;
    xpdot = w*s2 + u*c2;
    thetadot = q;

    % CLF
    eta = [(z-zr(xpos)); theta; zdot; thetadot];
    lamdaQ = min(eig(Q)); % Q的最小特征值
    lamdaP = max(eig(P)); % P的最大特征值
    gamma = lamdaQ/lamdaP;
    if sum(eta.'*eta) ~= 0
        % CLF
        A1 = 2.*eta.'*Peps*G;
        b1 = -eta.'*(F.'*Peps + Peps*F)*eta - gamma/epsilon.*eta.'*Peps*eta;
        % CBF
        [Bx, Bdot, BA, BB] = B(x);
        etab = [Bx; Bdot];
        p = 1; % 放松CLF
        A = [[A1,-1]; [BA,0]];% + alpha(eta); % alpha为RL项
        b = [b1; Kb*etab+BB];% + beta(eta); % beta为RL项
        result = quadprog(blkdiag(eye(m),p), zeros(m+1,1), A, b);
        ydot2 = result(1:2);
%         A = [A1; -BA];
%         b = [b1; Kb*etab+BB];
%         [ydot2,favl,exitflag] = quadprog(eye(m), zeros(m,1), A, b);

        % 手动求解二次规划 -------------------------------------------------
%         ydot2 = [0;0];
%         if max(A*ydot2 - b) > 0
%             ydot2 = -A1.'*abs(b1)/norm(A1)/norm(A1);
%         end
%         if max(A*ydot2 - b) > 0
%             ydot2 = BA.'*abs(Kb*etab+BB)/norm(BA)/norm(BA);
%         end
%         if max(A*ydot2 - b) > 0
%             ydot2 = A\b; % A不满秩的情况，前面两种一定有解
%         end
        % -----------------------------------------------------------------
    else
        ydot2 = [0;0];
    end
    % u恒定 ---------------------------------------------------------------
    % 计算当前状态下的模型参数
    [F1, G1, F2, G2] = REMUS_XOZ(x); % 这是标称模型，与实际模型有误差
    mu = [G1*c2; G2]\(ydot2 - [F1*c2-w*q*s2-u*q*c2; F2]);
    % dynamics (real dynamics, different from nominal model)
    xdot = [        0      ;
            F1+w1( )+ G1*mu;
            F2+w2( )+ G2*mu;
           -s2*u    + c2* w;
                          q];
    Bdot2 = BA*ydot2+BB;
    % 指标计算
    Y_ses(:,i) = [ydot2; mu];
    V = eta.'*Peps*eta;
    V_ses(i) = V; % 为什么V会在某一段有抖动上升？因为仿真步长太大

%     % 简单情况，比CLF效果反而要好，说明CLF中b项过于保守
%     % 在LQR中，李函数的系数矩阵是什么呢？[Q + P*G/R*G.'*P]
%     K = -R \ G.' * P;
%     ydot2 = K*eta;
%     [F1, G1, F2, G2] = REMUS_XOZ(x);
%     mu = [G1*c2; G2]\(ydot2 - [F1*c2-w*q*s2-u*q*c2; F2]);
%     Y_ses(:,i) = [ydot2; mu];
%     V = eta.'*Peps*eta;
%     V_ses(i) = V;
%     xdot = [      0      ;
%             F1    + G1*mu;
%             F2    + G2*mu;
%            -s2*u  + c2*w ;
%                        q ];
    % ---------------------------------------------------------------------

    % u变化 ---------------------------------------------------------------
%     % 结果异常大概是因为内动态u不稳定
%     mu = [c2.*G2-s2.*G1; G3]\(ydot2-[F2*c2-F1*s2-w*q*s2-u*q*c2; F3]);
%     Y_ses(:,i) = [ydot2; mu];
%     V = eta.'*Peps*eta;
%     V_ses(i) = V; % V是纯上升的...
%     % dynamics
%     [xdot, F1, G1, F2, G2, F3, G3] = REMUS_XOZ_u(x, mu);

%     % 简单情况尝试
%     [F1, G1, F2, G2, F3, G3] = REMUS_XOZ_u(x);
%     K = -R \ G.' * P;
%     ydot2 = [0;0];%K*eta;
%     mu = [c2.*G2-s2.*G1; G3]\(ydot2-[F2*c2-F1*s2-w*q*s2-u*q*c2; F3]);
%     Y_ses(:,i) = [ydot2; mu];
%     V = eta.'*Peps*eta;
%     V_ses(i) = V;
%     
%     xdot = [F1   + G1*mu;
%             F2   + G2*mu;
%             F3   + G3*mu;
%            -s2*u + c2*w ;
%                       q ];
    % ---------------------------------------------------------------------

    x = x + xdot*Ts;
    x_ses(:,i) = x;
    xpos = xpos + xpdot*Ts;
    x_pos(:,i) = xpos;
end

%% plot
% % u-t plot
% figure(1)
% plot(1:N, x_ses(1,:));
% title('Surge velocity'); xlabel('step（'); ylabel('u(m/s)'); grid;

% w-t plot
figure(2)
plot(1:N, x_ses(2,:));
title('Heave velocity'); xlabel('step('); ylabel('w(m/s)'); grid;

% q-t plot
figure(3)
plot(1:N, x_ses(3,:));
title('Pitch rate'); xlabel('step('); ylabel('q(rad/s)'); grid;

% z-t plot
figure(4)
plot(1:N, x_ses(4,:));
title('Depth'); xlabel('step('); ylabel('z(m)'); grid;
% ylim([-5 5]);

% theta-t plot
figure(5)
plot(1:N, x_ses(5,:));
title('Pitch angle'); xlabel('step('); ylabel('theta(rad)'); grid;

% z-x plot
figure(6)
plot(x_pos, x_ses(4,:), '-'); hold on;
plot(x_pos, zr(x_pos).*ones(size(x_pos)), 'r--');
title('z-x plot'); xlabel('x(m)'); ylabel('z(m)'); legend('ctr', 'ref'); grid;

% Lyapunov V-t plot
figure(7)
plot(1:N, V_ses(1,:));
title('Lyapunov func value'); xlabel('step'); ylabel('V'); grid;

% ydot-t plot
figure(8)
plot(1:N, Y_ses(1,:)); hold on;
plot(1:N, Y_ses(2,:));
title('ydot-t plot'); xlabel('t'); ylabel('ydot'); legend('zdot', 'thetadot'); grid;

% mu-t plot
figure(9)
plot(1:N, Y_ses(3,:)); hold on;
plot(1:N, Y_ses(4,:));
title('mu-t plot'); xlabel('t'); ylabel('mu'); legend('mu1', 'mu2'); grid;

%% define refference trajectory
function ref_traj = zr(x)

    choise = 1;
    if choise == 1
        ref_traj = -10 ; % 恒定深度控制
    elseif choise == 2
        ref_traj = -10 + 0.1*sin(0.1*x); % 已知轨迹跟踪
    else
        ref_traj = 1.5 + (-10 + 0.1*sin(0.1*x)); % 海底地形跟踪
    end

end

%% define barrier function
function [Bx, Bdot, BA, BB] = B(x)
% pi/2 - theta > 0; pi/2 + theta > 0; 10.5 + z > 0
% B(x) += -log(a) +log(a+1)
    u = x(1);
    w = x(2);
    q = x(3);
    z = x(4);
    theta = x(5);
    c2 = cos(theta);
    s2 = sin(theta);
%     a = 0.4 - theta;
%     b = 0.4 + theta;
    c = 10.4 + z;

    Bx = 0;%-log(a) +log(a+1);
%     Bx = Bx -log(b) +log(b+1);
    Bx = Bx -log(c) +log(c+1);

% Bdot(x) += -(1/a -1/(a+1))*adot;
    Bdot = 0;%-(1/a -1/(a+1))*(-q);
%     Bdot = Bdot - (1/b -1/(b+1))*q;
    Bdot = Bdot - (1/c -1/(c+1))*(w*c2-u*s2);

% Bdot2(x) = BA*ydot2 + BB
% BA += -(1/a -1/(a+1))*adot2
% BB += (1/a^2 -1/(a+1)^2)*adot^2
    BA(1) = -(1/c -1/(c+1));
    BA(2) = 0;%1/a -1/(a+1);
%     BA(2) = BA(1) -(1/b -1/(b+1));
    BB = 0;%(1/a^2 -1/(a+1)^2)*q^2;
%     BB = BB + (1/b^2 -1/(b+1)^2)*q^2;
    BB = BB + (1/c^2 -1/(c+1)^2)*(w*c2-u*s2)^2;

end

%% set model error
function mderr1 = w1()
    mderr1 = 1; %0.01*randn();
end

function mderr2 = w2()
    mderr2 = 2; %1*randn();
end