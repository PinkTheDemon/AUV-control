classdef MyEnvClass < rl.env.MATLABEnvironment
    %MYENVCLASS: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties % 这些是在学习过程中不会改变的属性
        % Specify and initialize environment's necessary properties    
        % Sample time
        Ts = 0.01;
        Maxstepnum = 2000;

        % input-output linearization
        r = 2; % 相对度为2
        m = 2; % 输出的维数为2
        epsilon = 0.15; % 取值范围0-1，越小收敛越快
        F = cat(2, zeros(4, 2), cat(1, eye(2), zeros(2, 2))); % mr×mr
        G = cat(1, zeros(2, 2), eye(2)); % mr×m

        % CLF、CBF相关量
        Peps = zeros(4, 4);
        Gamma = 0;
        Kb = zeros(1*2);

    end
    
    properties % 这些是在每个step会变化的属性
        % Initialize system state [x,dx,theta,dtheta]'
        State   = zeros(5,1);
        dState  = zeros(5,1);
        xpos    = 0;
        ddylast = zeros(2,1);
        stepnum = 0;
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false;
    end

    %% Necessary Methods
    methods
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = MyEnvClass()
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([5 1]);
            ObservationInfo.Name = 'AUV States';
            ObservationInfo.Description = 'z, dz, theta, dtheta';
            
            % Initialize Action settings   
            ActionInfo = rlNumericSpec([4 1]);
            ActionInfo.Name = 'compensate term';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Initialize property values and pre-compute necessary values
%             updateActionInfo(this);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = []; % 不知道这玩意起什么作用
            
            % Get action 还是把getcontrol函数移到这里来，可以减少反复定义的麻烦
%             mu = getcontrol(this,Action);
            
            % Unpack state vector
            u     = this.State(1);
            w     = this.State(2);
            q     = this.State(3);
            z     = this.State(4);
            theta = this.State(5);
            
            % Cache to avoid recomputation
            s2    = sin(theta);
            c2    = cos(theta);
            dxp = w*s2 + u*c2;
            dz  = w*c2 - u*s2;
            dtheta = q;

            eta = [(z-zr(this)); theta; dz; dtheta];
            if sum(eta.'*eta) ~= 0
                % CLF
                A1 = 2.*eta.'*this.Peps*this.G;
                b1 = -eta.'*(this.F.'*this.Peps + this.Peps*this.F)*eta ...
                     - this.Gamma/this.epsilon.*eta.'*this.Peps*eta;
                % CBF
                [Bx, Bdot, BA, BB] = B(this);
                etab = [Bx; Bdot];
                % 放松CLF
                p = 1;
                A = [[A1,-1]; [BA,0]] + [Action(1); Action(2)]; % action为RL项
                b = [b1; this.Kb*etab+BB]  + [Action(3); Action(4)];
                IsDone = any(any(isinf(A)+isinf(b)));
                IsDone = IsDone || any(isinf(this.State));
                IsDone = IsDone || ~isreal(Bx);
                if ~IsDone
                    result = quadprog(blkdiag(eye(this.m),p), zeros(this.m+1,1), A, b);
                    ddy = result(1:2);
                else 
                    ddy = [0;0];
                end
            else 
                ddy = [0;0];
            end
            [F1, G1, F2, G2] = REMUS_XOZ(this.State); % 这是标称模型，与实际模型有误差
            mu = [G1*c2; G2]\(ddy - [F1*c2-w*q*s2-u*q*c2; F2]);

            % Apply motion equations
            % generate model error
            [e1, e2] = GenModelerror(this);

            % dynamics (real dynamics, different from nominal model)
            dx = [      0      ;
                  F1+e1 + G1*mu;
                  F2+e2 + G2*mu;
                 -s2*u  + c2* w;
                              q];
            ddyreal = (dx(4:5) - this.dState(4:5))/this.Ts;
            
            % Euler integration
            Observation = this.State + this.Ts.*dx;

            % Update system states
            this.State  = Observation;
            this.dState = dx;
            this.xpos = this.xpos + this.Ts*dxp;
            this.ddylast = ddy;
            this.stepnum = this.stepnum + 1;
            
            % Check terminal condition，提前结束条件是跳出安全范围
            IsDone = IsDone || this.stepnum>=this.Maxstepnum;
            this.IsDone = IsDone;
            
            % Get reward
            Reward = getReward(this ,ddyreal);
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            % u
            u0 = 5;
            % w
            w0 = 0;
            % q 
            q0 = 0; 
            % z
            z0 = -5;
            % theta
            theta0 = 0.15;
            InitialObservation = [u0;w0;q0;z0;theta0];

            % CLF系数矩阵Peps计算
            Q = 1*eye(this.m*this.r); % 任意正定矩阵,mr×mr
            R = 1*eye(2);
            Meps = kron([1/this.epsilon 0;0 1], eye(this.m));
            P = are(this.F, this.G/R*this.G.', Q); % Riccati方程的解
            peps = Meps * P * Meps;
            lamdaQ = min(eig(Q)); % Q的最小特征值
            lamdaP = max(eig(P)); % P的最大特征值
            gamma = lamdaQ/lamdaP;

            % CBF极点配置计算Kb
            Fb = [0 1;0 0];
            Gb = [0;1];
            poles = [-1+2i, -1-2i];
            kb = place(Fb, Gb, poles); % 极点配置

            this.State = InitialObservation;
            this.xpos = 0;
            this.dState = zeros(5,1);
            this.Peps = peps;
            this.Gamma = gamma;
            this.Kb = kb;
            this.stepnum = 0;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
        % Helper methods to create the environment
        % 根据动作（模型误差补偿项），求出控制量
%         function mu = getcontrol(this,action) % 这个在我们的问题中是求解二次规划计算控制量
% 
%             u     = this.State(1);
%             w     = this.State(2);
%             q     = this.State(3);
%             z     = this.State(4);
%             theta = this.State(5);
%             s2    = sin(theta);
%             c2    = cos(theta);
%             dz  = w*c2 - u*s2;
%             dtheta = q;
%             
%             eta = [(z-zr(this)); theta; dz; dtheta];
%             if sum(eta.'*eta) ~= 0
%                 % CLF
%                 A1 = 2.*eta.'*this.Peps*this.G;
%                 b1 = -eta.'*(this.F.'*this.Peps + this.Peps*this.F)*eta ...
%                      - this.Gamma/this.epsilon.*eta.'*this.Peps*eta;
%                 % CBF
%                 [Bx, Bdot, BA, BB] = B(this);
%                 etab = [Bx; Bdot];
%                 % 放松CLF
%                 p = 1;
%                 A = [[A1,-1]; [BA,0]] + [action(1); action(2)]; % action为RL项
%                 b = [b1; this.Kb*etab+BB]  + [action(3); action(4)];
%                 result = quadprog(blkdiag(eye(this.m),p), zeros(this.m+1,1), A, b);
%                 ddy = result(1:2);
%             else 
%                 ddy = [0;0];
%             end
% 
%             [F1, G1, F2, G2] = REMUS_XOZ(this.State); % 这是标称模型，与实际模型有误差
%             mu = [G1*c2; G2]\(ddy - [F1*c2-w*q*s2-u*q*c2; F2]);
%         end

        % define refference trajectory
        function ref_traj = zr(this)

            choise = 1;
            if choise == 1
                ref_traj = -10 ; % 恒定深度控制
            elseif choise == 2
                ref_traj = -10 + 0.1*sin(0.1*this.xpos); % 已知轨迹跟踪
            else
                ref_traj = 1.5 + (-10 + 0.1*sin(0.1*this.xpos)); % 海底地形跟踪
            end

        end

        % define barrier function
        function [Bx, dB, BA, BB] = B(this)
        % pi/2 - theta > 0; pi/2 + theta > 0; 10.5 + z > 0
        % B(x) += -log(a) +log(a+1)
            u = this.State(1);
            w = this.State(2);
            q = this.State(3);
            z = this.State(4);
            theta = this.State(5);
            c2 = cos(theta);
            s2 = sin(theta);
%             a = 0.4 - theta;
%             b = 0.4 + theta;
            c = 10.4 + z;

            Bx = 0;%-log(a) +log(a+1);
%             Bx = Bx -log(b) +log(b+1);
            Bx = Bx -log(c) +log(c+1);

        % dB(x) += -(1/a -1/(a+1))*adot;
            dB = 0;%-(1/a -1/(a+1))*(-q);
%             dB = dB - (1/b -1/(b+1))*q;
            dB = dB - (1/c -1/(c+1))*(w*c2-u*s2);

        % ddB(x) = BA*ddy + BB
        % BA += -(1/a -1/(a+1))*da^2
        % BB += (1/a^2 -1/(a+1)^2)*da^2
            BA(1) = -(1/c -1/(c+1));
            BA(2) = 0;%1/a -1/(a+1);
%             BA(2) = BA(1) -(1/b -1/(b+1));
            BB = 0;%(1/a^2 -1/(a+1)^2)*q^2;
%             BB = BB + (1/b^2 -1/(b+1)^2)*q^2;
            BB = BB + (1/c^2 -1/(c+1)^2)*(w*c2-u*s2)^2;
        end

        % update the action info based on max force
%         function updateActionInfo(this) % 这个函数也不知道有什么用
%             this.ActionInfo.Elements = zeros(4,1);
%         end
        
        % Reward function 这样设置奖励函数的话，反正都走不到2000步，
        % 那提前结束奖励就会大了。应该根据距离2k步的剩余步数设置提前终止的惩罚
        % 奖励设置错误
        function Reward = getReward(this, ddyreal)
            if this.stepnum == 1
                Reward = 0;
            elseif ~this.IsDone
                Reward = -norm(this.ddylast -ddyreal)^2;
            elseif this.IsDone
                Reward = -1e3*(this.Maxstepnum - this.stepnum);
            end
        end

        % generate model error
        function [mderr1, mderr2] = GenModelerror(this)
            mderr1 = 1;
            mderr2 = 2;
        end

        % (optional) Visualization method
        function plot(this)
            % Initiate the visualization
            
            % Update the visualization
            envUpdatedCallback(this)
        end
        
        % (optional) Properties validation through set methods
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',5},'','State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end
        function set.Ts(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Ts');
            this.Ts = val;
        end
    end


    methods (Static)
        
    end

    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
        end
    end
end