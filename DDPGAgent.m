% DDPG Agent
clear;clc;
Tf = 20;
Ts = 0.01;

env = MyEnvClass;

obsInfo = rlNumericSpec([4 1]);
obsInfo.Name = 'eta states';
obsInfo.Description = 'w, q, z, theta';
% actInfo = rlNumericSpec([12 1]); % 2 for CLF and 3*2 for CBF
actInfo = rlNumericSpec([4 1]); % 2 for CLF and 3*2 for CBF
actInfo.Name = 'compensate term';
actInfo.Description = 'alpha(*u), beta';
% actInfo.LowerLimit = -1e2*ones(12,1);
% actInfo.UpperLimit =  1e2*ones(12,1);

mainPath = [
    featureInputLayer(4, Name='obsInLyr')
    fullyConnectedLayer(300)
    additionLayer(2, Name='add')
    reluLayer
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(1, Name='QValLyr')
    ];
actionPath = [
%     featureInputLayer(12, Name='actInLyr')
    featureInputLayer(4, Name='actInLyr')
    fullyConnectedLayer(300, Name='actOutLyr')
    ];

criticNet = layerGraph(mainPath);
criticNet = addLayers(criticNet, actionPath);
criticNet = connectLayers(criticNet,'actOutLyr','add/in2');
criticNet = dlnetwork(criticNet);
critic = rlQValueFunction(criticNet,obsInfo,actInfo,...
    ObservationInputNames="obsInLyr",ActionInputNames="actInLyr");

actorNet = [
    featureInputLayer(4)
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(400)
    reluLayer
%     fullyConnectedLayer(12)
    fullyConnectedLayer(4)
%     tanhLayer
%     scalingLayer(Scale=1e3*ones(8,1),Bias=zeros(8,1))
    ];

actorNet = dlnetwork(actorNet);
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);
criticOptions = rlOptimizerOptions( ...
    LearnRate = 1e-3, ...
    GradientThreshold = 1, ...
    L2RegularizationFactor = 1e-2);
actorOptions = rlOptimizerOptions( ...
    LearnRate = 1e-4, ...
    GradientThreshold = 1, ...
    L2RegularizationFactor = 1e-2);
% 增大正则化参数，减小buffer大小

agentOptions = rlDDPGAgentOptions( ...
    SampleTime = Ts, ...
    DiscountFactor = 0.9, ...
    MiniBatchSize = 128, ...
    TargetSmoothFactor = 1e-3, ...
    ActorOptimizerOptions = actorOptions, ...
    CriticOptimizerOptions = criticOptions, ...
    ExperienceBufferLength = 1e6);

% specify policy noise 甚至成功的那次这里都没改正，也就是噪声项都是默认的
% % OU 噪声 -- DDPG默认就是OU 噪声
agentOptions.NoiseOptions.InitialAction = 0;
agentOptions.NoiseOptions.Mean = 0;
agentOptions.NoiseOptions.Variance = 0.81;
agentOptions.NoiseOptions.VarianceDecayRate = 0.01;
agentOptions.NoiseOptions.MeanAttractionConstant = 0.5;

agent = rlDDPGAgent(actor,critic,agentOptions);

% % 加载预训练模型
% load("TD3_Model.mat");
% % 设置 Actor 网络为预训练模型
% agent.Actor = loadedActorNetwork;
% % 设置 Critic 网络为预训练模型
% agent.Critic = loadedActorNetwork;

maxepisodes = 1e4;
maxsteps = ceil(Tf/Ts);
trainingOpts = rlTrainingOptions( ...
    MaxEpisodes = maxepisodes, ...
    MaxStepsPerEpisode = maxsteps, ...
    Verbose = false, ...
    Plots = "training-progress", ...
    ScoreAveragingWindowLength = 10, ...
    StopTrainingCriteria = 'AverageSteps', ...
    StopTrainingValue = 2000);
% trainingOpts.UseParallel = true;
% trainingOpts.ParallelizationOptions.Mode = 'async';
% trainingOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;
% trainingOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';

dotraining = true;
if dotraining
    trainingStats = train(agent, env, trainingOpts);
end
generatePolicyFunction(agent);
save('DDPG_Model.mat', 'agent');