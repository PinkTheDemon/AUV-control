% DDPG Agent
clear;clc;
Tf = 20;
Ts = 0.01;

env = MyEnvClass;

obsInfo = rlNumericSpec([4 1]);
obsInfo.Name = 'eta states';
obsInfo.Description = 'w, q, z, theta';
actInfo = rlNumericSpec([4 1]); % 2 for CLF and 2 for CBF
actInfo.Name = 'compensate term';
actInfo.Description = 'alpha(*u), beta';
actInfo.LowerLimit = -1e3*ones(4,1);
actInfo.UpperLimit =  1e3*ones(4,1);

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
    fullyConnectedLayer(4)
%     tanhLayer
%     scalingLayer(Scale=1e3*ones(4,1),Bias=zeros(4,1))
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

agentOptions = rlTD3AgentOptions( ...
    SampleTime = Ts, ...
    ActorOptimizerOptions = actorOptions, ...
    CriticOptimizerOptions = criticOptions, ...
    ExperienceBufferLength = 1e4);
% agentOptions.ExplorationModel.StandardDeviation = [0.1; 0.1; 0.2; 0.2];
% agentOptions.ExplorationModel.StandardDeviationDecayRate = 1e-3;
agent = rlTD3Agent(actor,critic,agentOptions);

% % 加载预训练模型
% load("TD3_Model.mat");
% % % 设置 Actor 网络为预训练模型
% % agent.Actor = loadedActorNetwork;
% % % 设置 Critic 网络为预训练模型
% % agent.Critic = loadedActorNetwork;

maxepisodes = 5e2;
maxsteps = ceil(Tf/Ts);
trainingOpts = rlTrainingOptions( ...
    MaxEpisodes = maxepisodes, ...
    MaxStepsPerEpisode = maxsteps, ...
    Verbose = false, ...
    Plots = "training-progress", ...
    StopTrainingCriteria = 'EpisodeCount', ...
    StopTrainingValue = 500);
% trainingOpts.UseParallel = true;
% trainingOpts.ParallelizationOptions.Mode = 'async';
% trainingOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;
% trainingOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';

dotraining = true;
if dotraining
    trainingStats = train(agent, env, trainingOpts);
end
generatePolicyFunction(agent);
save('TD3_Model1.mat', 'agent');