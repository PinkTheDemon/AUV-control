% DDPG Agent
clear;clc;
Tf = 20;
Ts = 0.01;

env = MyEnvClass;

obsInfo = rlNumericSpec([5 1]);
obsInfo.Name = 'eta states';
obsInfo.Description = 'w, q, z, theta';
actInfo = rlNumericSpec([4 1]); % 2 for CLF and 2 for CBF
actInfo.Name = 'compensate term';
actInfo.Description = 'alpha(*u), beta';

mainPath = [
    featureInputLayer(5, Name='obsInLyr')
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
    featureInputLayer(5)
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(4)
    ];

actorNet = dlnetwork(actorNet);
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);

criticOptions = rlOptimizerOptions( ...
    LearnRate = 1e-3, ...
    GradientThreshold = 1, ...
    L2RegularizationFactor = 1e-4);
actorOptions = rlOptimizerOptions( ...
    LearnRate = 1e-4, ...
    GradientThreshold = 1, ...
    L2RegularizationFactor = 1e-4);

agentOptions = rlDDPGAgentOptions( ...
    SampleTime = Ts, ...
    ActorOptimizerOptions = actorOptions, ...
    CriticOptimizerOptions = criticOptions, ...
    ExperienceBufferLength = 1e2);
agentOptions.NoiseOptions.Variance = [0.1;0.1;0.1;0.1];
agentOptions.NoiseOptions.VarianceDecayRate = 1e-2;

agent = rlDDPGAgent(actor,critic,agentOptions);

maxepisodes = 5e2;
maxsteps = ceil(Tf/Ts);
trainingOpts = rlTrainingOptions( ...
    MaxEpisodes = maxepisodes, ...
    MaxStepsPerEpisode = maxsteps, ...
    Verbose = false, ...
    Plots = "training-progress", ...
    StopTrainingCriteria = 'EpisodeCount', ...
    StopTrainingValue = 450);

dotraining = true;
if dotraining
    trainingStats = train(agent, env, trainingOpts);
end