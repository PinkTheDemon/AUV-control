# AUV-control
用于保存大规模修改前的代码版本

第一版是只有CLF和CBF的版本，还存在的问题是：
--原本不加CBF时较小的范围，加了CBF之后反而会变大？是不是CBF写反了？
--如果仅仅只对z做范围限制，那么对CLF做松弛，就可以实现。但要是再加上对θ的限制，那就会无解，不知道是什么原因。也就是说不同变量的CBF会相互影响。
--不知道MATLAB的二次规划算法内部是怎么写的，应该有什么处理，而自己写的简单版本手动求解二次规划又不知道问题出在哪里
接下来要加上RL模块，如果按照MATLAB提供的DDPG工具箱的标准模式来写，好像是要自己写环境文件，也就是要把原来的主体代码depth_control改写成RL环境，而且训练也是自动训练。但是如果不这样改写而要自己手动训练的话，暂时不清楚MATLAB怎么修改网络的权重，所以打算先改写环境吧。

20230402 优化器修改：
gurobi求的和MATLAB求的不一样，MATLAB和自己的是一样的，暂且认为gurobi不太行；而自己求解的可能是因为存在舍入误差，导致优化效果不如MATLAB求解器；最终结论还是先继续使用MATLAB自带的的二次规划求解器。

20230404 细节修改：
把变量名稍微改了下，原来叫xxdot的都改成了dxx