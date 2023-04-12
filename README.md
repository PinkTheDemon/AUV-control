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

把变量名稍微改了下，原来叫xxdot的都改成了dxx；另外修改了一些语句的写法，都不改变代码的实际功能.

20230406 奖励函数项：

增加了两个变量用来探究奖励函数项应该是哪两项相减。在model_error设置为0时，ddylast是和ddyreal相等的，因此应该是ddylast和ddyreal相减。

20230406 RL部分增加：

增加了RL部分的代码文件。但是还存在一些问题，目前的状态是偶尔能学习到接近标称模型的性能。另外还有一个问题，在模型存在误差的时候，那CLF和CBF不就都失效了吗？那就没有性能保证了？

20230407 留存版本：

增加了测试部分代码。目前的版本存在过多问题，需要一点点修改，但当前版本能够输出一个还算能看的结果（agentData1、2），所以进行留存。

20230410 RL修改：

把状态量State从x改正为eta。目前的问题在于：
1.为什么训练过程会出现收敛到较大的奖励值之后，过了一些episode之后又出现很小的奖励？而且一般是以突然出现一个非常之小的奖励值为标志
2.以及为什么每次训练到最后都是很快的结束？--是因为Action输出一直为-Inf，但为什么会这样？

20230411 人为误差补正：

搞清楚了误差补正项（ $\Delta_1$ 、 $\Delta_2$ ）的理论表达式。并且发现现行的奖励函数设置方法有错误，误差情况下，二次规划问题计算的解是 $\tilde{ddy}$ ，而实际应用到系统上的控制量是 $ddy$ ，二者之间差了误差项，所以在理论上二者就不可能相等，奖励函数不能设置成减小这二者的误差。

![](https://z4a.net/images/2023/04/11/07f5642bf616bae057fb57fba8764609.png)

图1：在没有添加误差补正项时，模型误差会导致系统出现稳态误差

![](https://z4a.net/images/2023/04/11/305a08fb815e7585ee4e24cc202c4e5c.png)

图2：添加人为误差补正项后，稳态误差被消除

![](https://z4a.net/images/2023/04/11/CLF.png)

图3：对CLF进行放松后，即使添加了正确的人为误差补正项，仍然会有一点稳态误差

20230412 奖励函数项：

搞清楚了奖励函数应该怎么设置，即文献中的 $||\dot{V}-\hat{\dot{V}}||^2$ 无误，也搞清楚了计算方法。 $\dot{V}$ 用真实的 $\ddot{y}$ 带进表达式来计算， $\hat{\dot{V}}$ 用二次规划问题的解 $\tilde{\ddot{y}}$ 带进表达式加上学习项来计算。
然后把相同的项略去，得到现在代码中的dV和dVhat表达式。注意，相同的系数2.\*eta.'\*Peps\*G不能约去，因为误差项中是包括了这一系数的，也就是如果用学习项来近似误差项，那么就不会具有相同的系数了。

20230412 更正奖励函数项:

更正了RL部分中的奖励函数项，但学习效果仍然不好，还是会出现输出一直为Inf的情况。

20230412 效果还行的版本留存:

总之是能得到收敛的结果，所以先留存下来，之后能不能用这个作为预训练模型接着训练？