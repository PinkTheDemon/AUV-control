# AUV-control
用于保存大规模修改前的代码版本

第一版是只有CLF和CBF的版本，还存在的问题是：
--原本不加CBF时较小的范围，加了CBF之后反而会变大？是不是CBF写反了？
--如果仅仅只对z做范围限制，那么对CLF做松弛，就可以实现。但要是再加上对θ的限制，那就会无解，不知道是什么原因。也就是说不同变量的CBF会相互影响。
--不知道MATLAB的二次规划算法内部是怎么写的，应该有什么处理，而自己写的简单版本手动求解二次规划又不知道问题出在哪里
