# 2022.5.20
## 仓库总体介绍
#### 前期主要研究最新NAS方向的一些工作，后期投完简历时开始研究深度学习算法的FPGA和ASIC专用芯片硬件加速实现。

# 2022.5.21
#### 重新阅读DARTS ，完成阅读报告
#### 1.DARTS其实是一个搜索加上训练的模型，即先在一个数据集上搜索收敛得到模型参数和架构参数，在重新跑一个数据集得到准确率，但是搜出来的模型为什么适用不同数据集呢
#### 2.DARTS的batch_size不能调的太大，显存会爆，也是论文说的是用cifar10训练的原因。而不是imagenet。
#### 3.DARTS是由cell堆叠起来的，每一层一个cell，但是cell（两类）是一模一样的，这样感觉没能完全自动化搜索。
#### 4.为什么搜索时采用的标准来决定的操作一定在重新训练时也是最优的。
#### 5.这个连续松弛为什么softmax后选出的一个候选操作将其变为1会大于几个候选操作分别各自乘以权？

# 2022.5.23
#### 阅读了PC-DARTS，完成报告
#### 1.减少信道数我觉得是和batchsize减少和dropout差不多的思路，所以并没有是一个大的改变，因为你整个网络喂入的数据变少了。
#### 2.edge normalization是在两两节点之间加上参数而不是在两个节点之内加上参数。这个还是蛮巧妙的。
#### 3.文章的代码量感觉会非常少。

# 2022.5.24
#### 阅读了Big NAS，完成报告
#### 1.BigNAS是直接训练的一个大的超网，里面会有几千个小模型（搜索空间的堆叠，里面其实是一个权重共享或者说是蒸馏的方式），最终参数优化到一个比较好的地方，然后遇到具体的硬件限制我再从里面具体搜索。
#### 2.这个权重共享是否就是DARTS中的连续松弛和优化？那么创新的点不就是使用策略将所有候选模块都变得比较好而且能够留到搜索阶段？

#### 阅读了AttentiveNAS
#### 1.正式整理出两阶段的概念，即先进行训练（这部分也得确定合适的架构参数，这部分在DARTS中一直说的都是搜索，所以之前一直很难理解这个概念）确定出好的网络结构，然后再进行挑选，不能说是搜索，得到自己任务下想要的结果。
#### 2.说是将搜索和训练联系到一起，其实就是有返回darts的意思，将硬件的正则加入搜索的过程之中。
#### 3.但是这样以来他的搜索阶段在哪里？这不是直接要了一个最好的模型？

# 2022.5.25
####  阅读了B-DARTS
#### 1.引入架构参数正则的数学推导还可以。
#### 2。代码量太少，还是基于最初的DARTS做的一些工作。




# 总结和展望
### 关于搜索空间，（卷积核大小，最大池化和均值池化），层数，信道数，之外是否可以更近一步？比如学习率？batch_size？然后就是人调整的都可以？
### 然后就是搜索策略
### 包括NAS和各种方向的结合，水一水论文，实在不行可以应用到各种领域比如什么生物信息学，网络分类等等
