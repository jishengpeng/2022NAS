import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *   #引用了operations中一些函数，例如在每个cell的与处理中，其实还是引用了一些层
from torch.autograd import Variable
from genotypes import PRIMITIVES   #宏观上定义操作，八个
from genotypes import Genotype  #引入一些东西，来保存一个cell里面选出来的每个node（每个node在比他小的node中选两个作为输入）


class MixedOp(nn.Module):  #把每两个节点的之间的候选操作都遍历过去，返回sum（每一个候选操作*架构系数）
  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)  #传回genotypes中的开头，其实就是遍历所有的候选操作
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))  #有池化加一个标准化。
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))  #返回架构参数乘以经过操作的结果，返回的是一个矩阵，即返回每个node的结果，八条边的和


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction     #记录当前节点类型

    if reduction_prev:    #记录之前的节点类型
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)   #operation中定义，实际是一些激活和卷积操作
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)  #operation中定义，实际是一些激活和卷积操作
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)   #也是一些层
    self._steps = steps
    self._multiplier = multiplier   #乘的系数

    self._ops = nn.ModuleList()   #集合，放mixedop的结果，可以看到所有step的所有step都放进去了
    self._bns = nn.ModuleList()   #集合，这个有啥用？
    for i in range(self._steps):  #每一步有i+2的操作
      for j in range(2+i):  #到当前节点他需要与之前多少节点进行操作
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1) #两个输入分别进行预处理

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))  
      #这个地方的self._ops[offset+j]其实是遍历找出每一个op，这个op其实一个节点经过所有候选操作之后的特征图，这个地方为啥后面还要跟一个
      #(h, weights[offset+j])其实没太理解。因为h也应该是一个特征图，但是最后就是返回经过这个cell的结果
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)  #把一个cell中所有node的输出结果都cat在一起，即0，1，2，3每个特征图加起来形成最终cell的输出


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3): #后面三个参数，最后两个就是乘数
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C    #主函数传进来的信道数再次乘以了一个倍数
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),   #输入信道，输出信道，卷积核大小
      nn.BatchNorm2d(C_curr)  #标准化
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C   #重新进行了一个卷积信道的赋值
    self.cells = nn.ModuleList()       #这边也是将不同的层整体包装一下
    reduction_prev = False      #啥用的
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True                #这部分说明是整个网络结构的1/3或者2/3层部分试reduction类型的cell，这部分的区别就是信道多了2倍
      else:
        reduction = False
      #这边还是比较关键的，看的出来初始化model的时候，每一层的cell都已经初始产生了
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)  #前两个参数传一个乘的系数，中间三个是信道，每一个cell的输入是前两个cell的输出信道，最后两个看他是节点类型，
      reduction_prev = reduction
      self.cells += [cell]    #把cell堆叠在一起
      C_prev_prev, C_prev = C_prev, multiplier*C_curr  #更新一些信道参数，但是这些不是动态变化的，每次都是固定变换的

    self.global_pooling = nn.AdaptiveAvgPool2d(1)    #平均池化，相比 nn.AvgPool2d() 多了个自适应，即维度约束力很弱
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()   #初始化架构参数a，调用下面的一个函数，每个option上面都有一个参数实际上是一个几乘几的列表

  def new(self):   #返回一个模型，但是实际上跟之前是一摸一样的
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):  #序列解包同时遍历
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):  
    s0 = s1 = self.stem(input)  #输入先经过卷积和池化，注意信道有个倍数的乘
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)#传入前两个cell的输出和权重，这里经过的是cell的foward部分
    out = self.global_pooling(s1) #最后对最终的输出做一个池化
    logits = self.classifier(out.view(out.size(0),-1))  #softmax分类器得到最终结果
    return logits

  def _loss(self, input, target):
    logits = self(input)   
    return self._criterion(logits, target)   #用的是主函数传递过来的交叉熵损失函数

  def _initialize_alphas(self):   #架构参数初始化，这个还是在一个cell里面搞
    k = sum(1 for i in range(self._steps) for n in range(2+i))  #统计节点与节点之间的联系的个数
    num_ops = len(PRIMITIVES)   #操作的数量

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)   #初始化，可以看出来两类cell都是一样的
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):   #返回架构参数，在architect文件中被调用
    return self._arch_parameters

  #这个函数很细节，之前一直遗漏掉，主要是为了找出当前node的输入应该来自比他小的node的哪两个
  #选出来权重值大的两个前驱节点，并把(操作，前驱节点)存下来
  #这个还是在一个cell里面搞
  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      # range(i + 2)表示x取0，1，到i+2 x也就是前驱节点的序号 ，所以W[x]就是这个前驱节点的所有权重[α0,α1,α2,...,α7]
      # max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')) 就是把操作不是NONE的α放到一个list里，得到最大值
      # sorted 就是把每个前驱节点对应的权重最大的值进行逆序排序，然后选出来top2
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        # 把这两条边对应的最大权重的操作找到
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
        # 记录下最好的op，和对应的连接边（与哪个节点相连）
        # 对于每个节点，选择两个边和对应op，即一个cell有2*4=8个操作，定义死了，不够灵活！
          gene.append((PRIMITIVES[k_best], j))   #最好的边是哪个和经过这条边的特征图
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())  #分别生成不同类型的cell
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

