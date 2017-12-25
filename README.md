# Codelib
**Codelib** stores lots of code snippets, which is useful for building a deep-learning based system.

![](img/fig.png)

Forgive me for using Chinese.

### dataset

### augmentation
- 统一使用Compose组合所有的操作，方便使用。
- 同时提供train和test两种接口（因为像`RandomHorizontalFlip`这样的操作，在推断的时候，不需要使用）。
- 针对不同的问题，提供不同的接口操作。比如分类，可能只针对输入图像操作，目标检测还需要对2组坐标处理，显著性可能要定义各种groundtruth的形式。
- 所有的代码，均要通过测试。
- 每一类操作的接口，都必须相同。如果不同，有以下两种可能：
 - 它是其他类型的操作。
 - 它应该定义在`dataset`。

一个合理的写`augmentation`的流程是：
1. 针对具体的数据集形式，写`dataset`类，留好`transforms`和`target_transforms`接口。
2. 设计`transforms`和`target_transforms`分别要做的事情。
3. 在`augmentation.py`中写各个操作的接口定义。
4. 用`gen-test`生成测试脚本，写测试脚本。
5. 实现`augmentation.py`中的各个操作，并通过测试。
