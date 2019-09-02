# adversarial_crafts说明文档

161250098 彭俊杰

#### 1.分类模型训练（Classifier）

为了考虑攻击算法在不同模型间的迁移性，我针对FashionMNIST数据集训练了两个层次结构存在差异的分类模型，其中fashionMNIST模型识别准确率差不多92%（训练数据未增强），cnn5模型识别准确率大约93%（对训练数据进行数据增强），其结构分别如下：

fashionMNIST:

![](assets\fashionMNIST.png)

cnn5:

![](assets\cnn5.png)

#### 2.失败探索（attack(failed).py）

在学习了GAN相关知识之后，准备借鉴其原理，尝试通过深度神经网络构建图片生成器，Generator直接向原始图片添加扰动生成对抗样本。大体设想如下：首先将之前训练好的分类模型作为Discriminator，冻结其各个层的权重（不在训练过程中变化），然后在输入之前添加若干个全连接层，用于调整原图片像素（添加扰动）；然后关键在于保持对抗样本与原始图片的相近性，最初我的做法是在作为generator的若干个全连接层中间均添加Lambda层（用clip对各个像素点的变化进行限制），不过由于keras模型构建的僵硬性，Lambda层在训练过程中被直接忽视了......于是我决定采取另一种做法，在自定义loss函数中添加ssim相关的项，不过又苦于无法取得ssim相关项与错误分类项的平衡，又只能放弃......

#### 3.Github开源项目foolbox（依赖Python3,NumPy和SciPy库）

在研究对抗样本生成相关的论文及github开源项目时，偶然发现foolbox——IBM's Python toolbox to create adversarial examples（https://github.com/bethgelab/foolbox），通过查阅其使用手册，发现其提供了我之前看到的各种图像攻击算法的实现，截图如下：

![](assets\Gradient-based attacks.png)

![](assets\Score-based attacks.png)

![](assets\Decision-based attacks.png)

![](assets\Other attacks.png)

通过对以上各种攻击算法的尝试，发现：

1.尽管Gradient-based类算法（FGSM,DeepFool...）具有极高的ssim(80~90)，但是对抗样本的迁移性极差，基于模型fashionMNIST生成的对抗样本就算能让fashionMNIST的正确预测类别概率降到0.1以下，却无法对模型cnn5的预测产生影响，依旧高达0.9......

2.Score-based类算法由于基于预测的梯度，在我梯度差异悬殊的两个分类模型fashionMNIST和cnn5上同样表现差强人意，对抗样本迁移性依旧不太行......

3.Other attacks是一些特殊情况下的攻击算法，同样不符合我的预期目标......

4.所幸Decision-based类算法对模型的依赖性并不强，对抗样本能较好地在模型间迁移。通过对比各个Decision-based算法生成的对抗样本的攻击成功率以及ssim指标，发现**AdditiveUniformNoiseAttack**及**AdditiveGaussianNoiseAttack**表现较好，最后通过反复对比，选择通过**AdditiveUniformNoiseAttack**实现我的adversarial_crafts，**AdditiveUniformNoiseAttack**通过向输入添加高斯噪声，逐渐增加标准偏差，直到输入被错误分类。在参考foolbox源码后，即将**AdditiveUniformNoiseAttack**代码实现整合为util包供项目使用。

#### 4.项目本地运行情况

##### 基于fashionMNIST模型利用AdditiveUniformNoiseAttack生成对抗样本，并在cnn5模型上进行迁移攻击：

生成100张对抗样本结果：

![](assets\100.png)

测试1000张对抗图片结果：

![](assets\1000.png)

ps：对抗样本生成时间约为 1.5s / 张