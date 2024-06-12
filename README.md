# DEM_SRNet
AI学习框架与科学计算大作业
# Super-resolution Reconstruction of a 3 Arc-Second Global DEM Dataset

## 1. 论文介绍

### 1.1 背景介绍

全球数字高程模型（DEM）在地球科学中具有重要意义，用于地形测绘、地质研究、气候建模等领域。然而，高分辨率的全球DEM的构建面临挑战，尤其是获取高分辨率的海洋地形数据。虽然陆地的30米分辨率DEM（如NASADEM和FABDEM）已经可用，但由于海洋数据的稀缺和难以获取，全球高分辨率DEM数据仍不完善。

生成更高分辨率的全球DEM涉及多个数据源的融合，这些数据的分辨率和传感器存在差异，导致数据融合时的准确性问题。为了解决这些问题，本论文利用深度学习的超分辨率技术来提高全球DEM数据的分辨率，从而生成一个3角秒分辨率的全球DEM数据集。

### 1.2 论文方法

**《Super-resolution Reconstruction of a 3 Arc-Second Global DEM Dataset》** 提出了使用深度学习技术进行超分辨率重建的方法，以生成分辨率为3角秒的全球DEM数据集GDEM_2022。

**本文的主要方法和优势：**
- **数据融合与预处理**：论文使用NASADEM和GEBCO_2021数据集进行模型的预训练，并结合有限的高分辨率区域海洋DEM数据进行微调。所有数据集在处理前统一转换到EGM96大地水准面和WGS84坐标系，保证一致性。
- **深度残差网络（DEM-SRNet）**：提出了一个基于增强深度超分辨率网络（EDSR）的深度残差网络DEM-SRNet。该网络包含42个残差块，每个块包含两个卷积层和一个ReLU激活函数层，用于提取特征。
- **模型训练与验证**：模型通过一个包含约579,900个全球分布的陆地样本的数据集进行训练，并使用约28,500个海洋样本数据进行微调和测试，以保证模型在不同区域的泛化能力和准确性。

**实验与结果**：
- GDEM_2022的数据集在多个测试区域中（如北美、墨西哥湾、南大洋等）表现优异，相较于现有的GEBCO_2021数据集显示了更高的分辨率和精确度，证明了DEM-SRNet模型的有效性。

### 1.3 数据集介绍

论文使用的数据集包括：
- **NASADEM**：分辨率为1角秒（30米）的全球陆地DEM数据集，涵盖80%的地球陆地表面。
- **GEBCO_2021**：分辨率为15角秒的全球海洋和陆地DEM数据集。
- **区域高分辨率海洋DEM数据**：来自多个不同的高分辨率海洋DEM数据集，用于模型微调和测试，包括加拿大水文服务非航行（NONNA）数据集、墨西哥湾深水测深网格数据集、南大洋区域的地形数据等。
## 2. DEM-SRNet模型

### 2.1 模型介绍

基于深度残差网络的DEM-SRNet的预训练数据来源于地面DEM数据。如图所示，设计的预训练结构源自增强型深度超分辨率网络（EDSR）。DEM-SRNet网络的第一个卷积层提取特征集合。EDSR模型默认的残差块数量扩大到32。通过实验比较，最佳残差块数量（ResBlocks）为42，每个残差块由两个卷积层组成，并用ReLU激活函数进行插值。最后，由卷积层和逐元素加法层组成。后者包括用于提取特征图的卷积层，利用比例因子为5的插值层从输入的低分辨率15角秒数据上采样到目标高分辨率3角秒数据，最后，卷积层聚合低分辨率空间中的特征图并生成SR输出数据。与典型的反卷积层相反，使用插值函数层，可以对低分辨率数据进行超分辨率，具备更好的效果。

该网络由88个卷积层、43个元素加法层和1个插值层组成。每个卷积层中的卷积核大小设置为3，填充设置为1。总共有50,734,080个网络参数。在训练阶段，使用初始学习率为0.0001的Adam优化器和指数衰减方法使用大数据集来训练模型。当模型的性能在验证集中开始劣化时，使用为容忍度为6的早停技术来终止训练，以防止过度拟合。小批量梯度下降法通常需要300个epoch才能从头开始构建预训练网络。初始参数源自地面数据预训练网络，预训练网络的冻结层与海洋DEM结合使用，以微调全局DEM-SRNet模型。由于微调样本有限，学习率对收敛过程有显着影响，并将学习率调整为0.00001。
![](https://github.com/ouyangzehong/DEM_SRNet/blob/main/image/dem_DEMNet.png)

### 2.2 模型结构

- **输入层**：接受低分辨率的DEM影像。
- **卷积层**：输入层后的第一个卷积层，用于初步特征提取。
- **残差块**：网络由42个残差块组成，每个残差块包含两个卷积层和一个ReLU激活函数层，用于特征提取。每个卷积层后接ReLU激活函数。
- **卷积层**：残差块后的一个卷积层，用于整合提取到的特征。
- **像素清晰度层**：用于将提取的特征转换为超分辨率的DEM影像。
- **输出层**：输出超分辨率的DEM影像。
- **损失函数**：使用均方误差（MSE）作为损失函数，衡量模型预测结果与真实高分辨率影像之间的差异。
## 3. 具体实现

代码仓库：[https://github.com//ouyangzehong/DEM_SRNet]

### 3.1 环境准备

使用华为HECS(云耀云服务器)，操作系统Ubuntu 22.04。

**安装Anaconda环境**：

```bash
wget https://mirrors.bfsu.edu.cn/anaconda/archive/Anaconda3-2022.10-Linux-x86_64.sh --no-check-certificate
bash Anaconda3-2022.10-Linux-x86_64.sh
```

**创建虚拟环境并切换到环境**：
```bash
conda create -n dem_srnet python=3.7
conda activate dem_srnet
```
**克隆已经实现好的MindSpore版本DEM-SRNet代码**:
```bash
git clone https://github.com/ouyangzehong/DEM_SRNet
```
**下载依赖包**:
```bash
pip install mindspore==2.0.0a0
pip install msadapter
pip install pyyaml
pip install matplotlib
```
### 3.2 实现过程
**创建数据集**:

从dataset下载训练数据集、验证数据集、测试数据集到当前目录./dataset。
修改DEM-SRNet.yaml配置文件中的root_dir参数，该参数设置了数据集的路径。
./dataset中的目录结构如下所示：
```bash
.
├── train
│   └── train.h5
├── valid
│   └── valid.h5
├── test
│   └── test.h5

```
**模型构建**:

初始化Dem模型。

```bash
model = init_model(config)
```
**损失函数**:

DEM-SRNet使用均方误差进行模型训练。

```bash
loss_fn = nn.MSELoss()
```

**模型训练**:

DEM-SRNet使用均方误差进行模型训练。

```bash
class DemSrTrainer(Trainer):
    r"""
    Self-define forecast model inherited from `Trainer`.

    Args:
        config (dict): parameters for training.
        model (Cell): network for training.
        loss_fn (str): user-defined loss function.
        logger (logging.RootLogger): tools for logging.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self, config, model, loss_fn, logger):
        super(DemSrTrainer, self).__init__(config, model, loss_fn, logger, weather_data_source="DemSR")
        self.model = model
        self.optimizer_params = config["optimizer"]
        self.train_dataset, self.valid_dataset = self.get_dataset()
        self.optimizer = self.get_optimizer()
        self.solver = self.get_solver()

    def get_optimizer(self):
        r"""define the optimizer of the model, abstract method."""
        self.steps_per_epoch = self.train_dataset.get_dataset_size()
        if self.logger:
            self.logger.info(f'steps_per_epoch: {self.steps_per_epoch}')
        if self.optimizer_params['name']:
            optimizer = nn.Adam(self.model.trainable_params(),
                                learning_rate=Tensor(self.optimizer_params['learning_rate']))
        else:
            raise NotImplementedError(
                "self.optimizer_params['name'] not implemented, please overwrite get_optimizer()")
        return optimizer

    def get_callback(self):
        r"""define the callback of the model, abstract method."""
        pred_cb = EvaluateCallBack(self.model, self.valid_dataset, self.config, self.logger)
        return pred_cb

    def train(self):
        r""" train """
        callback_lst = [LossMonitor(), TimeMonitor(), self.ckpt_cb]
        if self.pred_cb:
            callback_lst.append(self.pred_cb)
        self.solver.train(epoch=config['data']['epoch_size'],
                          train_dataset=self.train_dataset,
                          callbacks=callback_lst,
                          dataset_sink_mode=True)
trainer = DemSrTrainer(config, model, loss_fn, logger)

```

### 3.3 参数说明

- `--config_file_path`：配置文件路径
- `--device_target`：设备类型，可选"Ascend"或"GPU"（默认"Ascend"）
- `--device_id`：设备ID（默认3）
- `--distribute`：是否分布式训练（默认False）
- `--rank_size`：设备数量（默认1）
- `--amp_level`：混合精度级别（默认'O2'）
- `--run_mode`：运行模式，可选"train"或"test"（默认'train'）
- `--load_ckpt`：是否加载检查点文件（默认False）
- `--num_workers`：数据加载器的工作线程数（默认1）
- `--epochs`：训练轮数（默认100）
- `--valid_frequency`：验证频率（默认100）
- `--output_dir`：输出目录（默认'./summary'）
- `--ckpt_path`：检查点文件路径（测试模式下需要指定）


## 4. 实验结果

trainer.train()

```bash
epoch: 1 step: 109, loss is 0.0018688203
Train epoch time: 55616.483 ms, per step time: 510.243 ms
epoch: 2 step: 109, loss is 0.0008327974
Train epoch time: 31303.473 ms, per step time: 287.188 ms
epoch: 3 step: 109, loss is 0.00022218125
...
epoch: 98 step: 109, loss is 1.3421039e-05
Train epoch time: 29786.506 ms, per step time: 273.271 ms
epoch: 99 step: 109, loss is 1.113452e-05
Train epoch time: 31082.307 ms, per step time: 285.159 ms
epoch: 100 step: 109, loss is 2.0731915e-05
Train epoch time: 30750.022 ms, per step time: 282.110 ms
```

对第100个模型的真实值和预测值进行可视化。

```bash

data = next(create_test_data)

inputs = data['inputs']
labels = data['labels']

low_res = inputs[0].asnumpy()[0].astype(np.float32)
pred = inference_module.forecast(inputs)
pred = pred[0].asnumpy()[0].astype(np.float32)
label = labels[0].asnumpy()[0].astype(np.float32)

plt.figure(num='e_imshow', figsize=(15, 36))
plt.subplot(1, 3, 1)
plt_dem_data(low_res, "Low Reslution")
plt.subplot(1, 3, 2)
plt_dem_data(label, "Ground Truth")
plt.subplot(1, 3, 3)
plt_dem_data(pred, "Prediction")
```
![](https://github.com/ouyangzehong/DEM_SRNet/blob/main/image/result.png)
