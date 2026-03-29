# 分布式 4 层 MLP（8×H100：DP=2，PP=2，TP=2，可选 SP）

## 环境要求

- PyTorch（NCCL 后端）

## 运行方式（单机 8 卡）

使用模块方式（确保包导入正确）：

```bash
./launcher.sh
```

## 代码结构

- `dist_mlp/distributed_setup.py`：初始化分布式与 DP/PP/TP 分组工具。
- `dist_mlp/tensor_parallel.py`：TP 线性层（列/行并行）与基础张量切分函数。
- `dist_mlp/mlp_model.py`：4 层 MLP 的两段式实现（Stage0/Stage1）。
- `dist_mlp/pipeline.py`：极简两阶段流水线的点对点发送/接收与邻居推导。
- `dist_mlp/train.py`：训练循环（合成数据、微批次、MSE 损失、DP 梯度同步）。
- `launcher.sh`：基于 torchrun 的启动脚本（可通过环境变量覆盖参数）。
- `requirements.txt`：项目依赖。

# 只有TP 框架

模型结构：（layernorm-fc1-gelu-fc2）\*4

序号划分先PP，再DP，再TP

```
PP0 0 1 2 3
PP1  4 5 6 7
```

1、生成数据/导入数据，通过控制生成种子保证每个tp组得到的数据相同,相同dp组初始参数相同

2、forward，backward，分类讨论pp_size=1，pp_size!=1,分为开始、中间、结尾三个部分

- 用isend、irec+tag的方式实现通讯功能

## 验证：

### DP验证

dp=2 tp=2 pp=2 step=0 cuda=0 结果保存在 results/step_0_model_222_0.pt 中

1、all_reduce后 dp组中模型参数相同。(check.ipynb 1 finish)

问题如下：数据生成用的随机种子和dp有关，所以无法验证

考虑修改：在rank=0上生成batch_size发给其他

修改完成，数据一致性检查完成

2 2 2 结果保存在input文件夹中，{dp*index}*{step}_{input} {dp_index}_{step}_{target} all_{step}

4 1 2 结果保存在input2文件夹中 dp分割等等结果一致

解决问题：tp=1，pp=1 dp=1和dp=4的loss曲线不同

1、检查输入：check_input_same.py 比较每个dp=1（input_dp1_pp1_tp1_mb4）和dp=4（input_dp4_pp1_tp1_mb4）输入拼接后的结果，对齐

2、检查forward结果：check_dp_a_mb_same.py 比较每个dp=1（dp_test_dp1_tp1_pp1）和dp=4（dp_test_dp4_tp1_pp1）输入拼接后的结果，有误差但误差很小

以下验证均在在fp32环境下，step=50

```
[CHK ] step=46 mb=1 equal=False allclose=False shape_eq=True max|diff|=1.735e-15 mean|diff|=4.449e-16 | unbiased_ok=True |mean(diff)|=9.559e-17 se=2.548e-18 z=3.0 n=65536
[CHK ] step=46 mb=2 equal=False allclose=False shape_eq=True max|diff|=1.552e-15 mean|diff|=3.921e-16 | unbiased_ok=True |mean(diff)|=1.621e-16 se=2.122e-18 z=3.0 n=65536
[CHK ] step=46 mb=3 equal=False allclose=False shape_eq=True max|diff|=1.759e-15 mean|diff|=5.566e-16 | unbiased_ok=True |mean(diff)|=3.804e-17 se=3.152e-18 z=3.0 n=65536
[CHK ] step=47 mb=0 equal=False allclose=False shape_eq=True max|diff|=2.421e-15 mean|diff|=6.997e-16 | unbiased_ok=True |mean(diff)|=1.085e-16 se=3.869e-18 z=3.0 n=65536
[CHK ] step=47 mb=1 equal=False allclose=False shape_eq=True max|diff|=2.421e-15 mean|diff|=6.467e-16 | unbiased_ok=True |mean(diff)|=2.215e-16 se=3.700e-18 z=3.0 n=65536
[CHK ] step=47 mb=2 equal=False allclose=False shape_eq=True max|diff|=2.558e-15 mean|diff|=5.967e-16 | unbiased_ok=True |mean(diff)|=8.343e-17 se=3.401e-18 z=3.0 n=65536
[CHK ] step=47 mb=3 equal=False allclose=False shape_eq=True max|diff|=2.558e-15 mean|diff|=4.747e-16 | unbiased_ok=True |mean(diff)|=2.962e-17 se=2.997e-18 z=3.0 n=65536
[CHK ] step=48 mb=0 equal=False allclose=False shape_eq=True max|diff|=2.416e-15 mean|diff|=5.762e-16 | unbiased_ok=True |mean(diff)|=7.906e-18 se=3.430e-18 z=3.0 n=65536
[CHK ] step=48 mb=1 equal=False allclose=False shape_eq=True max|diff|=2.135e-15 mean|diff|=5.186e-16 | unbiased_ok=True |mean(diff)|=5.116e-17 se=2.957e-18 z=3.0 n=65536
[CHK ] step=48 mb=2 equal=False allclose=False shape_eq=True max|diff|=2.105e-15 mean|diff|=6.259e-16 | unbiased_ok=True |mean(diff)|=1.419e-16 se=3.320e-18 z=3.0 n=65536
[CHK ] step=48 mb=3 equal=False allclose=False shape_eq=True max|diff|=2.416e-15 mean|diff|=6.330e-16 | unbiased_ok=True |mean(diff)|=9.867e-17 se=3.547e-18 z=3.0 n=65536
[CHK ] step=49 mb=0 equal=False allclose=False shape_eq=True max|diff|=2.141e-15 mean|diff|=5.493e-16 | unbiased_ok=True |mean(diff)|=1.766e-16 se=3.141e-18 z=3.0 n=65536
[CHK ] step=49 mb=1 equal=False allclose=False shape_eq=True max|diff|=3.907e-15 mean|diff|=6.874e-16 | unbiased_ok=True |mean(diff)|=2.690e-16 se=4.149e-18 z=3.0 n=65536
[CHK ] step=49 mb=2 equal=False allclose=False shape_eq=True max|diff|=1.437e-15 mean|diff|=4.354e-16 | unbiased_ok=True |mean(diff)|=7.436e-17 se=2.418e-18 z=3.0 n=65536
[CHK ] step=49 mb=3 equal=False allclose=False shape_eq=True max|diff|=3.907e-15 mean|diff|=5.594e-16 | unbiased_ok=True |mean(diff)|=3.712e-16 se=3.444e-18 z=3.0 n=65536
[SUM ] total=200 passed_equal/allclose=0 failed_equal/allclose=200 unbiased_pass=200 unbiased_fail=0 (rtol=0.0, atol=0.0, z=3.0)
[TRIALS] unbiased over pairs: ok=True mean=3.711e-23 se=1.067e-17 n=200
```

验证通过

3、检查loss结果，绘制图像肉眼观察发现基本相同 python check_dp_loss_same.py 比较dp_test_dp1_tp1_pp1/loss_rank0.txt和dp_test_dp4_tp1_pp1/loss_rank0.txt

```
[STEP] idx=0 a=5.374570e-01 b=5.374570e-01 diff=0.000e+00 allclose=True
[STEP] idx=1 a=4.816674e-01 b=4.816673e-01 diff=-5.960e-08 allclose=False
[STEP] idx=2 a=5.153566e-01 b=5.153566e-01 diff=0.000e+00 allclose=True
[STEP] idx=3 a=4.753829e-01 b=4.753828e-01 diff=-2.980e-08 allclose=False
[STEP] idx=4 a=5.230186e-01 b=5.230185e-01 diff=-5.960e-08 allclose=False
[STEP] idx=5 a=4.857151e-01 b=4.857151e-01 diff=0.000e+00 allclose=True
[STEP] idx=6 a=5.468459e-01 b=5.468458e-01 diff=-5.960e-08 allclose=False
[STEP] idx=7 a=4.664428e-01 b=4.664429e-01 diff=2.980e-08 allclose=False
[STEP] idx=8 a=5.063280e-01 b=5.063281e-01 diff=5.960e-08 allclose=False
[STEP] idx=9 a=4.909006e-01 b=4.909006e-01 diff=0.000e+00 allclose=True
[STEP] idx=10 a=5.358157e-01 b=5.358156e-01 diff=-5.960e-08 allclose=False
[STEP] idx=11 a=4.942738e-01 b=4.942738e-01 diff=-2.980e-08 allclose=False
[STEP] idx=12 a=4.713843e-01 b=4.713843e-01 diff=0.000e+00 allclose=True
[STEP] idx=13 a=4.997166e-01 b=4.997166e-01 diff=0.000e+00 allclose=True
[STEP] idx=14 a=4.657712e-01 b=4.657712e-01 diff=0.000e+00 allclose=True
[STEP] idx=15 a=4.787969e-01 b=4.787969e-01 diff=2.980e-08 allclose=False
[STEP] idx=16 a=5.083370e-01 b=5.083369e-01 diff=-5.960e-08 allclose=False
[STEP] idx=17 a=5.086830e-01 b=5.086830e-01 diff=0.000e+00 allclose=True
[STEP] idx=18 a=4.997192e-01 b=4.997191e-01 diff=-5.960e-08 allclose=False
[STEP] idx=19 a=4.991847e-01 b=4.991846e-01 diff=-8.941e-08 allclose=False
[STEP] idx=20 a=4.384495e-01 b=4.384495e-01 diff=0.000e+00 allclose=True
[STEP] idx=21 a=4.904537e-01 b=4.904537e-01 diff=-2.980e-08 allclose=False
[STEP] idx=22 a=4.954126e-01 b=4.954126e-01 diff=0.000e+00 allclose=True
[STEP] idx=23 a=4.808428e-01 b=4.808428e-01 diff=-2.980e-08 allclose=False
[STEP] idx=24 a=4.899096e-01 b=4.899096e-01 diff=-2.980e-08 allclose=False
[STEP] idx=25 a=4.513537e-01 b=4.513537e-01 diff=2.980e-08 allclose=False
[STEP] idx=26 a=4.908710e-01 b=4.908710e-01 diff=-2.980e-08 allclose=False
[STEP] idx=27 a=5.189685e-01 b=5.189685e-01 diff=0.000e+00 allclose=True
[STEP] idx=28 a=5.205300e-01 b=5.205300e-01 diff=5.960e-08 allclose=False
[STEP] idx=29 a=5.356041e-01 b=5.356041e-01 diff=0.000e+00 allclose=True
[STEP] idx=30 a=4.998430e-01 b=4.998430e-01 diff=-2.980e-08 allclose=False
[STEP] idx=31 a=4.686161e-01 b=4.686161e-01 diff=0.000e+00 allclose=True
[STEP] idx=32 a=5.334889e-01 b=5.334889e-01 diff=0.000e+00 allclose=True
[STEP] idx=33 a=5.730983e-01 b=5.730983e-01 diff=0.000e+00 allclose=True
[STEP] idx=34 a=4.989186e-01 b=4.989187e-01 diff=2.980e-08 allclose=False
[STEP] idx=35 a=4.795954e-01 b=4.795954e-01 diff=2.980e-08 allclose=False
[STEP] idx=36 a=4.368824e-01 b=4.368824e-01 diff=-2.980e-08 allclose=False
[STEP] idx=37 a=5.221871e-01 b=5.221871e-01 diff=5.960e-08 allclose=False
[STEP] idx=38 a=5.265297e-01 b=5.265297e-01 diff=0.000e+00 allclose=True
[STEP] idx=39 a=4.893988e-01 b=4.893987e-01 diff=-5.960e-08 allclose=False
[STEP] idx=40 a=5.358990e-01 b=5.358990e-01 diff=0.000e+00 allclose=True
[STEP] idx=41 a=5.233994e-01 b=5.233994e-01 diff=0.000e+00 allclose=True
[STEP] idx=42 a=5.337992e-01 b=5.337992e-01 diff=-5.960e-08 allclose=False
[STEP] idx=43 a=4.836189e-01 b=4.836189e-01 diff=0.000e+00 allclose=True
[STEP] idx=44 a=5.532579e-01 b=5.532579e-01 diff=0.000e+00 allclose=True
[STEP] idx=45 a=5.014980e-01 b=5.014980e-01 diff=-5.960e-08 allclose=False
[STEP] idx=46 a=4.927768e-01 b=4.927768e-01 diff=0.000e+00 allclose=True
[STEP] idx=47 a=5.411624e-01 b=5.411625e-01 diff=5.960e-08 allclose=False
[STEP] idx=48 a=5.127928e-01 b=5.127928e-01 diff=5.960e-08 allclose=False
[STEP] idx=49 a=4.707188e-01 b=4.707188e-01 diff=2.980e-08 allclose=False
[RANGE] compare steps [0, 50) out of 50
[DIFF ] max|diff|=8.940697e-08 mean|diff|=2.682209e-08 mse=1.403322e-15 (rtol=0.0, atol=0.0)
[ALLC ] allclose_all=False passed=21/50
[UNB  ] unbiased_ok=True mean(diff)=-7.748604e-09 se=5.235828e-09 std=3.702290e-08 n=50 z=3.0
```

4、检查backward结果，check_dp_grad_same.py 检查每个dp=1（dp_test_dp1_tp1_pp1）和dp=4（dp_test_dp4_tp1_pp1）输入拼接后的结果

数据分组的结果为

```
DP=1 mb0+mb1+mb2+mb3
DP=4 (DP0.mb0+mb1+mb2+mb3)+(DP1.mb0+mb1+mb2+mb3)+(DP2.mb0+mb1+mb2+mb3)+(DP3.mb0+mb1+mb2+mb3)
```

grad是前缀和形式,从loss比较的角度相对合理

grad需要比较 mb0 和 DP0.mb3 ，mb1 和 DP0.mb3+DP1.mb3 ，mb2 和 DP0+DP1+DP2.mb3,mb3和 DP0+DP1+DP2+DP3.mb3

而代码实现中mb0grad保存的是实际值/mb，所以比较是前者需要乘上DPsize()

```
[CHK ] step=0 mb=0 equal=False allclose=False shape_eq=True max|diff|=7.276e-12 mean|diff|=3.704e-16 | unbiased_ok=True |mean(diff)|=3.650e-16 se=5.407e-18 z=3.0 n=67133440
[CHK ] step=0 mb=1 equal=False allclose=False shape_eq=True max|diff|=5.230e-12 mean|diff|=3.215e-16 | unbiased_ok=True |mean(diff)|=3.167e-16 se=4.986e-18 z=3.0 n=67133440
[CHK ] step=0 mb=2 equal=False allclose=False shape_eq=True max|diff|=2.910e-11 mean|diff|=1.197e-15 | unbiased_ok=True |mean(diff)|=1.189e-15 se=2.019e-17 z=3.0 n=67133440
[CHK ] step=0 mb=3 equal=False allclose=False shape_eq=True max|diff|=3.638e-11 mean|diff|=1.745e-15 | unbiased_ok=True |mean(diff)|=1.745e-15 se=2.643e-17 z=3.0 n=67133440
[CHK ] step=1 mb=0 equal=False allclose=False shape_eq=True max|diff|=7.276e-12 mean|diff|=2.371e-16 | unbiased_ok=True |mean(diff)|=2.094e-16 se=4.910e-18 z=3.0 n=67133440
[CHK ] step=1 mb=1 equal=False allclose=False shape_eq=True max|diff|=7.276e-12 mean|diff|=3.647e-16 | unbiased_ok=True |mean(diff)|=3.647e-16 se=5.363e-18 z=3.0 n=67133440
[CHK ] step=1 mb=2 equal=False allclose=False shape_eq=True max|diff|=2.183e-11 mean|diff|=7.011e-16 | unbiased_ok=True |mean(diff)|=6.456e-16 se=1.472e-17 z=3.0 n=67133440
[CHK ] step=1 mb=3 equal=False allclose=False shape_eq=True max|diff|=2.910e-11 mean|diff|=9.202e-16 | unbiased_ok=True |mean(diff)|=8.642e-16 se=1.962e-17 z=3.0 n=67133440
[CHK ] step=2 mb=0 equal=False allclose=False shape_eq=True max|diff|=1.455e-11 mean|diff|=7.252e-16 | unbiased_ok=True |mean(diff)|=1.629e-16 se=1.073e-17 z=3.0 n=67133440
[CHK ] step=2 mb=1 equal=False allclose=False shape_eq=True max|diff|=2.910e-11 mean|diff|=1.281e-15 | unbiased_ok=True |mean(diff)|=4.950e-16 se=2.054e-17 z=3.0 n=67133440
[CHK ] step=2 mb=2 equal=False allclose=False shape_eq=True max|diff|=2.910e-11 mean|diff|=1.595e-15 | unbiased_ok=True |mean(diff)|=1.812e-16 se=2.242e-17 z=3.0 n=67133440
[CHK ] step=2 mb=3 equal=False allclose=False shape_eq=True max|diff|=2.910e-11 mean|diff|=1.743e-15 | unbiased_ok=True |mean(diff)|=3.294e-17 se=2.360e-17 z=3.0 n=67133440
[CHK ] step=3 mb=0 equal=False allclose=False shape_eq=True max|diff|=2.387e-12 mean|diff|=1.512e-16 | unbiased_ok=True |mean(diff)|=1.512e-16 se=2.279e-18 z=3.0 n=67133440
[CHK ] step=3 mb=1 equal=False allclose=False shape_eq=True max|diff|=2.183e-11 mean|diff|=1.007e-15 | unbiased_ok=True |mean(diff)|=1.007e-15 se=1.557e-17 z=3.0 n=67133440
[CHK ] step=3 mb=2 equal=False allclose=False shape_eq=True max|diff|=2.910e-11 mean|diff|=1.133e-15 | unbiased_ok=True |mean(diff)|=1.133e-15 se=1.994e-17 z=3.0 n=67133440
[CHK ] step=3 mb=3 equal=False allclose=False shape_eq=True max|diff|=1.705e-12 mean|diff|=1.107e-16 | unbiased_ok=True |mean(diff)|=1.107e-16 se=1.632e-18 z=3.0 n=67133440
[CHK ] step=4 mb=0 equal=False allclose=False shape_eq=True max|diff|=1.478e-12 mean|diff|=9.201e-17 | unbiased_ok=True |mean(diff)|=9.201e-17 se=1.409e-18 z=3.0 n=67133440
[CHK ] step=4 mb=1 equal=False allclose=False shape_eq=True max|diff|=5.457e-12 mean|diff|=1.777e-16 | unbiased_ok=True |mean(diff)|=1.638e-16 se=3.681e-18 z=3.0 n=67133440
[CHK ] step=4 mb=2 equal=False allclose=False shape_eq=True max|diff|=9.095e-12 mean|diff|=3.478e-16 | unbiased_ok=True |mean(diff)|=2.071e-16 se=6.202e-18 z=3.0 n=67133440
……
[CHK ] step=49 mb=0 equal=False allclose=False shape_eq=True max|diff|=3.638e-12 mean|diff|=1.720e-16 | unbiased_ok=True |mean(diff)|=6.533e-17 se=2.595e-18 z=3.0 n=67133440
[CHK ] step=49 mb=1 equal=False allclose=False shape_eq=True max|diff|=3.638e-12 mean|diff|=1.675e-16 | unbiased_ok=True |mean(diff)|=7.734e-17 se=2.564e-18 z=3.0 n=67133440
[CHK ] step=49 mb=2 equal=False allclose=False shape_eq=True max|diff|=3.638e-12 mean|diff|=1.373e-16 | unbiased_ok=True |mean(diff)|=1.373e-16 se=2.477e-18 z=3.0 n=67133440
[CHK ] step=49 mb=3 equal=False allclose=False shape_eq=True max|diff|=7.276e-12 mean|diff|=4.272e-16 | unbiased_ok=True |mean(diff)|=1.674e-17 se=5.703e-18 z=3.0 n=67133440
[SUM ] total=200 passed_equal/allclose=0 failed_equal/allclose=200 unbiased_pass=200 unbiased_fail=0 (rtol=0.0, atol=0.0, z=3.0)
[TRIALS] unbiased over pairs: ok=True mean=-4.607e-17 se=4.960e-17 n=200
```

PS：该验证最后每个step对mb做前缀和 可能会影响无偏性

### TP验证

分别验证ColumnParallel和RowParallel前向和反向传播，见dist_mlp/tests/test_tp.py

使用方法：

```bash
cd /mnt/user-ssd/wangshengfan
source .venv/bin/activate
./dist_mlp/tests/test_tp.sh
```

用z检测验证每组数据误差为正态分布，所有误差的平均值迭代1w次也呈现正太分布

结果：

```
=== TP test summary over 10000 random seeds ===
- ColumnParallel Forward:  unbiased_pass=1.000, avg|mean|=0.000e+00, avg max|diff|=0.000e+00
- ColumnParallel Grad(W): unbiased_pass=0.989, avg|mean|=5.910e-02, avg max|diff|=4.060e+01
- RowParallel    Forward:  unbiased_pass=0.996, avg|mean|=8.114e-05, avg max|diff|=9.688e-03
- RowParallel    Grad(W): unbiased_pass=0.993, avg|mean|=8.920e-05, avg max|diff|=7.564e-02
=== Unbiasedness over trials (|mean| <= z*SE) ===
col_fwd_mean: ok=True mean=0.000e+00 se=0.000e+00 z=3.0
col_grad_mean: ok=True mean=-8.186e-04 se=7.976e-04 z=3.0
row_fwd_mean: ok=True mean=-7.890e-07 se=1.027e-06 z=3.0
row_grad_mean: ok=True mean=-7.864e-07 se=1.166e-06 z=3.0
[TP Test] 结果已保存: dist_mlp/test_result/result_test_tp.xlsx
[TP Test] 误差曲线已保存: dist_mlp/test_result/tp_test_mean_error.png
```

### PP验证

验证pp=1和pp=2和pp=4是否一致，以pp=1为基准

1） 先验证tp=dp=1，在fp32环境下

肉眼发现pp=1，pp=2，pp=4结果相同,测试1000组发现结果完全相同

2） Tp=2，Dp=2，Pp=2结果和TP=2，DP=2，PP=1相同

### 使用 nsys 分析程序性能

cloudml上使用wandb疑似出现连不上的情况，需要关闭wandb

```bash
nsys profile --gpu-metrics-devices all -o report_dp2tp2pp2 ./launcher.sh
```

结果保存在 report_dp2tp2pp2_bf16.nsys-rep

### 显存占用

支持snapshot功能保存显存快照

对显存的分析见飞书

# 修改框架

加入Pre-layernorm层

# 加入sp

通讯原语部分：reduce-scatter会先压成一维，再进行reduce-scatter

所以有时候需要调整维度
