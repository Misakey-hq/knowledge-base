---
title: "Diffusion Policy 实机 Push-T 输入输出说明"
date: "2026-07-25"
tags:
  - robotics
  - imitation-learning
  - diffusion-policy
  - push-t
---

# 实机 Push-T 任务中的 Diffusion Policy 输入输出说明（2026-07-25）

## 1. 任务概述

在实机 Push-T 任务中，机械臂通过末端安装的圆柱形推杆，在桌面平面上推动一个 T 形物体，使其移动并旋转到目标区域。

典型系统包括：

- 一台顶视 RGB 相机；
- 一台机械臂；
- 一个末端圆柱推杆；
- 桌面上的 T 形物体；
- 一个目标区域；
- Diffusion Policy 策略模型；
- 机械臂底层位置或速度控制器。

标准 Push-T 是二维任务。迁移到实机后，通常让机械臂末端保持固定高度和固定姿态，只控制推杆在桌面平面内的 \(x\)、\(y\) 运动。

---

## 2. 整个策略的输入与输出

从系统层面看，Diffusion Policy 完成的映射是：

\[
\boxed{
\text{最近若干步观测}
\longrightarrow
\text{未来一段动作序列}
}
\]

以常见配置为例：

\[
T_o=2,\qquad T_p=16,\qquad T_a=8
\]

其中：

- \(T_o=2\)：输入最近 2 个时刻的观测；
- \(T_p=16\)：预测未来 16 个动作；
- \(T_a=8\)：实际执行预测动作中的前 8 个。

因此可以简化为：

\[
\boxed{
\text{最近两帧图像}
+
\text{最近两个推杆位置}
\longrightarrow
\text{未来 16 个推杆目标位置}
}
\]

机器人执行前 8 个动作后，会重新采集观测并再次预测，这种方式称为滚动时域控制（Receding-Horizon Control）。

---

## 3. 输入一：相机图像

在控制时刻 \(t\)，模型输入最近两帧图像：

\[
I_{t-1},\quad I_t
\]

图像中通常包含：

- T 形物体的位置；
- T 形物体的朝向；
- 机械臂推杆的位置；
- 目标区域的位置；
- 推杆与物体之间的接触关系；
- 物体当前的平移和旋转趋势。

假设图像经过裁剪和缩放后尺寸为：

\[
I_t\in\mathbb{R}^{3\times96\times96}
\]

两帧图像组合后的形状为：

\[
\text{image}\in\mathbb{R}^{2\times3\times96\times96}
\]

加入 Batch 维度后：

```text
image.shape = [B, 2, 3, 96, 96]
```

其中：

- \(B\) 表示 Batch Size；
- 2 表示两个观测时刻；
- 3 表示 RGB 三个通道；
- 96 × 96 表示图像分辨率。

### 为什么使用两帧图像？

单张图像主要反映当前状态，而两张连续图像还可以帮助模型判断：

- 推杆正在向哪个方向移动；
- T 形物体是否正在运动；
- 物体是在平移还是旋转；
- 推杆是否已经接触物体；
- 推杆与物体之间是否发生打滑；
- 上一个动作对物体产生了什么影响。

---

## 4. 输入二：机器人自身状态

除图像外，模型还会输入最近两个时刻的推杆二维位置。

在时刻 \(t-1\)：

\[
p_{t-1}=(x_{t-1},y_{t-1})
\]

在时刻 \(t\)：

\[
p_t=(x_t,y_t)
\]

组合后：

\[
\text{agent\_pos}
=
\begin{bmatrix}
x_{t-1}&y_{t-1}\\
x_t&y_t
\end{bmatrix}
\in\mathbb{R}^{2\times2}
\]

加入 Batch 维度后：

```text
agent_pos.shape = [B, 2, 2]
```

在实机中，推杆位置一般可以由机械臂关节状态通过正运动学获得：

\[
q_t
\xrightarrow{\text{Forward Kinematics}}
(x_t,y_t,z_t)
\]

因为 Push-T 通常只在桌面平面内运动，所以策略只保留：

\[
(x_t,y_t)
\]

而推杆高度 \(z\) 和末端姿态通常保持不变。

---

## 5. 一次完整的观测输入

在控制时刻 \(t\)，完整观测可以写成：

\[
O_t=
\left\{
I_{t-1},I_t,\;
p_{t-1},p_t
\right\}
\]

对应的伪代码形式如下：

```python
obs = {
    "image": [
        image_t_minus_1,
        image_t
    ],  # shape: [2, 3, 96, 96]

    "agent_pos": [
        [x_t_minus_1, y_t_minus_1],
        [x_t, y_t]
    ]   # shape: [2, 2]
}
```

在图像版本的 Push-T 中，通常不会直接向模型输入 T 形物体的真实坐标和角度。

也就是说，模型需要自行从 RGB 图像中识别：

- T 形物体在哪里；
- T 形物体朝向哪里；
- 目标区域在哪里；
- 推杆应该从哪个方向接触物体。

---

## 6. 策略最终输出什么？

Diffusion Policy 最终输出一段未来动作序列：

\[
A_t=
[a_t,a_{t+1},\ldots,a_{t+15}]
\]

每个动作是一个二维推杆目标位置：

\[
a_i=
(x_i^{\text{target}},y_i^{\text{target}})
\]

因此，完整输出的形状为：

\[
A_t\in\mathbb{R}^{16\times2}
\]

加入 Batch 维度后：

```text
action_pred.shape = [B, 16, 2]
```

例如，模型可能输出：

```text
[
  [0.42, 0.71],
  [0.43, 0.69],
  [0.44, 0.66],
  [0.46, 0.63],
  ...
  [0.58, 0.40]
]
```

这些数值表示推杆未来应该依次到达的二维目标位置。

需要特别注意：

\[
\boxed{
\text{模型输出的是推杆轨迹，而不是 T 形物体的未来轨迹}
}
\]

图像告诉模型物体和目标区域在哪里，策略根据这些信息生成推杆应该如何移动。

---

## 7. 输出动作的具体含义

在标准 Push-T 任务中，每个动作通常表示推杆的绝对二维目标位置：

\[
a_t=
(x_t^{\text{target}},y_t^{\text{target}})
\]

控制器根据当前位置和目标位置之间的误差，驱动推杆靠近目标点。

一个简化的 PD 控制形式为：

\[
\ddot p
=
K_p(a_t-p)
+
K_v(0-\dot p)
\]

其中：

- \(a_t\)：目标位置；
- \(p\)：当前推杆位置；
- \(\dot p\)：当前推杆速度；
- \(K_p\)：位置增益；
- \(K_v\)：速度增益。

在自定义实机系统中，也可以把动作定义为：

### 相对位移

\[
a_t=(\Delta x_t,\Delta y_t)
\]

### 平面速度

\[
a_t=(v_{x,t},v_{y,t})
\]

### 末端位姿增量

\[
a_t=(\Delta x_t,\Delta y_t,\Delta z_t,\Delta r_t)
\]

但是，无论采用哪种定义，训练数据和部署系统必须保持一致。

例如：

- 示教数据记录绝对位置，模型输出就必须按绝对位置解释；
- 示教数据记录相对位移，模型输出就必须按相对位移解释；
- 训练时使用米，部署时也必须使用米；
- 训练时使用归一化坐标，部署时需要执行相同的归一化和反归一化。

---

## 8. Diffusion 网络内部的输入与输出

从整个策略看：

\[
O_t\rightarrow A_t
\]

但 Diffusion Policy 并不是通过一次前向传播直接得到最终动作。

它从一段随机噪声动作开始，通过多次去噪逐步生成合理的动作序列。

---

## 9. 初始化随机动作序列

首先生成一段随机高斯噪声：

\[
A_t^K\sim\mathcal N(0,I)
\]

其形状与最终动作序列相同：

\[
A_t^K\in\mathbb{R}^{16\times2}
\]

例如：

```text
[
  [-0.81,  1.24],
  [ 0.35, -0.62],
  [ 1.10,  0.48],
  ...
]
```

这些随机值一开始没有实际控制意义。

---

## 10. 单次去噪网络的输入

在第 \(k\) 个扩散步骤中，网络通常接收三个输入：

\[
\boxed{
A_t^k,\quad k,\quad O_t
}
\]

分别表示：

1. 当前带噪动作序列 \(A_t^k\)；
2. 当前扩散时间步 \(k\)；
3. 当前观测条件 \(O_t\)。

展开后，网络输入包括：

- 带噪的 \(16\times2\) 动作序列；
- 扩散步骤编号；
- 最近两帧 RGB 图像；
- 最近两个推杆二维位置。

伪代码如下：

```python
noise_pred = model(
    noisy_action,       # [B, 16, 2]
    diffusion_step,     # [B]
    observation         # image + agent_pos
)
```

---

## 11. 单次去噪网络的输出

最常见的 Diffusion Policy 网络输出是预测噪声：

\[
\hat\epsilon_\theta
=
\epsilon_\theta(A_t^k,k,O_t)
\]

预测噪声的形状和带噪动作序列相同：

\[
\hat\epsilon_\theta\in\mathbb{R}^{16\times2}
\]

加入 Batch 维度后：

```text
noise_pred.shape = [B, 16, 2]
```

采样器根据网络预测的噪声，将当前动作序列更新为噪声更少的动作序列：

\[
A_t^k\rightarrow A_t^{k-1}
\]

重复多次后：

\[
A_t^K
\rightarrow
A_t^{K-1}
\rightarrow
\cdots
\rightarrow
A_t^0
\]

最终得到的 \(A_t^0\) 就是可执行的动作轨迹。

因此，单次去噪网络的输入输出可以总结为：

\[
\boxed{
\text{带噪动作序列}
+
\text{扩散步}
+
\text{观测条件}
\longrightarrow
\text{预测噪声}
}
\]

而整个 Diffusion Policy 的输入输出是：

\[
\boxed{
\text{机器人观测}
\longrightarrow
\text{未来动作序列}
}
\]

---

## 12. 模型输出如何变成实机运动？

假设最终生成的动作序列为：

\[
A_t^0=
\begin{bmatrix}
x_t&y_t\\
x_{t+1}&y_{t+1}\\
\vdots&\vdots\\
x_{t+15}&y_{t+15}
\end{bmatrix}
\]

模型预测了 16 个动作，但系统通常只执行前 8 个：

\[
A_t^{\text{exec}}
=
[a_t,a_{t+1},\ldots,a_{t+7}]
\]

对每个二维目标点：

\[
(x_i,y_i)
\]

补上固定高度和固定姿态：

\[
T_i=
(x_i,y_i,z_{\text{push}},R_{\text{fixed}})
\]

其中：

- \(z_{\text{push}}\)：推杆接触桌面物体时的固定高度；
- \(R_{\text{fixed}}\)：机械臂末端的固定旋转姿态。

之后，动作经过以下转换：

```text
Diffusion Policy 输出二维目标点 (x, y)
                  ↓
加入固定高度 z 和固定姿态 R
                  ↓
生成末端目标位姿
                  ↓
逆运动学或笛卡尔控制器
                  ↓
生成关节位置、关节速度或力矩指令
                  ↓
机械臂执行运动
```

因此，Diffusion Policy 通常不直接输出：

- 电机电流；
- 电机电压；
- 关节力矩；
- 底层伺服器指令。

它输出的是较高层的动作目标，底层机械臂控制器负责跟踪这些目标。

---

## 13. 完整的实机运行流程

假设外层策略控制频率为 10 Hz。

### 第一次预测

在时刻 \(t=10\)，系统收集：

\[
I_9,\quad I_{10},\quad p_9,\quad p_{10}
\]

模型预测：

\[
a_{10},a_{11},\ldots,a_{25}
\]

系统只执行：

\[
a_{10},a_{11},\ldots,a_{17}
\]

### 第二次预测

执行完前 8 个动作后，在时刻 \(t=18\) 再次收集：

\[
I_{17},\quad I_{18},\quad p_{17},\quad p_{18}
\]

模型重新预测：

\[
a_{18},a_{19},\ldots,a_{33}
\]

然后继续执行前 8 个动作。

整个闭环过程如下：

```text
采集最近两帧图像和两个推杆位置
                  ↓
Diffusion Policy 预测未来 16 步
                  ↓
执行前 8 步
                  ↓
重新采集观测
                  ↓
重新预测未来动作
                  ↓
循环执行，直到任务完成
```

---

## 14. 输入输出总结表

| 层级 | 输入 | 输出 |
|---|---|---|
| 整个策略 | 最近两帧图像、最近两个推杆位置 | 未来 16 个推杆二维目标位置 |
| 单次 Diffusion 去噪 | 带噪动作序列、扩散步、观测条件 | 预测噪声 |
| 动作执行模块 | 推杆二维目标位置 | 机械臂末端目标位姿 |
| 底层机器人控制器 | 末端目标位姿 | 关节位置、速度或力矩指令 |

---

## 15. 核心公式

### 策略整体

\[
\boxed{
\begin{aligned}
\text{输入：}\quad&
I_{t-1},I_t,\;
(x_{t-1},y_{t-1}),\;
(x_t,y_t)
\\[2mm]
\text{输出：}\quad&
[(x_t^*,y_t^*),\ldots,(x_{t+15}^*,y_{t+15}^*)]
\end{aligned}
}
\]

### Diffusion 单次去噪

\[
\boxed{
\begin{aligned}
\text{输入：}\quad&
A_t^k,\;k,\;O_t
\\
\text{输出：}\quad&
\hat\epsilon_\theta
\end{aligned}
}
\]

### 实机执行转换

\[
\boxed{
(x_i^*,y_i^*)
\rightarrow
(x_i^*,y_i^*,z_{\text{fixed}},R_{\text{fixed}})
\rightarrow
\text{机械臂控制器}
}
\]

---

## 16. 最重要的理解

在实机 Push-T 任务中：

- 相机图像告诉模型 T 形物体、推杆和目标区域在哪里；
- `agent_pos` 告诉模型推杆自身的准确二维位置；
- Diffusion Policy 输出推杆未来应该走过的一串二维位置；
- 输出动作经过坐标转换后，变成机械臂末端目标位姿；
- 底层控制器负责将末端目标位姿转换为真实的机械臂运动；
- 模型预测的是推杆动作，不是 T 形物体的目标轨迹；
- 系统通过“观测—预测—执行—再观测”形成闭环控制。

最终可用一句话概括：

\[
\boxed{
\text{图像与机器人状态}
\longrightarrow
\text{推杆未来动作序列}
\longrightarrow
\text{机械臂真实运动}
}
\]

---

## 参考资料

- Diffusion Policy 项目主页与论文：<https://diffusion-policy.cs.columbia.edu/>
- Diffusion Policy 官方代码仓库：<https://github.com/real-stanford/diffusion_policy>
