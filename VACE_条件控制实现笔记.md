# VACE 条件控制生成实现笔记

## 1. 论文信息

- 论文: `VACE: All-in-One Video Creation and Editing`
- arXiv: `2503.07598`
- 仓库首页在 [README.md](/Users/longxiao/Documents/projects/video_gen/VACE/README.md:1) 给出了论文和项目页链接。

这篇工作的核心目标不是只做某一种 control，而是把多种视频生成/编辑任务统一到一套输入接口里:

- `prompt`
- `src_video`
- `src_mask`
- `src_ref_images`

这一点在 [README.md](/Users/longxiao/Documents/projects/video_gen/VACE/README.md:78) 和 [UserGuide.md](/Users/longxiao/Documents/projects/video_gen/VACE/UserGuide.md:31) 里写得很明确。


## 2. 先说结论: VACE 的条件控制是怎么实现的

VACE 的“条件控制生成”不是 ControlNet 那种单独拷一套完整 U-Net 分支，也不是简单把控制图直接拼到 noisy latent 上。

它的实现更像是:

1. 先把各种条件统一预处理成 `src_video/src_mask/src_ref_images`
2. 再把这些输入编码成一个统一的 `vace_context`
3. 用一条额外的 `vace_blocks` 分支从 `vace_context` 提取多层 hint
4. 再把这些 hint 按层加回主干 Wan DiT block，影响每一层的去噪

代码上最关键的就是两段:

- 条件分支提特征: [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:149)
- 主干按层注入 hint: [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:259)


## 3. 统一输入接口

### 3.1 论文/文档层面的统一表示

[UserGuide.md](/Users/longxiao/Documents/projects/video_gen/VACE/UserGuide.md:33) 定义了三类视觉条件:

- `src_video`: 控制视频、待编辑视频，或者带灰色占位的扩展视频
- `src_mask`: 白色区域表示需要生成，黑色区域表示保留
- `src_ref_images`: 参考图像

这一步非常重要，因为 VACE 不是为 depth、pose、inpainting、frameref 分别写不同模型头，而是把它们都塞进同一套表示里。

### 3.2 预处理入口

统一预处理入口是 [vace/vace_preproccess.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/vace_preproccess.py:125)。

它的流程是:

1. 根据 `task` 从配置表里找到对应 annotator
2. 读入视频/图像/框/mask 等原始输入
3. 调 annotator 生成标准化结果
4. 输出成 `src_video` / `src_mask` / `src_ref_images`

任务注册表在 [vace/configs/__init__.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/configs/__init__.py:45)。


## 4. 不同条件是怎么变成统一输入的

### 4.1 Depth / Pose / Scribble / Flow / Layout 这类 control

这类任务本质上都是“把原视频变成一种控制视频”，通常只输出 `src_video`，推理时 `src_mask` 会默认全 1，表示整段都由模型生成。

配置见 [vace/configs/video_preproccess.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/configs/video_preproccess.py:7)。

例子:

- depth: [vace/annotators/depth.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/annotators/depth.py:9)
- pose: [vace/annotators/pose.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/annotators/pose.py:35)
- layout bbox/track: [vace/annotators/layout.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/annotators/layout.py:10)

它们的共同点是:

- 输入原视频帧
- 输出一段“控制帧序列”
- 这段控制帧序列会作为 `src_video`

例如 pose 控制就是先用 DWPose 检测关键点，再画成 skeleton 图，返回整段 `ret_frames`，见 [vace/annotators/pose.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/annotators/pose.py:130)。

### 4.2 Inpainting / Outpainting 这类 masked editing

这类任务会同时产生:

- `src_video`: 被挖空或扩边后的输入视频
- `src_mask`: 要重绘的位置

关键实现见 [vace/annotators/inpainting.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/annotators/inpainting.py:135)。

以 inpainting 为例:

- `apply_plain_mask` / `apply_seg_mask` 负责把待编辑区域涂成 mask color
- 同时生成二值 mask
- 最终返回 `frames` 和 `masks`

也就是说，VACE 把“编辑任务”转成了“带已知区域和未知区域的视频补全”。

### 4.3 FrameRef / ClipRef 这类 reference-driven extension

这类任务的思路是:

- 已知参考帧位置放真实图像
- 其余帧放灰色占位
- 未知帧对应 mask 置白

关键实现见 [vace/annotators/frameref.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/annotators/frameref.py:70)。

比如 `firstframe`:

- 第 1 帧是真图
- 后面 `expand_num` 帧全是灰色视频帧
- 后面的 mask 全是白色

这就是“给首帧，生成后续视频”的实现方式。

### 4.4 Reference image

参考图像不混在 `src_video` 里预处理，而是作为单独的 `src_ref_images` 输入。

推理前在 [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:245) 里加载、缩放、居中到目标画布大小。


## 5. 推理阶段怎样把条件喂进模型

### 5.1 统一准备输入

入口在 [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:216) 的 `prepare_source`。

它做了几件事:

- 读取 `src_video/src_mask`
- 如果没有 `src_video`，就创建全零视频 + 全 1 mask，用于纯文本/纯参考图生成
- 如果只有 `src_video` 没有 mask，就自动补全 1 mask
- 把参考图 resize 到视频分辨率

这一步说明: VACE 在推理时总能把不同任务整理成统一的 `(video, mask, refs)` 三元组。

### 5.2 把视频和 mask 编码成 latent

关键代码在:

- 帧编码: [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:149)
- mask 编码: [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:179)
- 合并: [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:213)

`vace_encode_frames` 的关键逻辑:

- 如果没有 mask，直接 `vae.encode(frames)`
- 如果有 mask，就把输入拆成:
  - `inactive = i * (1 - m)`，即保留区域
  - `reactive = i * m`，即待生成区域
- 两者分别进 VAE，再在 channel 维拼接

这说明 VACE 显式区分了“已知内容”和“待重绘内容”。

`vace_encode_masks` 则把像素空间 mask 重新整理到 latent patch 对齐的时空尺度上，最后也变成可拼接的张量。

最终 `vace_latent` 直接把视频 latent 和 mask latent 拼在一起，形成 `vace_context`。


## 6. 真正的“条件控制注入”发生在哪里

### 6.1 模型结构

`VaceWanModel` 定义在 [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:64)。

它在原始 `WanModel` 基础上多了三样东西:

- 主干 block: `self.blocks`
- 条件分支 block: `self.vace_blocks`
- 条件输入 patch embedding: `self.vace_patch_embedding`

其中 `vace_layers` 默认是每隔一层注一次，也就是 `[0, 2, 4, ...]`，见 [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:88)。

### 6.2 条件分支生成多层 hint

`forward_vace` 在 [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:149)。

流程是:

1. `vace_context` 先过 `vace_patch_embedding`
2. 展平成 token 序列
3. 经过多层 `vace_blocks`
4. 每一层输出一个 `c_skip`
5. 把每层的 `c_skip` 存成 `hints`

这里的 `VaceWanAttentionBlock` 有两个关键投影:

- `before_proj`: 仅第 0 层使用，把主干的 `x` 混进条件 token，见 [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:25)
- `after_proj`: 每层输出零初始化的 residual hint，见 [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:29)

零初始化很像 ControlNet/Adapter 系方法常见的稳定训练做法: 一开始不破坏原基模行为，再逐渐学会条件控制。

### 6.3 主干按层注入 hint

主干注入逻辑在 [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:259) 之后。

每经过一个主干 `block`:

- 如果该层在 `vace_layers` 里
- 就取对应的 `hint`
- 然后在 `BaseWanAttentionBlock.forward` 里执行

`x = x + hint * context_scale`

对应代码在 [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:57)。

这就是 VACE 条件控制的本质:

- 条件分支抽特征
- 主干按层加 residual hint
- `context_scale` 控制条件强度


## 7. 采样时怎么用

生成入口在 [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:284)。

关键步骤:

1. 文本编码得到 `context` 和 `context_null`
2. 条件输入编码得到 `z = vace_context`
3. 每个扩散步都调用两次模型:
   - `noise_pred_cond = model(..., vace_context=z, context=prompt)`
   - `noise_pred_uncond = model(..., vace_context=z, context=negative_prompt)`
4. 用 CFG 组合两者

注意这里有个很关键的点:

- 条件分支 `vace_context=z` 在 cond / uncond 两路里都存在
- 变化的只是文本条件

所以 VACE 的“控制条件”并不是 classifier-free dropout 掉的对象，而是始终作为结构约束存在；CFG 主要调的是文本语义强度。


## 8. 组合任务是怎么实现的

VACE 论文强调可以把多任务组合起来。代码上，这不是“多个条件分支同时并联进模型”，而是先在预处理阶段把多个任务融合成一个统一输入，再走同一套推理。

核心在 [vace/annotators/composition.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/annotators/composition.py:5)。

`CompositionAnnotator` 把任务归成三类:

- `control`
- `extension`
- `repaint`

然后根据组合类型，把两组 `(frames, masks)` 融成一组新的 `(output_video, output_mask)`，见 [vace/annotators/composition.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/annotators/composition.py:24)。

例如:

- `AnimateAnything = pose control + reference images`
  - 见 [vace/annotators/composition.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/annotators/composition.py:76)
- `SwapAnything = inpainting + reference images`
  - 见 [vace/annotators/composition.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/annotators/composition.py:93)
- `ExpandAnything = outpainting + frameref + references`
  - 见 [vace/annotators/composition.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/annotators/composition.py:112)

所以它的“all-in-one”更多体现在:

- 输入接口统一
- 预处理组合统一
- 模型注入方式统一


## 9. 对论文方法和代码的一一对应

### 9.1 统一多任务输入

- 文档定义: [UserGuide.md](/Users/longxiao/Documents/projects/video_gen/VACE/UserGuide.md:33)
- 代码入口: [vace/vace_preproccess.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/vace_preproccess.py:125)
- 任务注册: [vace/configs/__init__.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/configs/__init__.py:45)

### 9.2 条件视频/掩码/参考图编码

- source 准备: [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:216)
- frame latent 编码: [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:149)
- mask latent 编码: [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:179)
- 合成 `vace_context`: [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:213)

### 9.3 条件分支提取 hint

- 条件 block: [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:10)
- 条件 patch embedding: [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:109)
- hint 生成: [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:149)

### 9.4 主干注入控制

- 主干 block 定义: [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:41)
- hint 注入公式: [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:57)
- 主干 forward 调用: [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:176)

### 9.5 采样阶段使用

- 生成主循环: [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:360)


## 10. 我对这套实现的理解

可以把 VACE-Wan 的控制机制理解成:

- 文本条件: 通过原始 Wan 的 cross-attention 生效
- 视觉条件: 通过额外 `vace_blocks` 产生多层 residual hint 生效
- 空间/时间编辑区域: 通过 `src_mask` 编进 `vace_context`
- 参考主体: 通过 `src_ref_images` 编成额外参考 latent，并与视频 latent 在时间维拼接

换句话说，VACE 真正统一的不是“所有任务共享同一个 preprocess”，而是:

- 所有任务都尽量转写成同一种时空条件张量
- 然后用同一种多层 residual 注入方式控制 Wan 主干


## 11. 最短总结

一句话总结:

VACE 的条件控制生成 = “把不同任务统一成 `src_video + src_mask + src_ref_images`，编码成 `vace_context`，再通过 `vace_blocks` 提取多层 hint，并以 `x = x + hint * context_scale` 的方式逐层注入 Wan 主干”。

如果你接下来想继续深挖，最值得顺着读的代码顺序是:

1. [UserGuide.md](/Users/longxiao/Documents/projects/video_gen/VACE/UserGuide.md:33)
2. [vace/vace_preproccess.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/vace_preproccess.py:125)
3. [vace/models/wan/wan_vace.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/wan_vace.py:149)
4. [vace/models/wan/modules/model.py](/Users/longxiao/Documents/projects/video_gen/VACE/vace/models/wan/modules/model.py:64)

