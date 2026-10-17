# LLaVA：CLIP 和 Llama 基础

LLaVA 受 InstructGPT、LLaMA 等工作中“用指令微调预训练语言模型”的成功启发，探索用**视觉指令微调大型多模态模型**，取得了很好的效果。模型架构上，LLaVA 追求对已有模型的最简单修改，由两个预训练组件构成：**CLIP 预训练的 ViT 图像编码器**和 **LLaMA/Vicuna 等预训练大语言模型（LLM）**。

LLaVA 的核心思路是使用一个投影模块——LLaVA-1.0 中是一个全连接层，LLaVA-1.5 中改为两层 MLP——将 **ViT 输出的 patch/grid 视觉特征**投影到 LLM 的词嵌入空间，作为 vision tokens 与文本 tokens 拼接。这一模块可简称为**对齐单元**。训练分为两个阶段：

- **特征对齐预训练**：冻结 ViT 和 LLM 权重，使用**图像-文本对数据集**训练投影/对齐单元。  
- **端到端微调**：冻结 ViT 权重，使用**视觉指令数据集**同时调整对齐单元和 LLM 的权重。

LLaVA-1.5 及后续变体在部分多模态基准上表现亮眼，展示了“简单投影 + 视觉指令微调”将 LLM 扩展到多模态的有效思路；这一路线也被后续 LMM（Large Multimodality Models）研究广泛沿用。注意：LLaVA 使用的是 CLIP 的 **ViT 视觉骨干**，不使用 CLIP 的文本编码器，也不继承 CLIP 的对比预训练目标。

下图展示 Llava-1.5 的架构，考虑到训练成本，我们将以小规模数据去复现一个未完全训练的模型。

![Llava-1.5 Arch](<img/Llava-1.5 Architechture.png>)

为了理解 LLaVA 架构，下面简单介绍 CLIP 和 LLaMA/LLM 的相关信息。如果已经理解 LLaVA 架构，下面的部分可以不看。

## CLIP 双塔模型

[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

CLIP 是规模化图文对比预训练的代表性工作之一。它把图像和文本映射到同一个嵌入空间，使自然语言可以作为图像分类、检索和匹配的开放词汇接口。**CLIP 的核心是图文对齐和判别式匹配，不是生成式多模态对话模型**，没有 LLM 的指令跟随和文本生成接口。

普通 ViT 预训练后通常接分类头做固定类别分类；CLIP 则用文本编码器把类名/描述编码成零样本分类器，通过图像嵌入与文本嵌入的余弦相似度在候选类上做 softmax。例如把候选类名填入 `A photo of a {label}.`，文本编码器输出文本嵌入，再与图像嵌入匹配，即可实现零样本分类。

CLIP 的核心工作是将图片和文本拉到同一个语义空间，而**没有实现自然语言指令功能**。而后续的很多研究也只是通过架构上或数据上的微调，专精一个方面的自然语言指令，如描述图片场景、回答图片中的问题、判断图片是否有不合理的地方等等的具体问题，没有出现一个自然语言指令引导视觉的通用模型。直到 GPT-4 和 Llava，初步实现了**自然语言指令引导的通用视觉模型**。

### Vision Transformer

ViT 是对于图片的 Transformer Encoder。这是首个证明了 大规模预训练 + Transformer 架构 在计算机视觉领域同样有效的模型，使用一个很简单的思路：**将图片切成大小相等的 patches，经过线性投影得到长度相等的向量，直接 stack 形成输入序列**。同时在最前方加一个 cls token 抽取图片信息，用于适应分类的下游任务。

![ViT Arch](<img/ViT Architecture.png>)

### CLIP 模型架构

下图展示的是 CLIP 模型架构，其中 Text Encoder 和 Image Encoder 都是用 Transformer 的变种实现的，我们称为**双塔模型**。

![CLIP Arch](<img/CLIP Architechture.png>)

#### Text Encoder

使用 GPT-2 改动的 Transformer Decoder。GPT-2 做的主要改动为：
- 使用可学习位置编码，代替正余弦绝对位置编码。
- 删除 encoder-decoder 交叉注意力，一个 Transformer 块中只用因果掩码多头自注意力。
- 增加末尾 LayerNorm。
- 所有归一化层改为 pre-norm，比 post-norm 更加稳定。
- 使用 GELU 激活和 Dropout。

#### Image Encoder

ViT 版本基本沿用 ViT 的 Transformer Encoder，主要改动是：在 patch 嵌入与位置嵌入相加后、进入 Transformer 前额外加一层 LayerNorm，并采用略有不同的初始化。CLIP 也提供 ResNet 图像编码器，包括 RN50、RN101、RN50x4、RN50x16、RN50x64；ResNet 版本使用 ResNet-D 改进、antialiased blur pooling，并用 attention pooling 替代 global average pooling。

### CLIP 预训练

CLIP 的预训练任务是**图文配对判别**的对比学习目标：给定一个 batch 中 N 对（图像，文本），模型要判断哪些图像和哪些文本是真实配对的。

具体流程如下：

1. **数据**：CLIP 构建了 WIT（WebImageText）数据集，约 **4 亿对（图像，文本）**，来自互联网公开来源。  
2. **双塔编码**：图像编码器输出图像特征，文本编码器输出文本特征，两者分别经线性投影映射到共享多模态嵌入空间，并做 L2 归一化。
3. **对比目标**：对一个 batch 的 N 对图文，计算 N×N 的**图像-文本余弦相似度矩阵**，损失是对相似度矩阵**按行和按列分别做交叉熵**后的对称交叉熵：  
   - 图像→文本分类：每张图在 N 个文本中找正确文本；  
   - 文本→图像分类：每段文本在 N 张图中找正确图像；  
   - 两者取平均。  
4. **零样本迁移**：测试时，把目标数据集的类名或描述填入 prompt 模板（如 `A photo of a {label}.`），图像嵌入与所有类文本嵌入计算余弦相似度并 softmax，取最高分。

通过这种图文匹配的预训练，CLIP 摆脱了 categorical label 的限制。可以接受任意的文本输入和任意的图片输入，也达到了很好的 zero-shot 性能。给任何一张照片，都能通过一个句子（自然文本），判断照片中有没有我想要的东西。

总的来说，CLIP 预训练提供的是：**一个可迁移的视觉骨干 + 一个图文共享嵌入空间 + 用自然语言定义分类器的零样本能力**。

### CLIP v.s. LLaVA 

**CLIP 预训练是“双塔图文对比对齐”，LLaVA 是“借用 CLIP 的 ViT 视觉骨干，把视觉 token 投影进 LLM，再用视觉指令微调成多模态助手”**。

| 维度 | CLIP 预训练 | LLaVA |
| --- | --- | --- |
| 模型结构 | 双塔：图像编码器 + 文本编码器 | CLIP ViT 图像编码器 + 投影层 + LLM |
| 预训练目标 | batch 内图文对比/配对判别 | 第一阶段：冻结 ViT 和 LLM，只训练投影层做图文特征对齐；第二阶段：视觉指令微调，训练投影层和 LLM |
| 输出 | 图像/文本嵌入、相似度、零样本分类 | 自然语言回答 |
| 数据 | 4 亿网络图文对 | 图文对 + 视觉指令数据 |
| 能力 | 图文匹配、检索、零样本分类 | 指令跟随、视觉问答、多模态对话 |


## Llama 

Llava 使用的是 Vicuna 模型，Vicuna 本身**是基于 LLaMA 微调出来的对话/指令模型**。Vicuna 对 Llama 架构本身没有做修改，只是**使用 ShareGPT 数据集微调**：ShareGPT 的数据是用户自愿分享的真实 ChatGPT 多轮对话记录，这使得 Vicuna 能够直接学习到高质量、多样化的对话模式，而非仅仅学习语言的一般规律。

Llava-1.5 使用 Vicuna 的后续版本 v1.5，将基座模型从 Llama 切换至 Llama 2，并进一步将上下文长度支持扩展到了 4K 和 16K，以适应更长的对话场景。这里直接介绍 Llama 2 模型架构。

### Llama2 模型架构

Llama2 整体仍是 decoder-only Transformer，在具体的层中的算法细节中有做改动。我们基于 **GPT-2 对 Transformer Decoder 架构的主要修改：使用可学习位置编码、删除 encoder-decoder 交叉注意力、增加末尾 LayerNorm、所有归一化层改为 pre-norm、使用 GELU 激活和 Dropout**，简要讲解 Llama2 架构。

Llama2 架构可以概括为：

\[
x \leftarrow x+\mathrm{Attn}(\mathrm{RMSNorm}(x))
\]

\[
x \leftarrow x+\mathrm{FFN}(\mathrm{RMSNorm}(x))
\]

\[
\mathrm{logits}=W_{\text{out}}\,\mathrm{RMSNorm}(x)
\]


下图直观展示了 GPT-2、Llama-2、Llama-3 的架构与差异。图源：`github rasbt/LLMs-from-scratch/ch05/07_gpt_to_llama`

![gpt2-to-llama2-llama3](img/gpt2-to-llama2-llama3.webp)

Llama2-7B 的典型配置是：词表 32k、上下文 4096、隐藏维度 4096、32 层、32 个注意力头、FFN 中间维度 11008，训练/推理常用 bf16。与 GPT-2 相比，Llama2 的主要结构改动集中在**归一化、激活/FFN、位置编码和偏置项**上。

### RMSNorm

Llama2 用 RMSNorm 替换 LayerNorm。LayerNorm 同时做“减均值”和“除以标准差”，而 RMSNorm 只做均方根缩放，不减均值、通常也不加偏置：

\[
\mathrm{RMSNorm}(x)=\gamma\odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}
\]

其中 \(\gamma\) 是可学习的缩放向量。这样计算更省，参数更少，也更容易和残差相加融合。在 pre-norm 结构中，残差路径本身较简单，去掉 re-centering 通常不影响稳定性。

### SwiGLU FFN

Llama2 把 GPT-2 的 GELU MLP 换成 SwiGLU 门控 FFN。SwiGLU 使用 SiLU 作为门控激活：

\[
\mathrm{SiLU}(x)=x\sigma(x)
\]

\[
\mathrm{FFN}(x)=\big(\mathrm{SiLU}(xW_g)\odot xW_u\big)W_d
\]

其中 \(W_g\) 是 gate 投影，\(W_u\) 是 up 投影，\(W_d\) 是 down 投影。相比传统“升维→激活→降维”的 MLP，SwiGLU 多了一个门控分支，因此中间维度通常不取 \(4d\)，而取约 \(\frac{8}{3}d\)，再对齐到硬件友好的数值，Llama2-7B 取 11008。

### RoPE 位置编码

Llama2 不再使用可学习绝对位置嵌入，而是**把位置信息通过旋转注入到注意力层**的 Query/Key 中。把 head 维度分成 \(d/2\) 个二维子空间，每个子空间用不同频率旋转：

\[
\theta_i=\text{base}^{-2i/d},\qquad i=0,\dots,d/2-1
\]

\[
R_{m,i}=
\begin{pmatrix}
\cos m\theta_i & -\sin m\theta_i\\
\sin m\theta_i & \cos m\theta_i
\end{pmatrix}
\]

对位置 \(m\) 的 Query 和位置 \(n\) 的 Key：

\[
q_m=R_mW_qx_m,\qquad k_n=R_nW_kx_n
\]

由于 \(R_m^TR_n=R_{n-m}\)，注意力分数满足：

\[
q_m^Tk_n=(W_qx_m)^TR_{n-m}(W_kx_n)
\]

也就是说，**内积只依赖相对位置 \(n-m\)**。RoPE 不是“加在 embedding 上”的位置编码，而是放在注意力层里对 Q/K 做旋转。Llama2 的 `theta_base` 默认是 10000，Llama3 增大到 500000 以支持更长上下文。

### 注意力与 KV Cache

Llama2 的注意力仍是缩放点积因果自注意力：

\[
\mathrm{Attn}(X)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
\]

其中 \(M\) 是因果 mask。Llama2 的线性层去掉 bias，Q/K/V/O 投影都不带偏置。Llama2-7B/13B 使用传统多头注意力（MHA），34B/70B 使用 GQA。GQA 的详细内容放在 Llama3 讲，本仓库不介绍。

推理时使用 KV Cache：每生成一个新 token，只计算当前 token 的 \(q_t,k_t,v_t\)，把 \(k_t,v_t\) 追加到历史缓存，再用 \(q_t\) 与整个 \(K_{\text{cache}}\) 做注意力，避免重复计算历史 K/V。

### GPT-2 v.s. Llama2 

| 维度 | GPT-2 | Llama2 |
|---|---|---|
| 归一化 | LayerNorm | RMSNorm |
| FFN | GELU MLP | SwiGLU 门控 FFN |
| 位置编码 | 可学习绝对位置嵌入 | RoPE，旋转注入 Q/K |
| 注意力 | MHA | 7B/13B MHA；34B/70B GQA |
| 上下文 | 1024 | 4096 |

## Vicuna 与 LLaVA 所用 LLM

LLaVA-1.5 使用的 LLM 是 Vicuna v1.5，而 **Vicuna 本身是在 LLaMA/LLaMA2 基础上用 ShareGPT 对话数据做指令微调得到的对话模型**。Vicuna 基本不改变 Llama 架构，主要改变的是指令/对话数据分布，因此 Llama2 的结构结论可以直接迁移到 Vicuna。LLaVA 论文的消融显示：Vicuna-v1.5 整体最好，LLaMA2-based 模型一般优于 LLaMA1-based。**语言指令微调会影响多语言和特定任务表现。**

## LLaVA 架构与训练细节

LLaVA 的架构可以概括为：

\[
\text{image}\rightarrow \text{CLIP ViT}\rightarrow \text{projection}\rightarrow \text{visual tokens}
\]

\[
\text{visual tokens}\oplus \text{text tokens}\rightarrow \text{LLM}\rightarrow \text{response}
\]

LLaVA-1.5 由三部分组成：**CLIP ViT-L/14@336px 视觉编码器**、**两层 MLP 投影层**、**Vicuna v1.5 13B LLM**。LLaVA-1.0 使用单线性投影，LLaVA-1.5 改为两层 MLP。LLaVA 不把视觉特征做成 CLIP 那样的图文相似度匹配，而是把**视觉 token 投影到 LLM 词嵌入空间，作为前缀与文本 token 拼接，由 LLM 自回归生成回答**。

### 视觉特征与投影层

图像经 CLIP ViT 切 patch 后得到 grid patch 特征。**LLaVA 使用 patch 特征，同时不使用 CLS token**。LLaVA-1.5 使用 336×336 输入、patch 14，因此 grid 为 \(24\times24\)，得到 576 个 visual tokens：

\[
(336/14)^2=576
\]

投影层把 ViT 的 hidden size 映射到 LLM 的 hidden size。LLaVA-1.0 是单线性层；LLaVA-1.5 是两层 MLP，中间带非线性。**投影层的作用是把视觉特征对齐到 LLM 的词嵌入语义空间，使 LLM 能把 visual tokens 当作一种“视觉词”处理**。

LLaVA-1.5-HD 进一步探索高分辨率：**把高分辨率图像按 224×224 网格切分，每个网格独立送入视觉编码器，得到各自特征图后合并成一张大特征图；同时把一张下采样后的整图特征拼接到合并特征上，提供全局上下文**。后处理包括去掉 padding 对应特征、在每行末尾加特殊 token 以显式标记图像形状、展平后送入 LLM。这样可以把输入扩展到任意分辨率，论文中预定义支持最多 6 个网格，最大约 672×448 或 448×672。全局上下文能缓解 **低分辨率模型却对应需求图像细节指令** 造成的模型幻觉，并帮助模型定位高分辨率特征中的相关区域。

### Visual token 拼接与对话模板

LLaVA 把 visual tokens 放在文本 token 序列前，作为图像前缀；文本侧使用 Vicuna 的对话模板，包含 system/user/assistant 角色标记。原始 LLaVA 用 `<image>` 占位符表示图像位置，训练/推理时把占位符替换为投影后的视觉特征。**对话模板**通常形如：

- system：系统提示；
- user：用户问题；
- assistant：模型回答。

训练时通常只对 assistant 回复 token 计算自回归损失，system/user token 不计损失，使模型学习“给定图像和用户指令，生成回答”。推理时从 assistant 起始位置开始自回归生成。

### 两阶段训练

LLaVA 的训练分为两个阶段。

**阶段一：特征对齐预训练。** 冻结 CLIP ViT 和 LLM，只训练投影层。LLaVA-1.5 使用 558K 图文对（公开数据子集），目标是让视觉 token 进入 LLM 词嵌入空间。

**阶段二：视觉指令微调。** 冻结 ViT，训练投影层和 LLM。LLaVA-1.5 使用 665K 视觉指令数据，混合了 LLaVA-Instruct 158K、ShareGPT 40K、VQAv2 83K 等。相比 LLaVA-1.0，LLaVA-1.5 加入了 academic-task-oriented VQA 等数据，并用 **response formatting prompts** 统一输出格式，例如：

- “Answer the question using a single word or phrase.”
- “Answer with the option's letter from the given choices directly.”


这些 prompt 解决了一个关键问题：如果 VQA 数据中既有长回答又有短回答，而 prompt 只写 `Q: ... A:`，LLM 容易行为上过拟合到短答案；明确指定输出格式后，**LLaVA 能在短答案和自然对话之间取得平衡**，便于扩展数据源。

### 推理流程

推理时：
- 图像→CLIP ViT→投影层→visual tokens
- 文本按对话模板编码为 token
- visual tokens 与文本 token 拼接后送入 LLM
- LLM 自回归生成回答。
