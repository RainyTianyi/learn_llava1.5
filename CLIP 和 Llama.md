# CLIP 和 Llama 基础

Llava 模型受 Instruct-GPT 和 Llama 使用指令微调预训练模型在 NLP 领域的成功带来的启发，探索了用**视觉指令微调视觉大语言模型**，取得了优秀的成果。在模型架构上，Llava 追求对已有模型的最简单修改，整个模型由两个大模型基座构成：CLIP 中训练的 ViT 模型 和 Llama 中训练的 大语言模型（LLM）。

Llava 的核心思路是使用**一个全连接层**（在 Llava-1.5 中更改为一个两层的 MLP），**将 ViT 模型输出的所有 patches 的向量做投影，实现与 Llama 模型的词嵌入向量的语义空间对齐**。这一模块的作用类似于一个 vision token to text token 的嵌入层，我们简称为*对齐单元*。模型的训练过程被分为两个阶段：

- **特征对齐预训练**：冻结 ViT 和 LLM 权重，使用 图像-文本对数据集 训练 ViT 模型后的对齐单元。
- **端到端微调**：冻结 ViT 权重，使用 视觉指令数据集 同时调整对齐单元和 LLM 的权重。

Llava-1.5-HD 开源模型达到了接近 GPT-4 多模态大模型的能力，给出了一种简单而有效的将 LLM 的能力扩展到多模态的思路，这种思路也被后续 LMM（Large Multimodality Models） 领域的一些研究所沿用。

为了理解 Llava 架构，这里简单介绍 CLIP 和 Llama 模型的相关信息。

## CLIP 双塔模型

[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

CLIP 模型首次打通了语言和视觉两大领域，实现了能够使用自然语言与图片信息与大模型进行交互。如传统 ViT 仍只能处理分类问题，需要分类头的存在。而 CLIP 可以通过提示词如 `a photo of a {object}` 实现和 ViT 一致的分类效果。

这里需要注意，CLIP 的核心工作是**将图片和文本拉到同一个语义空间中，并没有实现自然语言指令功能**。而后续的很多研究也只是通过架构上或数据上的微调，专精一个方面的自然语言指令，如描述图片场景、回答图片中的问题、判断图片是否有不合理的地方等等的具体问题，没有出现一个自然语言指令引导视觉的通用模型。直到 GPT-4 和 Llava，成功将自然语言指令引入视觉模型中。

### Vision Transformer

ViT 是对于图片的 Transformer Encoder。这是首个将 Transformer 架构引入计算机视觉领域的模型，使用一个很简单的思路：**将图片切成大小相等的 patches，经过线性投影得到长度相等的向量，直接 stack 形成输入序列**。同时在最前方加一个 cls token 抽取图片信息，用于适应分类的下游任务。

![ViT Arch](<img/ViT Architecture.png>)

### CLIP 模型架构



## Llama 

Llava 使用的是 Vicuna 模型，Vicuna 本身**是基于 LLaMA 微调出来的对话/指令模型**。Vicuna 对 Llama 架构本身没有做修改，只是**使用 ShareGPT 数据集微调**：ShareGPT 的数据是用户自愿分享的真实 ChatGPT 多轮对话记录，这使得 Vicuna 能够直接学习到高质量、多样化的对话模式，而非仅仅学习语言的一般规律。

Llava-1.5 使用 Vicuna 的后续版本 v1.5，将基座模型从 Llama 切换至 Llama 2，并进一步将上下文长度支持扩展到了 4K 和 16K，以适应更长的对话场景。这里直接介绍 Llama 2 模型架构。

### Llama2 模型架构

Llama2 整体架构上依然是 Transformer Decoder，在具体的层中的算法细节中有做改动。我们基于 **GPT-2 对 Transformer Decoder 架构的主要修改：使用可学习位置编码、删除 encoder-decoder 交叉注意力、增加末尾 LayerNorm、所有归一化层改为 pre-norm、使用 GELU 激活和 Dropout**，简要讲解 Llama2 架构。

