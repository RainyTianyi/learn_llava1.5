"""LLaVA CC3M 数据工具：数据准备、Dataset 与 Collator。

本模块把 ``准备CC3M数据集和数据加载器.ipynb`` 中的关键代码抽成可复用的
Python 模块（不含数据集下载），这样调整数据量时不需要重跑整个 notebook，
训练脚本 / notebook 也可以直接导入 Dataset 和 Collator。

包含三部分：

1. :func:`prepare_dataset`：从 ``chat.json`` 中随机抽样子集，划分为训练集和
   验证集，并把用到的图片按需从 ``images.zip`` 解压到 ``images/``。
2. :class:`LlavaPretrainDataset`：返回可直接送入 LLaVA 的单个样本。
3. :class:`TrainCollator`：批内左 ``<pad>`` 填充，prompt 部分的 label 置 -100。

典型用法::

    # 命令行准备数据（随时改数据量，不需要运行 notebook）
    python llava_data.py --subset-size 2000 --val-size 100

    # 在训练 notebook / 脚本中
    from transformers import LlavaProcessor
    from llava_data import LlavaPretrainDataset, TrainCollator, prepare_dataset

    processor = LlavaProcessor.from_pretrained(MODEL_DIR)
    splits = prepare_dataset(subset_size=2000, val_size=100)  # 数据已准备好时可省略
    train_dataset = LlavaPretrainDataset(splits.train_json, splits.image_dir, processor)
    batch = TrainCollator(processor.tokenizer.pad_token_id)([train_dataset[0]])

数据路径默认值与 notebook 保持一致，可通过环境变量覆盖：
``LLAVA_DATA_ROOT``（默认 ``/root/autodl-tmp/CC3M``）、
``LLAVA_MODEL_DIR``（默认本文件所在目录下的 ``show_model/model001``）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import LlavaProcessor

__all__ = [
    "SYSTEM_PROMPT",
    "IMAGE_TOKEN",
    "LABEL_PAD_TOKEN_ID",
    "PROJECT_ROOT",
    "MODEL_DIR",
    "DATA_ROOT",
    "CHAT_PATH",
    "IMAGE_ZIP",
    "IMAGE_DIR",
    "SUBSET_SIZE",
    "VAL_SIZE",
    "SEED",
    "DatasetSplits",
    "prepare_dataset",
    "extract_images",
    "build_datasets",
    "LlavaPretrainDataset",
    "TrainCollator",
]

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

# 训练时使用的 system prompt 和图像占位符
SYSTEM_PROMPT = "You are a helpful assistant."
IMAGE_TOKEN = "<image>"

# 交叉熵损失忽略的 label 值（用于屏蔽 prompt 部分和 padding）
LABEL_PAD_TOKEN_ID = -100

# 本文件所在目录，即仓库根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 搭建 notebook 保存的初始 LLaVA 模型目录
MODEL_DIR = Path(
    os.environ.get("LLAVA_MODEL_DIR") or (PROJECT_ROOT / "show_model" / "model001")
)

# 数据集根目录（下载后的 CC3M 数据），与 notebook 默认值一致
DATA_ROOT = Path(os.environ.get("LLAVA_DATA_ROOT") or "/root/autodl-tmp/CC3M")
CHAT_PATH = DATA_ROOT / "chat.json"
IMAGE_ZIP = DATA_ROOT / "images.zip"
IMAGE_DIR = DATA_ROOT / "images"

# 默认的子集规模
SUBSET_SIZE = 5000  # 训练样本数
VAL_SIZE = 100      # 验证样本数
SEED = 42           # 抽样随机种子

Records = List[Dict[str, Any]]


@dataclass
class DatasetSplits:
    """:func:`prepare_dataset` 的返回值，描述一次数据准备的结果。"""

    data_root: Path        # 数据集根目录
    image_dir: Path        # 解压后的图片目录
    train_json: Path       # 训练集 json
    val_json: Path         # 验证集 json
    train_records: Records  # 训练集记录（已剔除缺图样本）
    val_records: Records    # 验证集记录（已剔除缺图样本）


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------


def _normalize_record(record: Dict[str, Any]) -> None:
    """保证一条记录中 human 有且只有一个 ``<image>``，并清理 gpt 段的占位符。"""
    conversations = record["conversations"]
    human = conversations[0]["value"]
    if human.count(IMAGE_TOKEN) != 1:
        # 数量不对时，删除原有占位符并在开头补一个
        conversations[0]["value"] = (
            IMAGE_TOKEN + "\n" + human.replace(IMAGE_TOKEN, "").strip()
        )
    conversations[1]["value"] = conversations[1]["value"].replace(IMAGE_TOKEN, "").strip()


def _write_json(path: Path, records: Records) -> None:
    """以 UTF-8 写出 json，非 ASCII 字符原样输出。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def extract_images(
    names: Sequence[str],
    image_zip: Union[str, Path] = IMAGE_ZIP,
    image_dir: Union[str, Path] = IMAGE_DIR,
    verbose: bool = True,
) -> List[str]:
    """把 ``names`` 中的图片从 ``image_zip`` 按需解压到 ``image_dir``。

    已经存在于 ``image_dir`` 的图片会跳过，因此反复调整数据量时只会增量解压。
    zip 内是扁平的 ``GCC_train_*.jpg``，这里仍遍历所有条目建立文件名映射，
    避免依赖 zip 内部目录结构。

    Returns:
        zip 中找不到的文件名列表（这些图片无法解压）。
    """
    image_zip = Path(image_zip)
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    # 去重并过滤掉本地已存在的图片
    todo = [name for name in dict.fromkeys(names) if not (image_dir / name).exists()]
    if verbose:
        print("images to extract:", len(todo))
    if not todo:
        return []

    if not image_zip.exists():
        raise FileNotFoundError(
            f"{image_zip} 不存在，但有 {len(todo)} 张图片尚未解压；"
            "请先下载数据集，或确认 image_dir 指向正确。"
        )

    missing: List[str] = []
    with zipfile.ZipFile(image_zip) as z:
        # 文件名 -> zip 内路径
        available = {
            Path(info.filename).name: info.filename
            for info in z.infolist()
            if not info.is_dir()
        }
        missing = [name for name in todo if name not in available]
        if verbose and missing:
            print("missing in zip:", len(missing), missing[:3])

        for name in tqdm(todo, disable=not verbose, desc="extract images"):
            if name in available:
                (image_dir / name).write_bytes(z.read(available[name]))
    return missing


def prepare_dataset(
    subset_size: int = SUBSET_SIZE,
    val_size: int = VAL_SIZE,
    seed: int = SEED,
    data_root: Union[str, Path] = DATA_ROOT,
    verbose: bool = True,
) -> DatasetSplits:
    """抽取 CC3M 子集、划分训练/验证集并按需解压图片。

    与 notebook 中「划分和解压」部分的行为一致：固定 ``seed`` 时，同样的
    ``subset_size`` / ``val_size`` 会得到同样的样本。修改规模后重新调用即可，
    已解压的图片不会重复解压。

    Args:
        subset_size: 训练集样本数。
        val_size: 验证集样本数，可为 0。
        seed: 随机抽样种子。
        data_root: 数据集根目录，需包含 ``chat.json`` 和 ``images.zip``。
        verbose: 是否打印进度和统计信息。

    Returns:
        :class:`DatasetSplits`，包含划分结果、json 路径和图片目录。

    Raises:
        FileNotFoundError: ``chat.json`` 或缺失图片所需的 ``images.zip`` 不存在。
        ValueError: 样本数不合法，或请求的样本数超过数据总量。
    """
    data_root = Path(data_root)
    chat_path = data_root / "chat.json"
    image_zip = data_root / "images.zip"
    image_dir = data_root / "images"

    if subset_size <= 0:
        raise ValueError(f"subset_size 必须为正整数，当前为 {subset_size}")
    if val_size < 0:
        raise ValueError(f"val_size 不能为负数，当前为 {val_size}")
    if not chat_path.exists():
        raise FileNotFoundError(f"{chat_path} 不存在，请先下载数据集")

    image_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取全部记录的文本信息（此时还不加载图片）
    with open(chat_path, encoding="utf-8") as f:
        all_records: Records = json.load(f)
    total = len(all_records)
    if verbose:
        print("data root:", data_root)
        print("total records:", total)

    need = subset_size + val_size
    if need > total:
        raise ValueError(f"请求 {need} 条样本，但 chat.json 只有 {total} 条")

    # 2. 随机抽样并取子集
    rng = random.Random(seed)
    indices = rng.sample(range(total), need)
    subset = [all_records[i] for i in indices]
    del all_records

    # 3. 规范化每条样本的对话文本
    for record in subset:
        _normalize_record(record)

    train_records = subset[:subset_size]
    val_records = subset[subset_size:]
    if verbose:
        print("train:", len(train_records), "val:", len(val_records))

    # 4. 按需解压选中样本的图片
    extract_images(
        [record["image"] for record in subset],
        image_zip=image_zip,
        image_dir=image_dir,
        verbose=verbose,
    )

    # 5. 剔除确实缺图的样本，并把最终划分写入磁盘
    missing_images = [
        record["image"]
        for record in subset
        if not (image_dir / record["image"]).exists()
    ]
    if verbose:
        print("missing images:", len(missing_images), missing_images[:5])
    if missing_images:
        missing_set = set(missing_images)
        train_records = [
            record for record in train_records if record["image"] not in missing_set
        ]
        val_records = [
            record for record in val_records if record["image"] not in missing_set
        ]

    train_json = data_root / "chat_train.json"
    val_json = data_root / "chat_val.json"
    _write_json(train_json, train_records)
    _write_json(val_json, val_records)

    if verbose:
        print("usable train:", len(train_records), "val:", len(val_records))
        print("extracted images:", len(list(image_dir.glob("*.jpg"))))

    return DatasetSplits(
        data_root=data_root,
        image_dir=image_dir,
        train_json=train_json,
        val_json=val_json,
        train_records=train_records,
        val_records=val_records,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LlavaPretrainDataset(Dataset):
    """LLaVA 第一阶段（图文对）训练用的 Dataset。

    每个样本返回四个张量：

    - ``input_ids``：格式化 prompt（含 ``<image>``）+ answer + ``<eos>``；
    - ``labels``：prompt 部分为 -100，只对 answer + ``<eos>`` 计算损失；
    - ``attention_mask``：全 1（padding 由 Collator 负责）；
    - ``pixel_values``：CLIP 图像处理后的张量，形状 ``(3, 336, 336)``。
    """

    def __init__(
        self,
        records_path: Union[str, Path],
        image_dir: Union[str, Path],
        processor: LlavaProcessor,
    ):
        """
        Args:
            records_path: 数据集 json 文件路径（``chat_train.json`` 或 ``chat_val.json``）。
            image_dir: 解压后的图片所在目录。
            processor: 与模型配套的 ``LlavaProcessor``。
        """
        with open(records_path, encoding="utf-8") as f:
            self.records: Records = json.load(f)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_token_id = self.tokenizer.encode(IMAGE_TOKEN)[0]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        human = record["conversations"][0]["value"]
        # 防御性处理：避免 human 中 <image> 不唯一、answer 中混入 <image>
        if human.count(IMAGE_TOKEN) != 1:
            human = IMAGE_TOKEN + "\n" + human.replace(IMAGE_TOKEN, "").strip()
        answer = record["conversations"][1]["value"].replace(IMAGE_TOKEN, "").strip()
        image = Image.open(self.image_dir / record["image"]).convert("RGB")

        # 用 tokenizer 的 chat template 格式化 user 指令
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": human},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # processor 负责 tokenize 文本并处理图片
        q_inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        q_ids = q_inputs["input_ids"]                # (1, Lq)
        pixel_values = q_inputs["pixel_values"][0]   # (3, 336, 336)

        # answer 词元化后接 <eos>，拼到 prompt 之后
        a_ids = self.tokenizer(
            answer, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]                               # (1, La)
        eos = torch.tensor(
            [[self.tokenizer.eos_token_id]], dtype=a_ids.dtype
        )                                            # (1, 1)

        input_ids = torch.cat([q_ids, a_ids, eos], dim=1)[0]
        # prompt 部分不参与损失计算
        labels = torch.cat(
            [torch.full_like(q_ids, LABEL_PAD_TOKEN_ID), a_ids, eos], dim=1
        )[0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "attention_mask": torch.ones_like(input_ids),
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


@dataclass
class TrainCollator:
    """把一个 batch 的样本左 ``<pad>`` 填充到相同长度。

    左填充使最后一个 token 始终是真实文本的末尾，便于和推理/生成链路保持一致。
    ``input_ids`` 补 ``pad_token_id``，``labels`` 补 -100（而不是 pad id），
    ``attention_mask`` 补 0。
    """

    pad_token_id: int
    label_pad_token_id: int = LABEL_PAD_TOKEN_ID

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Dataset 通过方括号访问取出的样本列表。
        """
        max_len = max(feature["input_ids"].size(0) for feature in features)

        input_ids, labels, attention_mask = [], [], []
        for feature in features:
            length = feature["input_ids"].size(0)
            pad_len = max_len - length

            input_ids.append(
                torch.cat([
                    torch.full(
                        (pad_len,), self.pad_token_id, dtype=feature["input_ids"].dtype
                    ),
                    feature["input_ids"],
                ])
            )
            labels.append(
                torch.cat([
                    torch.full(
                        (pad_len,),
                        self.label_pad_token_id,
                        dtype=feature["labels"].dtype,
                    ),
                    feature["labels"],
                ])
            )
            attention_mask.append(
                torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    feature["attention_mask"],
                ])
            )

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
            "pixel_values": torch.stack(
                [feature["pixel_values"] for feature in features]
            ),
        }


# ---------------------------------------------------------------------------
# 便捷构造与命令行入口
# ---------------------------------------------------------------------------


def build_datasets(
    processor: LlavaProcessor,
    splits: Optional[DatasetSplits] = None,
    data_root: Union[str, Path] = DATA_ROOT,
) -> Tuple[LlavaPretrainDataset, LlavaPretrainDataset]:
    """构造 ``(train_dataset, val_dataset)``。

    Args:
        processor: 与模型配套的 ``LlavaProcessor``。
        splits: :func:`prepare_dataset` 的返回值；为 ``None`` 时按 ``data_root``
            下已有的 ``chat_train.json`` / ``chat_val.json`` 加载。
        data_root: 未传 ``splits`` 时使用的数据集根目录。
    """
    if splits is not None:
        train_json, val_json = splits.train_json, splits.val_json
        image_dir = splits.image_dir
    else:
        root = Path(data_root)
        train_json = root / "chat_train.json"
        val_json = root / "chat_val.json"
        image_dir = root / "images"

    for path in (train_json, val_json):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} 不存在，请先运行 prepare_dataset() 或 python llava_data.py"
            )

    return (
        LlavaPretrainDataset(train_json, image_dir, processor),
        LlavaPretrainDataset(val_json, image_dir, processor),
    )


def main(argv: Optional[Sequence[str]] = None) -> DatasetSplits:
    """命令行入口：准备训练/验证子集。"""
    parser = argparse.ArgumentParser(
        description="准备 LLaVA CC3M 训练/验证子集（不包含数据集下载）"
    )
    parser.add_argument(
        "--data-root", default=str(DATA_ROOT), help="数据集根目录（含 chat.json 和 images.zip）"
    )
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE, help="训练样本数")
    parser.add_argument("--val-size", type=int, default=VAL_SIZE, help="验证样本数")
    parser.add_argument("--seed", type=int, default=SEED, help="抽样随机种子")
    parser.add_argument("--quiet", action="store_true", help="不打印进度")
    args = parser.parse_args(argv)

    splits = prepare_dataset(
        subset_size=args.subset_size,
        val_size=args.val_size,
        seed=args.seed,
        data_root=args.data_root,
        verbose=not args.quiet,
    )
    print(f"train json: {splits.train_json} ({len(splits.train_records)})")
    print(f"val json:   {splits.val_json} ({len(splits.val_records)})")
    print(f"image dir:  {splits.image_dir}")
    return splits


if __name__ == "__main__":
    main()
