#!/usr/bin/env python3
"""
生成 5 个测试用例所需的 GSM8K 格式数据集。

用例1: 128n / 128n±1 / 128n±2 三种长度混合（n 为 128 的整数倍），block-size 边界
用例2: max_model_len / max_model_len±3 混合，最大长度边界
用例3: 30k(50%) / 64k(40%) / 90k(10%) 加权混合，业务流量模拟
用例4: 32k 定长输入（长输出场景由测试脚本指定 output_len=16k）
用例5: 2k(90%) / 128k(10%) 长尾分布
"""

import argparse
import json
import logging
import os
import random
import sys

from generate_dataset import (
    _truncate_or_pad_text,
    create_dataset,
    create_prefix_dataset_bucketed,
    write_data,
)
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _resolve_tokenizer_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    # HuggingFace 模型 ID（如 zai-org/GLM-5）
    if "/" in model_path and not os.path.isabs(model_path):
        return model_path
    try:
        from config import TOKENIZER_FALLBACK
        fallback = TOKENIZER_FALLBACK
    except ImportError:
        fallback = "zai-org/GLM-5"
    logging.warning("MODEL_PATH %s 不存在，回退 tokenizer: %s", model_path, fallback)
    return fallback


def _load_tokenizer(model_path: str):
    path = _resolve_tokenizer_path(model_path)
    logging.info("加载 tokenizer: %s", path)
    kwargs = {}
    try:
        from config import TRUST_REMOTE_CODE
        if TRUST_REMOTE_CODE:
            kwargs["trust_remote_code"] = True
    except ImportError:
        pass
    return AutoTokenizer.from_pretrained(path, **kwargs)


def _verify_lengths(tokenizer, texts, target_lens):
    """抽样校验 token 长度是否符合预期"""
    mismatches = 0
    for i, (text, target) in enumerate(zip(texts, target_lens)):
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        if actual != target:
            mismatches += 1
            if mismatches <= 5:
                logging.warning("样本 %d: 目标长度 %d, 实际长度 %d", i, target, actual)
    if mismatches:
        logging.warning("共 %d/%d 条样本长度与目标不完全一致（tokenizer 编解码舍入导致，已尽力校正）", mismatches, len(texts))
    else:
        logging.info("长度校验通过 (%d 条)", len(texts))


def _build_texts_at_lengths(model_path: str, target_lens: list) -> list:
    """按目标 token 长度列表生成文本"""
    if not target_lens:
        return []

    tokenizer_path = _resolve_tokenizer_path(model_path)
    max_len = max(target_lens)
    pool_size = len(set(target_lens))
    # 预生成足够的基础文本池
    base_pool = create_dataset(tokenizer_path, max_len, max(pool_size, len(target_lens)), 0)
    if base_pool is None:
        logging.error("生成基础文本池失败，请清空 picked_ids.txt 后重试")
        sys.exit(1)

    tokenizer = _load_tokenizer(model_path)
    texts = []
    for idx, tl in enumerate(target_lens):
        src = base_pool[idx % len(base_pool)]
        texts.append(_truncate_or_pad_text(tokenizer, src, tl))
    return texts


def _save_dataset(save_dir: str, name: str, texts: list, meta: dict):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    write_data(path, texts, None)
    meta_path = path.replace(".jsonl", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logging.info("[完成] %s (%d 条) -> %s", name, len(texts), path)
    return path


def _repeat_to_count(items: list, count: int) -> list:
    if not items:
        return []
    repeats = count // len(items)
    remainder = count % len(items)
    result = items * repeats + items[:remainder]
    random.shuffle(result)
    return result


def generate_case1_block_boundary(tokenizer_path: str, save_dir: str, data_num: int, seed: int) -> str:
    """
    用例1: 128n / 128n±1 / 128n±2 混合，n 为 128 的整数倍。
    在 n ∈ {128, 256, 384, 512} 上均匀轮换，三类长度各占 1/3。
    """
    random.seed(seed)
    n_multiples = [128, 256, 384, 512]

    categories = [
        lambda base: base,           # 128n
        lambda base: base + random.choice([-1, 1]),   # 128n±1
        lambda base: base + random.choice([-2, 2]),   # 128n±2
    ]

    target_lens = []
    per_cat = data_num // 3
    for cat_idx, cat_fn in enumerate(categories):
        count = per_cat if cat_idx < 2 else (data_num - 2 * per_cat)
        for i in range(count):
            n = n_multiples[i % len(n_multiples)]
            base = 128 * n
            target_lens.append(max(1, cat_fn(base)))

    random.shuffle(target_lens)
    texts = _build_texts_at_lengths(tokenizer_path, target_lens)
    tokenizer = _load_tokenizer(tokenizer_path)
    _verify_lengths(tokenizer, texts, target_lens)

    meta = {
        "case": "case1_block_boundary",
        "description": "128n / 128n±1 / 128n±2 混合，n 为 128 的整数倍，验证 block-size 边界 KV cache 传输",
        "concurrency": 200,
        "output_len": 1024,
        "data_num": data_num,
        "length_distribution": "三类各约 1/3: 128n, 128n±1, 128n±2 (n∈128,256,384,512)",
    }
    return _save_dataset(save_dir, "case1_block_boundary_kv_cache.jsonl", texts, meta)


def generate_case2_max_len_boundary(tokenizer_path: str, save_dir: str, data_num: int,
                                    max_model_len: int, seed: int) -> str:
    """
    用例2: max_model_len / max_model_len±3 混合（裁剪到 [1, max_model_len]）。
    """
    random.seed(seed)
    deltas = [-3, -2, -1, 0, 1, 2, 3]
    target_lens = []
    for i in range(data_num):
        delta = deltas[i % len(deltas)]
        length = max(1, min(max_model_len, max_model_len + delta))
        target_lens.append(length)

    random.shuffle(target_lens)
    texts = _build_texts_at_lengths(tokenizer_path, target_lens)
    tokenizer = _load_tokenizer(tokenizer_path)
    _verify_lengths(tokenizer, texts, target_lens)

    meta = {
        "case": "case2_max_len_boundary",
        "description": "max_model_len±3 混合，验证最大长度边界场景",
        "max_model_len": max_model_len,
        "concurrency": 100,
        "output_len": 1024,
        "data_num": data_num,
        "length_distribution": f"max_model_len({max_model_len}) ± 3 循环混合",
    }
    return _save_dataset(save_dir, "case2_max_len_boundary.jsonl", texts, meta)


def generate_case3_business_traffic(tokenizer_path: str, save_dir: str, data_num: int, seed: int) -> str:
    """
    用例3: 30k(50%) / 64k(40%) / 90k(10%) 加权混合。
    """
    random.seed(seed)
    buckets = [
        (30720, 0.50),
        (65536, 0.40),
        (92160, 0.10),
    ]
    target_lens = []
    assigned = 0
    for length, ratio in buckets:
        count = int(round(data_num * ratio))
        if length == buckets[-1][0]:
            count = data_num - assigned
        target_lens.extend([length] * count)
        assigned += count

    random.shuffle(target_lens)
    texts = _build_texts_at_lengths(tokenizer_path, target_lens)
    tokenizer = _load_tokenizer(tokenizer_path)
    _verify_lengths(tokenizer, texts, target_lens)

    meta = {
        "case": "case3_business_traffic",
        "description": "30k(50%) / 64k(40%) / 90k(10%) 业务流量模拟",
        "concurrency": 200,
        "output_len": 1024,
        "data_num": data_num,
        "length_distribution": "30k:50%, 64k:40%, 90k:10%",
    }
    return _save_dataset(save_dir, "case3_business_traffic.jsonl", texts, meta)


def generate_case4_long_output(tokenizer_path: str, save_dir: str, data_num: int, seed: int) -> str:
    """
    用例4: 32k 定长输入，16k 输出（输出长度由测试命令指定）。
    """
    random.seed(seed)
    target_lens = [32768] * data_num
    texts = _build_texts_at_lengths(tokenizer_path, target_lens)

    meta = {
        "case": "case4_long_output",
        "description": "32k 输入 / 16k 输出，长输出重计算场景",
        "concurrency": 100,
        "input_len": 32768,
        "output_len": 16384,
        "data_num": data_num,
    }
    return _save_dataset(save_dir, "case4_long_output_32k_in.jsonl", texts, meta)


def generate_case5_long_tail(tokenizer_path: str, save_dir: str, data_num: int, seed: int) -> str:
    """
    用例5: 2k(90%) / 128k(10%) 长尾分布。
    """
    random.seed(seed)
    short_count = int(round(data_num * 0.90))
    long_count = data_num - short_count
    target_lens = [2048] * short_count + [131072] * long_count
    random.shuffle(target_lens)

    texts = _build_texts_at_lengths(tokenizer_path, target_lens)
    tokenizer = _load_tokenizer(tokenizer_path)
    _verify_lengths(tokenizer, texts, target_lens)

    meta = {
        "case": "case5_long_tail",
        "description": "2k(90%) / 128k(10%) 长尾分布场景",
        "concurrency": 200,
        "output_len": 1024,
        "data_num": data_num,
        "length_distribution": "2k:90%, 128k:10%",
    }
    return _save_dataset(save_dir, "case5_long_tail.jsonl", texts, meta)


# 混合业务流量长度分布（token）
# 注：用户原始需求中 "25k-25k" 按 25k-36k 处理，填补 25k 与 36k 区间
MIXED_TRAFFIC_BUCKETS = [
    (5000, 25000, 0.60),
    (25001, 35999, 0.07),
    (36000, 84000, 0.20),
    (84001, 93000, 0.05),
    (93001, 100000, 0.05),
    (100001, 128000, 0.03),
]

PREFIX_HIT_RATES = [0.10, 0.20, 0.30, 0.40]


def _clear_picked_ids():
    picked_ids = os.path.join(os.path.dirname(__file__), "picked_ids.txt")
    if os.path.exists(picked_ids):
        os.remove(picked_ids)
        logging.info("已清空 picked_ids.txt")


def generate_prefix_mixed_hit_dataset(tokenizer_path: str, save_dir: str, data_num: int,
                                      repeat_rate: float, seed: int, prefix_num: int = 1,
                                      dp: int = 1) -> tuple:
    """生成混合长度分布 + 指定 prefix cache 命中率的数据集"""
    _clear_picked_ids()
    os.makedirs(save_dir, exist_ok=True)

    hit_pct = int(round(repeat_rate * 100))
    logging.info("生成 prefix mixed 数据集: hit=%d%%, data_num=%d", hit_pct, data_num)

    prefix_path, dataset_path = create_prefix_dataset_bucketed(
        tokenizer_path, data_num, save_dir, MIXED_TRAFFIC_BUCKETS,
        repeat_rate, seed, prefix_num=prefix_num, dp=dp,
    )

    final_prefix = os.path.join(save_dir, f"prefix_mixed_hit{hit_pct}_prefix.jsonl")
    final_dataset = os.path.join(save_dir, f"prefix_mixed_hit{hit_pct}.jsonl")
    if os.path.exists(final_prefix):
        os.remove(final_prefix)
    if os.path.exists(final_dataset):
        os.remove(final_dataset)
    os.rename(prefix_path, final_prefix)
    os.rename(dataset_path, final_dataset)

    meta = {
        "case": f"prefix_mixed_hit{hit_pct}",
        "description": f"混合业务流量 + prefix cache 命中率 {hit_pct}%",
        "prefix_cache_hit_rate": repeat_rate,
        "data_num": data_num,
        "prefix_num": prefix_num,
        "dp": dp,
        "length_distribution": {
            "5k-25k": "60%",
            "25k-36k": "7%",
            "36k-84k": "20%",
            "84k-93k": "5%",
            "93k-100k": "5%",
            "100k-128k": "3%",
        },
        "prefix_file": final_prefix,
        "dataset_file": final_dataset,
    }
    meta_path = final_dataset.replace(".jsonl", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("[完成] prefix_mixed_hit%d (%d 条) -> %s", hit_pct, data_num, final_dataset)
    logging.info("[完成] 前缀文件 -> %s", final_prefix)
    return final_prefix, final_dataset


def generate_all_prefix_mixed_hit(tokenizer_path: str, save_dir: str, data_num: int,
                                  seed: int, prefix_num: int = 1, dp: int = 1) -> dict:
    generated = {}
    for idx, rate in enumerate(PREFIX_HIT_RATES):
        prefix_path, dataset_path = generate_prefix_mixed_hit_dataset(
            tokenizer_path, save_dir, data_num, rate, seed + idx * 100,
            prefix_num=prefix_num, dp=dp,
        )
        hit_pct = int(round(rate * 100))
        generated[f"prefix_mixed_hit{hit_pct}"] = dataset_path
        generated[f"prefix_mixed_hit{hit_pct}_prefix"] = prefix_path
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="生成 5 个 KV cache / 压测用例数据集")
    parser.add_argument("--model_path", type=str, default=None, help="模型权重路径（读取 tokenizer）")
    parser.add_argument("--save_dir", type=str, default=None, help="数据集输出目录")
    parser.add_argument("--max_model_len", type=int, default=None, help="用例2 的最大模型长度（默认读 config.MAX_MODEL_LEN）")
    parser.add_argument("--data_num_multiplier", type=int, default=4,
                        help="data_num = concurrency * multiplier（默认 4 倍并发）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=str, default="all",
                        help="要生成的用例: 1,2,3,4,5,prefix_mixed 或 all（不含 prefix_mixed）")
    parser.add_argument("--prefix_data_num", type=int, default=800,
                        help="prefix_mixed 数据集条数（默认 800）")
    parser.add_argument("--prefix_num", type=int, default=1, help="prefix 种类数")
    parser.add_argument("--dp", type=int, default=1, help="dp 域数量")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from config import DATASET_PATH, MODEL_PATH, MAX_MODEL_LEN as CONFIG_MAX_MODEL_LEN
    except ImportError:
        DATASET_PATH = "/root/aisbench_auto_tools_prefix/datasets"
        MODEL_PATH = "zai-org/GLM-5"
        CONFIG_MAX_MODEL_LEN = 131072

    model_path = args.model_path or MODEL_PATH
    save_dir = args.save_dir or os.path.join(DATASET_PATH, "custom_cases")
    max_model_len = args.max_model_len if args.max_model_len is not None else CONFIG_MAX_MODEL_LEN

    # 清理 picked_ids，避免 GSM8K 样本耗尽
    _clear_picked_ids()

    case_specs = {
        "1": ("case1", generate_case1_block_boundary, {"data_num": 200 * args.data_num_multiplier}),
        "2": ("case2", generate_case2_max_len_boundary,
              {"data_num": 100 * args.data_num_multiplier, "max_model_len": max_model_len}),
        "3": ("case3", generate_case3_business_traffic, {"data_num": 200 * args.data_num_multiplier}),
        "4": ("case4", generate_case4_long_output, {"data_num": 100 * args.data_num_multiplier}),
        "5": ("case5", generate_case5_long_tail, {"data_num": 200 * args.data_num_multiplier}),
    }

    if args.cases.strip().lower() == "all":
        selected = list(case_specs.keys())
    elif args.cases.strip().lower() == "prefix_mixed":
        selected = []
    else:
        selected = [c.strip() for c in args.cases.split(",") if c.strip() and c.strip() != "prefix_mixed"]

    logging.info("模型路径: %s", model_path)
    logging.info("输出目录: %s", save_dir)

    generated = {}

    if args.cases.strip().lower() == "prefix_mixed" or "prefix_mixed" in [c.strip() for c in args.cases.split(",")]:
        logging.info("生成用例: prefix_mixed (10%%/20%%/30%%/40%%)")
        prefix_generated = generate_all_prefix_mixed_hit(
            model_path, save_dir, args.prefix_data_num, args.seed,
            prefix_num=args.prefix_num, dp=args.dp,
        )
        generated.update(prefix_generated)

    if selected:
        logging.info("生成用例: %s", selected)
    for key in selected:
        if key not in case_specs:
            logging.error("未知用例编号: %s", key)
            continue
        name, func, kwargs = case_specs[key]
        kwargs = dict(kwargs)
        kwargs["tokenizer_path"] = model_path
        kwargs["save_dir"] = save_dir
        kwargs["seed"] = args.seed + int(key)
        path = func(**kwargs)
        generated[name] = path

    manifest_path = os.path.join(save_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"datasets": generated, "model_path": model_path}, f, ensure_ascii=False, indent=2)
    logging.info("manifest 已写入: %s", manifest_path)


if __name__ == "__main__":
    main()
