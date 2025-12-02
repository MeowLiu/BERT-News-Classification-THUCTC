#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THUCNews 数据预处理脚本
优化点：加入tqdm进度条、正则预编译、代码精简
"""

import pandas as pd
import numpy as np
import re
import jieba
import logging
import time
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 初始化 tqdm 对 pandas 的支持
tqdm.pandas()

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(
        self, stopwords_file: str = "data/stopwords.txt", random_seed: int = 42
    ):
        self.stopwords_file = Path(stopwords_file)
        self.stopwords = self._load_stopwords()
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # --- 优化：预编译正则表达式，避免在循环中重复编译 ---
        self.pat_html = re.compile(r"<[^>]+>")
        self.pat_url = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.pat_email = re.compile(r"\S+@\S+")
        # 保留中文、英文
        self.pat_non_content = re.compile(r"[^\u4e00-\u9fa5a-zA-Z]")
        self.pat_whitespace = re.compile(r"\s+")

        # 配置参数
        self.config = {
            "min_len": 10,  # 最小长度
            "max_len": 2000,  # 最大长度
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "remove_empty": True,
        }

    def _load_stopwords(self) -> set:
        """加载停用词"""
        if not self.stopwords_file.exists():
            logger.warning(
                f"停用词文件不存在: {self.stopwords_file}，将不使用停用词过滤。"
            )
            return set()
        try:
            with open(self.stopwords_file, "r", encoding="utf-8") as f:
                stopwords = {line.strip() for line in f if line.strip()}
            logger.info(f"加载停用词: {len(stopwords)} 个")
            return stopwords
        except Exception as e:
            logger.error(f"加载停用词出错: {e}")
            return set()

    def clean_text(self, text: str) -> str:
        """文本清洗核心函数"""
        if not isinstance(text, str) or not text:
            return ""

        # 利用预编译的正则进行替换
        text = self.pat_html.sub("", text)
        text = self.pat_url.sub("", text)
        text = self.pat_email.sub("", text)
        text = self.pat_non_content.sub(" ", text)  # 非中英文字符替换为空格
        text = self.pat_whitespace.sub(" ", text).strip()

        if not text:
            return ""

        # 分词并去停用词
        words = jieba.lcut(text)
        # 列表推导式过滤，效率略高于循环 append
        cleaned_words = [
            w for w in words if w not in self.stopwords and len(w.strip()) > 0
        ]

        return " ".join(cleaned_words)

    def process(self, input_file: str, output_dir: str = "data/cleaned_THUCNews") -> Dict:
        """执行完整处理流程"""
        start_time = time.time()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. 读取数据
        logger.info(f"读取文件: {input_file}")
        if not Path(input_file).exists():
            raise FileNotFoundError(f"文件未找到: {input_file}")

        df = pd.read_csv(input_file, encoding="utf-8")
        total_count = len(df)

        # 2. 清洗数据 (加入 tqdm 进度条)
        logger.info("正在清洗文本 (请耐心等待)...")
        # progress_apply 替代 apply
        df["content_cleaned"] = df["content"].progress_apply(self.clean_text)

        # 3. 过滤数据
        logger.info("开始过滤数据：移除无效、过短或过长的文本...")

        # 第一步：计算每条文本清洗后的长度
        # 使用 .str.len() 方法获取 'content_cleaned' 列中每个字符串的长度，并存入新列 'len'
        df["len"] = df["content_cleaned"].str.len()

        # 第二步：创建长度过滤条件 (Mask)
        # mask_length 是一个布尔序列(Series)，其中符合条件的行值为 True，不符合为 False
        # 条件：长度 >= 最小长度  并且(&)  长度 <= 最大长度
        mask_length = (df["len"] >= self.config["min_len"]) & (
            df["len"] <= self.config["max_len"]
        )

        # 第三步：处理非空过滤条件
        # 初始化最终的过滤条件为长度条件
        final_condition = mask_length

        # 如果配置要求移除空文本 (remove_empty=True)
        if self.config["remove_empty"]:
            # 创建一个标记“非空文本”的条件
            mask_not_empty = df["content_cleaned"] != ""
            # 将“长度合适”和“内容不为空”两个条件合并
            # 使用 & (逻辑与) 运算符：必须同时满足两个条件才为 True
            final_condition = final_condition & mask_not_empty

        # 第四步：应用过滤
        # df[final_condition] 只会选出条件为 True 的那些行
        # .copy() 是为了创建一个新的 DataFrame 对象，防止后续修改时出现 SettingWithCopyWarning 警告
        df_filtered = df[final_condition].copy()

        # 4. 数据切分
        logger.info("切分数据集...")
        train_df, val_df, test_df = self._split_data(df_filtered)

        # 5. 保存文件
        logger.info(f"保存结果至 {output_dir}...")
        df_filtered[["category", "content_cleaned"]].to_csv(
            out_path / "data_cleaned.csv", index=False
        )
        train_df[["category", "content_cleaned"]].to_csv(
            out_path / "train.csv", index=False
        )
        val_df[["category", "content_cleaned"]].to_csv(
            out_path / "val.csv", index=False
        )
        test_df[["category", "content_cleaned"]].to_csv(
            out_path / "test.csv", index=False
        )

        # 6. 生成并打印统计报告
        stats = {
            "total": total_count,
            "valid": len(df_filtered),
            "removed": total_count - len(df_filtered),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "time_sec": round(time.time() - start_time, 2),
        }
        self._print_stats(stats)
        return stats

    def _split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """切分训练/验证/测试集"""
        ratios = self.config["split_ratios"]
        # 第一次切分：分出训练集
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - ratios["train"]),
            random_state=self.random_seed,
            shuffle=True,
        )
        # 第二次切分：将剩余部分分给验证和测试
        relative_test_ratio = ratios["test"] / (ratios["val"] + ratios["test"])
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            random_state=self.random_seed,
            shuffle=True,
        )
        return train_df, val_df, test_df

    def _print_stats(self, stats: Dict):
        """打印简报"""
        print(f"\n{'=' * 20} 处理完成 {'=' * 20}")
        print(f"原始数据: {stats['total']}")
        print(f"有效数据: {stats['valid']} (剔除 {stats['removed']})")
        print(
            f"数据集划分: Train={stats['train']}, Val={stats['val']}, Test={stats['test']}"
        )
        print(f"总耗时: {stats['time_sec']} 秒")
        print("=" * 50)


if __name__ == "__main__":
    # 使用示例
    input_csv = "data/THUCNews.csv"  # 替换为你的实际文件名

    # 简单的检查文件是否存在，防止直接运行报错
    if not Path(input_csv).exists():
        # 生成一个假文件用于测试运行
        print("未找到输入文件，正在生成测试数据 sample.csv ...")
        pd.DataFrame(
            {
                "id": range(100),
                "category": ["体育"] * 50 + ["财经"] * 50,
                "content": ["这是一段测试文本，包含HTML标签<br>和http://test.com链接"]
                * 100,
            }
        ).to_csv("sample.csv", index=False)
        input_csv = "sample.csv"

    processor = DataPreprocessor()
    processor.process(input_csv)
