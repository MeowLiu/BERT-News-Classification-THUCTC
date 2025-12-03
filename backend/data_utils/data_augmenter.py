import pandas as pd
import nlpaug.augmenter.word as naw
import random
import os
import torch
from tqdm import tqdm


class DataAugmenter:
    """数据增强器，结合 BERT 上下文替换和随机删除字符"""

    def __init__(
        self,
        df,
        target_count=50000,
        bert_ratio=0.3,
        device="cuda",
        log_path="data/augment_analyze.txt",
    ):
        """
        初始化数据增强器
        :param df: 原始 DataFrame，必须包含 'category' 和 'content_cleaned' 列
        :param target_count: 每个类别的目标数量
        :param bert_ratio: 使用 BERT 增强的比例 (0.0 - 1.0)，剩余使用随机删除
        :param device: 'cuda' 或 'cpu'
        :param log_path: 分析结果保存路径
        """
        self.raw_df = df.copy()
        self.aug_df = None
        self.target_count = target_count
        self.bert_ratio = bert_ratio
        self.log_path = log_path
        self.device = device if torch.cuda.is_available() else "cpu"

        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # 初始化 BERT 模型 (仅当需要时)
        if bert_ratio > 0:
            print(f"[Init] 正在加载 BERT 模型至 {self.device}...")
            self.aug_bert = naw.ContextualWordEmbsAug(
                model_path="bert-base-chinese",
                action="substitute",
                device=self.device,
                aug_p=0.15,
            )
        else:
            self.aug_bert = None

    def _clean_text(self, text):
        """去除空格，还原为自然句"""
        if not isinstance(text, str):
            return ""
        return text.replace(" ", "")

    def _random_deletion(self, text, p=0.15):
        """随机删除字符"""
        chars = list(text)
        if len(chars) <= 1:
            return text

        new_chars = [c for c in chars if random.random() > p]

        if not new_chars:
            return random.choice(chars)

        return "".join(new_chars)

    def process(self):
        """执行增强主逻辑"""
        augmented_dfs = []
        categories = self.raw_df["category"].unique()

        print(f"\n[Start] 开始增强处理，目标数量: {self.target_count}")

        for cat in categories:
            subset = self.raw_df[self.raw_df["category"] == cat]
            current_count = len(subset)

            # 保留原始数据
            augmented_dfs.append(subset)

            if current_count >= self.target_count:
                continue

            diff = self.target_count - current_count
            n_bert = int(diff * self.bert_ratio)
            n_del = diff - n_bert

            print(
                f" -> 处理类别 [{cat}]: 原有 {current_count} | 需补 {diff} (BERT: {n_bert}, Del: {n_del})"
            )

            new_rows = []

            # --- BERT 增强 ---
            if n_bert > 0 and self.aug_bert:
                seeds = subset["content_cleaned"].sample(
                    n=n_bert, replace=True, random_state=42
                )
                failed_count = 0
                for text in tqdm(seeds, desc="BERT Augmenting", leave=False):
                    clean_txt = self._clean_text(text)

                    # 跳过空文本或过短文本
                    if len(clean_txt) < 3:
                        failed_count += 1
                        continue

                    try:
                        # 尝试最多2次
                        for attempt in range(2):
                            try:
                                res = self.aug_bert.augment(clean_txt)[0]
                                # 检查增强结果是否有效
                                if isinstance(res, str) and len(res) > 0:
                                    new_rows.append(
                                        {"category": cat, "content_cleaned": res}
                                    )
                                    break
                                else:
                                    # 如果结果不是字符串或为空，视为失败
                                    if attempt == 1:
                                        failed_count += 1
                            except (RuntimeError, IndexError, ValueError):
                                if attempt == 1:  # 最后一次尝试也失败了
                                    failed_count += 1
                                continue
                    except Exception:
                        # 捕获其他未预期的异常
                        failed_count += 1

                if failed_count > 0:
                    print(f"   -> BERT增强失败: {failed_count}/{n_bert} 个样本")

            # --- 随机删除增强 ---
            if n_del > 0:
                seeds = subset["content_cleaned"].sample(
                    n=n_del, replace=True, random_state=42
                )
                for text in seeds:  # 随机删除很快，一般不需要进度条，或者简单处理
                    clean_txt = self._clean_text(text)
                    res = self._random_deletion(clean_txt, p=0.15)
                    new_rows.append({"category": cat, "content_cleaned": res})

            if new_rows:
                augmented_dfs.append(pd.DataFrame(new_rows))

        # 合并结果
        self.aug_df = (
            pd.concat(augmented_dfs)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        print("[Done] 数据增强完成。")

        # 执行分析
        self._analyze_and_save()

        return self.aug_df

    def _analyze_and_save(self):
        """统计分析并写入文件"""
        if self.aug_df is None:
            return

        # 准备统计数据
        original_counts = self.raw_df["category"].value_counts()
        new_counts = self.aug_df["category"].value_counts()

        # 计算不平衡度 (Max / Min)
        imb_rate_origin = original_counts.max() / original_counts.min()
        imb_rate_new = new_counts.max() / new_counts.min()

        # 构建输出文本
        lines = []
        lines.append("=" * 40)
        lines.append("DATA AUGMENTATION ANALYSIS REPORT")
        lines.append("=" * 40)
        lines.append(f"原始总数: {len(self.raw_df)}")
        lines.append(f"增强后总数: {len(self.aug_df)}")
        lines.append(f"新增样本数: {len(self.aug_df) - len(self.raw_df)}")
        lines.append("-" * 40)
        lines.append(f"原始类别不平衡度 (Max/Min): {imb_rate_origin:.2f}")
        lines.append(f"当前类别不平衡度 (Max/Min): {imb_rate_new:.2f}")
        lines.append("-" * 40)
        lines.append(
            f"{'Category':<10} | {'Original':<10} | {'Augmented':<10} | {'Growth':<10}"
        )
        lines.append("-" * 45)

        for cat in new_counts.index:  # type: ignore
            orig = original_counts.get(cat, 0)
            curr = new_counts.get(cat, 0)
            growth = curr - orig
            lines.append(f"{cat:<10} | {orig:<10} | {curr:<10} | +{growth:<10}")

        lines.append("=" * 40)
        output_text = "\n".join(lines)

        # 1. 打印到控制台
        print(output_text)

        # 2. 写入文件
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"\n[Info] 分析报告已保存至: {self.log_path}")


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_THUCTCNews/data_cleaned.csv")

    # 实例化增强器
    # 注意：target_count 设置为期望的每类数量
    # bert_ratio=0.3 表示 30% 的新数据由 BERT 生成，70% 由随机删除生成
    augmenter = DataAugmenter(
        df,
        target_count=50000,  # 建议 50000 或更多
        bert_ratio=0.3,
        log_path="data/augment_analysis_report.txt",
    )

    # 运行增强
    df_new = augmenter.process()

    # 保存最终 csv
    df_new.to_csv("data/THUCTCNews_Augmented.csv", index=False)
