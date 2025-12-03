import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm


class DataAnalyzer:
    def __init__(self, df):
        """
        初始化分析器
        :param df: 包含 'category' 和 'content_cleaned' 列的 DataFrame
        """
        self.df = df

    def _analyze_single_category(self, category):
        """分析单个类别的数据"""
        try:
            # 筛选该类别的数据
            # 使用 .copy() 避免 SettingWithCopyWarning
            cat_df = self.df[self.df["category"] == category].copy()

            total_news = len(cat_df)
            if total_news == 0:
                return None, None

            # -------------------------------------------------
            # 1. 文本长度统计
            # -------------------------------------------------
            # 确保转为字符串处理，防止由 NaN 引起的报错
            cat_df["content_str"] = cat_df["content_cleaned"].astype(str)

            # 字符数：计算去除空格后的纯字符长度
            cat_df["char_len"] = cat_df["content_str"].apply(
                lambda x: len(x.replace(" ", ""))
            )

            # 词数：直接按空格分割计算
            cat_df["word_len"] = cat_df["content_str"].apply(lambda x: len(x.split()))

            char_lengths = cat_df["char_len"]
            word_lengths = cat_df["word_len"]

            # 计算统计指标
            stats = {
                "category": category,
                "total_news": total_news,
                "avg_chars": char_lengths.mean(),
                "median_chars": char_lengths.median(),
                "std_chars": char_lengths.std(),
                "min_chars": char_lengths.min(),
                "max_chars": char_lengths.max(),
                "avg_words": word_lengths.mean(),
                "median_words": word_lengths.median(),
                "std_words": word_lengths.std(),
                "min_words": word_lengths.min(),
                "max_words": word_lengths.max(),
                "q25_chars": char_lengths.quantile(0.25),
                "q75_chars": char_lengths.quantile(0.75),
                "q25_words": word_lengths.quantile(0.25),
                "q75_words": word_lengths.quantile(0.75),
            }

            # -------------------------------------------------
            # 2. 词频统计
            # -------------------------------------------------
            all_words = []
            # 直接遍历 Series，不再使用 jieba，利用已有的分词结果（空格分隔）
            for content in cat_df["content_str"]:
                words = content.split()
                # 过滤掉空字符
                all_words.extend([w for w in words if len(w.strip()) > 0])

            word_freq = Counter(all_words)
            # 取前 10 个高频词
            top_words = word_freq.most_common(10)

            return stats, top_words

        except Exception as e:
            # 打印错误但不中断整个程序
            print(f"\n[Error] 分析类别 {category} 时出错: {e}")
            return None, None

    def run(self, output_file="data/detailed_analysis_report.txt") -> None:
        """执行全部分析并生成报告"""
        print("开始中文新闻数据集详细统计分析...")
        print("=" * 60)

        all_stats = []
        all_top_words = {}

        # 自动获取所有类别
        categories = self.df["category"].unique()

        # 使用 tqdm 显示进度条
        with tqdm(categories, unit="类") as pbar:
            for category in pbar:
                pbar.set_description(f"正在分析 [{category}]类")
                stats, top_words = self._analyze_single_category(category)

                if stats:
                    all_stats.append(stats)
                    all_top_words[category] = top_words

        print("\n分析完成，正在生成报告文件...")
        self._generate_report(all_stats, all_top_words, output_file)

    def _generate_report(self, all_stats, all_top_words, output_file) -> None:
        """生成并保存报告"""
        if not all_stats:
            print("没有生成有效的统计数据。")
            return

        def print_and_write(f, text) -> None:
            # 同时打印到控制台和写入文件
            print(text)
            f.write(text + "\n")

        with open(output_file, "w", encoding="utf-8") as f:
            header = "中文新闻数据集详细统计分析报告"
            print_and_write(f, header)
            print_and_write(f, "=" * 80 + "\n")

            # 1. 基本统计表
            print_and_write(f, "1. 基本统计信息")
            print_and_write(f, "-" * 80)
            print_and_write(
                f,
                f"{'类别':<6} {'新闻数':<8} {'平均字符':<10} {'中位字符':<10} {'平均词数':<10} {'中位词数':<10}",
            )
            print_and_write(f, "-" * 80)

            for stats in all_stats:
                line = f"{stats['category']:<6} {stats['total_news']:<8,} {stats['avg_chars']:<10.1f} {stats['median_chars']:<10.1f} {stats['avg_words']:<10.1f} {stats['median_words']:<10.1f}"
                print_and_write(f, line)

            # 2. 字符长度分布
            print_and_write(f, "\n2. 字符长度分布")
            print_and_write(f, "-" * 80)
            print_and_write(
                f,
                f"{'类别':<6} {'最小值':<8} {'Q25':<8} {'中位数':<8} {'Q75':<8} {'最大值':<8} {'标准差':<8}",
            )
            print_and_write(f, "-" * 80)

            for stats in all_stats:
                line = f"{stats['category']:<6} {stats['min_chars']:<8} {stats['q25_chars']:<8.0f} {stats['median_chars']:<8.0f} {stats['q75_chars']:<8.0f} {stats['max_chars']:<8} {stats['std_chars']:<8.1f}"
                print_and_write(f, line)

            # 3. 高频词统计
            print_and_write(f, "\n3. 各类别高频词（前10个）")
            print_and_write(f, "-" * 80)

            for category, top_words in all_top_words.items():
                print_and_write(f, f"\n【{category}】:")
                for i, (word, count) in enumerate(top_words, 1):
                    print_and_write(f, f"  {i}. {word} ({count:,}次)")

            # 4. 总体统计
            print_and_write(f, "\n4. 总体统计")
            print_and_write(f, "-" * 80)

            total_news = sum(stats["total_news"] for stats in all_stats)
            avg_chars_all = np.mean([stats["avg_chars"] for stats in all_stats])
            news_counts = [stats["total_news"] for stats in all_stats]
            imbalance_ratio = max(news_counts) / min(news_counts) if news_counts else 0

            print_and_write(f, f"总新闻数: {total_news:,}")
            print_and_write(f, f"平均字符数: {avg_chars_all:.1f}")
            print_and_write(f, f"类别数: {len(all_stats)}")
            print_and_write(f, f"数据不平衡比例 (Max/Min): {imbalance_ratio:.2f}:1")

        print(f"详细报告已保存到当前目录: {output_file}")


if __name__ == "__main__":
    df_filtered = pd.read_csv("data/cleaned_THUCTCNews/data_cleaned.csv")

    analyzer = DataAnalyzer(df_filtered)
    analyzer.run("data/detailed_analysis_report.txt")
