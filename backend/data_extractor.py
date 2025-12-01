import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NewsDataExtractor:
    """
    新闻数据抽取器
    从data目录下的各个类别文件夹中抽取新闻文本, 并为每个文件打上对应的类别标签
    """
    def __init__(self, data_dir: str):
        """
        初始化数据抽取器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.categories = []
        self.extracted_data = []

    def get_categories(self) -> List[str]:
        """
        获取所有新闻类别

        Returns:
            类别列表
        """
        if not self.categories:
            # 获取data目录下的所有文件夹（排除stopwords.txt等文件）
            entries = list(self.data_dir.iterdir())
            catagories = []

            for entry in tqdm(
                entries, desc="获取新闻类别", unit="个", ncols=80, colour="cyan"
            ):
                if entry.is_dir() and not entry.name.startswith("."):
                    catagories.append(entry.name)

            catagories.sort()  # 排序确保一致性
            logger.info(f"发现 {len(catagories)} 个新闻类别: {catagories}")
            self.categories = catagories

        return self.categories

    def extract_text_from_file(self, file_path: Path) -> str:
        """
        从单个文件中提取文本内容

        Args:
            file_path: 文件路径

        Returns:
            文件内容（去除首尾空白字符）
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="strict") as f:
                content = f.read().strip()
                return content

        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {file_path}") from None
        except Exception as e:
            raise RuntimeError(f"处理文件时发生未预期错误: {file_path}") from e

    def extract_category_data(self, category: str) -> List[Dict]:
        """抽取单个类别的所有数据（带进度条）"""
        category_dir = self.data_dir / category
        if not category_dir.exists():
            logger.warning(f"类别目录不存在: {category_dir}")
            return []

        txt_files = list(category_dir.glob("*.txt"))
        if not txt_files:
            return []

        category_data, error_count = [], 0

        for file_path in tqdm(
            txt_files,
            desc=f"处理新闻类别：{category}",
            unit="个文件",
            ncols=80,
            colour="cyan",
        ):
            try:
                content = self.extract_text_from_file(file_path)
                if content.strip():  # 跳过空白内容
                    category_data.append(
                        {
                            "file_id": file_path.stem,
                            "category": category,
                            "content": content,
                            "file_path": str(file_path),
                        }
                    )
            except Exception as e:
                error_count += 1
                logger.error(f"文件处理失败 [{file_path.name}]: {str(e)}")

        logger.info(
            f"完成 '{category}': {len(category_data)} 有效文件"
            + (f", {error_count} 个错误" if error_count else "")
        )
        return category_data

    def extract_all_data(self) -> List[Dict]:
        """
        抽取所有类别的数据

        Returns:
            所有数据的列表
        """
        categories = self.get_categories()
        all_data = []

        for category in categories:
            category_data = self.extract_category_data(category)
            all_data.extend(category_data)

        self.extracted_data = all_data
        logger.info(f"数据抽取完成，总共 {len(all_data)} 个文件")

        return all_data

    def save_to_json(self, output_file: str = "data/THUCNews.json") -> None:
        """将抽取的数据保存为JSON格式(自动创建目录)"""
        if not self.extracted_data:
            logger.warning("无数据可保存，请先调用 extract_all_data()")
            return

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.extracted_data, f, ensure_ascii=False, indent=2)
            logger.info(
                f"数据已保存至: {output_path} ({len(self.extracted_data)}条记录)"
            )
        except IOError as e:
            logger.error(f"文件写入失败 [{output_path}]: {e}")
            raise

    def save_to_csv(self, output_file: str = "results/extracted_news_data.csv") -> None:
        """保存数据为CSV格式(仅类别和内容)"""
        if not self.extracted_data:
            logger.warning("无数据可保存，请先调用 extract_all_data()")
            return

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 直接构建DataFrame并保存为CSV
            pd.DataFrame(
                [
                    {"category": item["category"], "content": item["content"]}
                    for item in self.extracted_data
                ]
            ).to_csv(
                output_path, index=False, encoding="utf_8_sig"
            )  # utf_8_sig解决Excel乱码

            logger.info(f"CSV已保存: {output_path} | {len(self.extracted_data)}条记录")
        except IOError as e:
            logger.error(f"文件写入失败: {e}")
            raise

    def get_statistics(self) -> Dict:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        if not self.extracted_data:
            logger.warning("没有数据可统计，请先调用 extract_all_data()")
            return {}

        # 按类别统计
        category_counts = {}
        total_chars = 0

        for item in self.extracted_data:
            category = item["category"]
            content = item["content"]

            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1

            total_chars += len(content)

        stats = {
            "total_files": len(self.extracted_data),
            "total_categories": len(category_counts),
            "total_characters": total_chars,
            "avg_chars_per_file": total_chars / len(self.extracted_data)
            if self.extracted_data
            else 0,
            "category_distribution": category_counts,
        }

        return stats

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        if not stats:
            return

        print("\n" + "=" * 50)
        print("数据统计信息")
        print("=" * 50)
        print(f"总文件数: {stats['total_files']}")
        print(f"总类别数: {stats['total_categories']}")
        print(f"总字符数: {stats['total_characters']:,}")
        print(f"平均每文件字符数: {stats['avg_chars_per_file']:.1f}")
        print("\n类别分布:")

        for category, count in sorted(stats["category_distribution"].items()):
            percentage = (count / stats["total_files"]) * 100
            print(f"  {category}: {count} 个文件 ({percentage:.1f}%)")
        print("=" * 50)


if __name__ == "__main__":
    extractor = NewsDataExtractor(data_dir="data/THUCNews")
    extractor.extract_all_data()
    # extractor.print_statistics()
    extractor.save_to_csv(output_file="data/THUCNews.csv")
