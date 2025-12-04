import pandas as pd
import os
import pickle  # 用于保存模型
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # 引入进度条库
import warnings

warnings.filterwarnings("ignore")

# 设置matplotlib中文字体
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


class THUNewsClassifier:
    """
    适配 THUCTCNew 数据集的朴素贝叶斯分类器
    (支持加载预分割的 train.csv 和 test.csv，并支持保存模型和报告)
    """

    def __init__(
        self,
        train_path="data/cleaned_THUCTCNews/train.csv",
        test_path="data/cleaned_THUCTCNews/test.csv",
        model_save_path="naive_bayes_model.pkl",
        report_save_path="naive_bayes_report.txt",
    ):
        """
        初始化分类器
        :param train_path: 训练集CSV文件路径
        :param test_path: 测试集CSV文件路径
        :param model_save_path: 模型保存路径 (.pkl)
        :param report_save_path: 评估报告保存路径 (.txt)
        """
        self.train_path = train_path
        self.test_path = test_path
        self.model_save_path = model_save_path
        self.report_save_path = report_save_path

        self.train_df = None
        self.test_df = None
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = LabelEncoder()

        # 存放训练和测试数据
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

        # 结果指标
        self.metrics = {}

    def load_data(self, max_samples=None):
        """加载训练集和测试集数据"""
        if not os.path.exists(self.train_path):
            print(self.train_path)
            raise FileNotFoundError(f"找不到训练集文件: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"找不到测试集文件: {self.test_path}")

        print(f"\n[INFO] 正在从 {self.train_path} 和 {self.test_path} 加载数据...")

        # 读取训练集
        self.train_df = pd.read_csv(self.train_path, nrows=max_samples)
        self.train_df.dropna(subset=["content_cleaned", "category"], inplace=True)
        self.train_df["content_cleaned"] = self.train_df["content_cleaned"].astype(str)
        print(f"  - 训练集加载成功: {len(self.train_df)} 条记录")

        # 读取测试集
        self.test_df = pd.read_csv(self.test_path, nrows=max_samples)
        self.test_df.dropna(subset=["content_cleaned", "category"], inplace=True)
        self.test_df["content_cleaned"] = self.test_df["content_cleaned"].astype(str)
        print(f"  - 测试集加载成功: {len(self.test_df)} 条记录")

    def prepare_data(self):
        """数据预处理：标签编码"""
        if self.train_df is None or self.test_df is None:
            raise ValueError("数据未加载")

        print("\n[INFO] 正在进行标签编码...")

        self.X_train = self.train_df["content_cleaned"]
        self.X_test = self.test_df["content_cleaned"]

        # 拟合标签
        self.y_train = self.label_encoder.fit_transform(self.train_df["category"])

        # 处理测试集标签
        try:
            self.y_test = self.label_encoder.transform(self.test_df["category"])
        except ValueError:
            print("  [WARN] 测试集包含未知类别，正在重新拟合所有标签...")
            all_labels = pd.concat(
                [self.train_df["category"], self.test_df["category"]]
            )
            self.label_encoder.fit(all_labels)
            self.y_train = self.label_encoder.transform(self.train_df["category"])
            self.y_test = self.label_encoder.transform(self.test_df["category"])

        classes = self.label_encoder.classes_
        print(f"  - 类别数量: {len(classes)}")
        print(f"  - 类别列表: {list(classes)}")

    def vectorize_data(self, max_features=50000):
        """文本向量化 (TF-IDF)"""
        if self.X_train is None:
            raise ValueError("数据未准备好")

        print("\n[INFO] 正在进行 TF-IDF 向量化...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            token_pattern=r"(?u)\b\w+\b",
        )

        print("  - 正在 fit_transform 训练集 (这可能需要一点时间)...")
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)

        print("  - 正在 transform 测试集...")
        self.X_test_vec = self.vectorizer.transform(self.X_test)

        print(f"  - 向量化完成。特征矩阵形状: {self.X_train_vec.shape}")

    def train_model(self, alpha=0.01):
        """训练朴素贝叶斯模型"""
        if self.vectorizer is None:
            raise ValueError("数据未向量化")

        print("\n[INFO] 正在训练 MultinomialNB 模型...")
        self.classifier = MultinomialNB(alpha=alpha)
        self.classifier.fit(self.X_train_vec, self.y_train)
        print("  - 模型训练完成")

    def save_model(self):
        """保存模型参数"""
        if self.classifier is None:
            raise ValueError("模型未训练，无法保存")

        print(f"\n[INFO] 正在保存模型到 {self.model_save_path} ...")

        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "label_encoder": self.label_encoder,
        }

        try:
            with open(self.model_save_path, "wb") as f:
                pickle.dump(model_data, f)
            print("  - 模型保存成功！下次可直接加载使用。")
        except Exception as e:
            print(f"  [ERROR] 模型保存失败: {e}")

    def evaluate_model(self):
        """模型评估与预测"""
        if self.classifier is None:
            raise ValueError("模型未训练")

        print("\n[INFO] 正在评估模型 (预测测试集)...")
        self.y_pred = self.classifier.predict(self.X_test_vec)

        self.metrics["accuracy"] = accuracy_score(self.y_test, self.y_pred)
        self.metrics["f1_weighted"] = f1_score(
            self.y_test, self.y_pred, average="weighted"
        )
        print(f"  - 预测完成。准确率: {self.metrics['accuracy']:.4f}")

    def generate_report(self):
        """生成详细报告并保存"""
        print("\n[INFO] 正在生成并保存报告...")

        # 1. 生成分类报告字符串
        report_str = classification_report(
            self.y_test, self.y_pred, target_names=self.label_encoder.classes_
        )

        # 2. 打印到控制台
        def print_and_write(report_save_path, content):
            print(content)
            with open(report_save_path, "a", encoding="utf-8") as f:
                f.write(content + "\n")

        # 3. 保存到文件
        try:
            # with open(self.report_save_path, "w", encoding="utf-8") as f:
            print_and_write(self.report_save_path, "THUCTCNew 新闻分类器评估报告")
            print_and_write(self.report_save_path, "=" * 60 + "\n")
            print_and_write(
                self.report_save_path,
                f"模型准确率 (Accuracy): {self.metrics['accuracy']:.4f}",
            )

            print_and_write(
                self.report_save_path,
                f"加权 F1 分数 (Weighted F1): {self.metrics['f1_weighted']:.4f}\n",
            )
            print_and_write(self.report_save_path, "详细分类报告:")
            print_and_write(self.report_save_path, report_str)
            print_and_write(
                self.report_save_path, "\n混淆矩阵已保存为: confusion_matrix.png"
            )

            print(f"  - 文本报告已保存至: {self.report_save_path}")
        except Exception as e:
            print(f"  [ERROR] 保存文本报告失败: {e}")

        # 4. 绘制并保存混淆矩阵
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.title("THUCTC 新闻分类混淆矩阵")
        plt.ylabel("真实标签")
        plt.xlabel("预测标签")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        print("  - 混淆矩阵图已保存至: confusion_matrix.png")

    def run(self):
        """主运行函数"""
        tasks = [
            ("1. 加载数据 (Loading Data)", self.load_data),
            ("2. 准备数据 (Label Encoding)", self.prepare_data),
            ("3. 特征工程 (Vectorizing)", self.vectorize_data),
            ("4. 模型训练 (Training)", self.train_model),
            ("5. 模型保存 (Saving Model)", self.save_model),  # 训练后立即保存
            ("6. 模型评估 (Evaluating)", self.evaluate_model),
            ("7. 生成报告 (Reporting)", self.generate_report),
        ]

        print(">>> 开始执行文本分类任务 pipeline...")

        with tqdm(
            total=len(tasks),
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for desc, func in tasks:
                pbar.set_description(
                    f"阶段: {desc.split(' ')[1]}"
                )  # 简化进度条左侧显示
                try:
                    func()
                except Exception as e:
                    print(f"\n[CRITICAL] 错误发生在步骤 [{desc}]: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    break
                pbar.update(1)

        print("\n" + "=" * 60)
        if "accuracy" in self.metrics:
            print(f"流程圆满结束！最终准确率: {self.metrics['accuracy']:.4f}")
            print(f"模型已保存至: {self.model_save_path}")
            print(f"报告已保存至: {self.report_save_path}")
        else:
            print("流程异常终止。")


if __name__ == "__main__":
    classifier = THUNewsClassifier(
        train_path="data/cleaned_THUCTCNews_Augmented/train.csv",
        test_path="data/cleaned_THUCTCNews/test.csv",
        model_save_path="backend/parameters/augmented_naive_bayes_model.pkl",
        report_save_path="data/reports/augmented_naive_bayes_report.txt",
    )

    classifier.run()
