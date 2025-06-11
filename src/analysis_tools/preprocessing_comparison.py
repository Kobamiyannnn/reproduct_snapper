"""
前処理実施前後で検証用データセットがどのように変化したのかを確認するための分析ツール。
モデルを使用せずに、検証フェーズでの評価指標を取得し、jitteringによる人手ラフアノテーション再現の効果を測定します。
"""

import os
import sys
import torch
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import japanize_matplotlib


# 親ディレクトリからのインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dataset import CocoAdjustmentDataset
import utils


class ValidationDatasetAnalyzer:
    """検証データセットの評価指標を分析するクラス"""

    def __init__(self, annotations_path, img_dir, device="cpu"):
        self.annotations_path = annotations_path
        self.img_dir = img_dir
        self.device = device

        # CUDA利用可能性の確認と通知
        if torch.cuda.is_available():
            print(f"CUDA利用可能: {torch.cuda.get_device_name()}")
            print(f"指定デバイス: {device}")
        else:
            print("CUDA利用不可: CPUで実行します")

    def analyze_validation_metrics(self, use_full_dataset=True, batch_size=32):
        """
        検証用データセット全体での評価指標を分析

        Args:
            use_full_dataset: 全データセットを使用するか
            batch_size: バッチサイズ

        Returns:
            dict: 分析結果の辞書
        """
        print("検証データセットの評価指標分析を開始します...")

        # 前処理なしのデータセット（元のGTアノテーション）
        dataset_clean = CocoAdjustmentDataset(
            self.annotations_path, self.img_dir, for_training=False
        )

        # 前処理ありのデータセット（人手ラフアノテーション再現）
        dataset_jittered = CocoAdjustmentDataset(
            self.annotations_path, self.img_dir, for_training=True
        )

        if use_full_dataset:
            indices = list(range(len(dataset_clean)))
        else:
            # テスト用に少数のサンプルを使用
            indices = list(range(min(500, len(dataset_clean))))

        print(f"分析対象サンプル数: {len(indices)}")

        # データ収集（バッチ処理で効率化）
        clean_results = []
        jittered_results = []

        print("元のクリーンアノテーションでの評価指標取得中...")
        for i in tqdm(indices):
            try:
                image, targets = dataset_clean[i]
                # targetsのtensorをGPUに移動
                targets_gpu = {}
                for key, value in targets.items():
                    if isinstance(value, torch.Tensor):
                        targets_gpu[key] = value.to(self.device)
                    else:
                        targets_gpu[key] = value

                gt_bbox = self._extract_gt_bbox_from_targets(targets_gpu)
                clean_results.append(
                    {"index": i, "gt_bbox": gt_bbox, "targets": targets_gpu}
                )
            except Exception as e:
                print(f"警告: サンプル {i} でクリーンデータセットの処理中にエラー: {e}")
                continue

        print("人手ラフアノテーション再現での評価指標取得中...")
        for i in tqdm(indices):
            try:
                image, targets = dataset_jittered[i]
                # targetsのtensorをGPUに移動
                targets_gpu = {}
                for key, value in targets.items():
                    if isinstance(value, torch.Tensor):
                        targets_gpu[key] = value.to(self.device)
                    else:
                        targets_gpu[key] = value

                gt_bbox = self._extract_gt_bbox_from_targets(targets_gpu)
                jittered_results.append(
                    {"index": i, "gt_bbox": gt_bbox, "targets": targets_gpu}
                )
            except Exception as e:
                print(f"警告: サンプル {i} でジッターデータセットの処理中にエラー: {e}")
                continue

        # 評価指標計算
        analysis_results = self._compute_validation_metrics(
            clean_results, jittered_results
        )

        return analysis_results

    def _extract_gt_bbox_from_targets(self, targets):
        """ターゲットベクトルからGTバウンディングボックスを抽出"""
        # GPU上でargmaxを実行してからCPUに移動
        gt_y_min = torch.argmax(targets["top"]).cpu().item()
        gt_y_max = torch.argmax(targets["bottom"]).cpu().item()
        gt_x_min = torch.argmax(targets["left"]).cpu().item()
        gt_x_max = torch.argmax(targets["right"]).cpu().item()

        return [gt_x_min, gt_y_min, gt_x_max, gt_y_max]

    def _compute_validation_metrics(self, clean_results, jittered_results):
        """検証時の評価指標を計算"""
        print("検証評価指標を計算中...")

        # 共通のindexを持つサンプルのみを比較対象とする
        indices_clean = {r["index"] for r in clean_results}
        indices_jittered = {r["index"] for r in jittered_results}
        common_indices = indices_clean.intersection(indices_jittered)

        print(f"評価可能なサンプル数: {len(common_indices)}")

        # データを整理
        clean_data = {
            r["index"]: r for r in clean_results if r["index"] in common_indices
        }
        jittered_data = {
            r["index"]: r for r in jittered_results if r["index"] in common_indices
        }

        # バウンディングボックス情報の収集
        bboxes_clean = []
        bboxes_jittered = []

        for idx in sorted(common_indices):
            bboxes_clean.append(clean_data[idx]["gt_bbox"])
            bboxes_jittered.append(jittered_data[idx]["gt_bbox"])

        # tensorに変換（deviceを指定）
        bboxes_clean_tensor = torch.tensor(
            bboxes_clean, dtype=torch.float32, device=self.device
        )
        bboxes_jittered_tensor = torch.tensor(
            bboxes_jittered, dtype=torch.float32, device=self.device
        )

        print(f"Tensorをデバイス {self.device} で処理中...")
        print(f"処理サンプル数: {len(bboxes_clean)}")

        # GPU使用状況の詳細確認
        if self.device.type == "cuda":
            print(f"GPU メモリ使用量: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"tensor1 device: {bboxes_clean_tensor.device}")
            print(f"tensor2 device: {bboxes_jittered_tensor.device}")

        # 検証時の評価指標を計算
        # 1. クリーンアノテーションをGTとして、ジッターアノテーションを予測として評価
        print("評価指標計算中（人手ラフ vs クリーン）...")
        jittered_vs_clean_metrics = utils.compute_metrics(
            bboxes_jittered_tensor,  # 人手ラフアノテーション（予測として扱う）
            bboxes_clean_tensor,  # クリーンアノテーション（GTとして扱う）
            iou_threshold=config.IOU_THRESHOLD,
            deviance_thresholds=config.DEVIANCE_THRESHOLDS,
        )

        # 2. 逆方向の評価（参考用）
        print("評価指標計算中（クリーン vs 人手ラフ）...")
        clean_vs_jittered_metrics = utils.compute_metrics(
            bboxes_clean_tensor,  # クリーンアノテーション
            bboxes_jittered_tensor,  # 人手ラフアノテーション
            iou_threshold=config.IOU_THRESHOLD,
            deviance_thresholds=config.DEVIANCE_THRESHOLDS,
        )

        # GPU メモリクリア（必要に応じて）
        if self.device.type == "cuda":
            del bboxes_clean_tensor, bboxes_jittered_tensor
            torch.cuda.empty_cache()
            print("GPU メモリをクリアしました")

        # 統計情報の計算
        bbox_stats = self._compute_bbox_statistics(bboxes_clean, bboxes_jittered)

        # 人手アノテーション品質の分析
        annotation_quality = self._analyze_annotation_quality(
            bboxes_clean, bboxes_jittered
        )

        results = {
            "validation_metrics": {
                "human_rough_vs_clean": jittered_vs_clean_metrics,  # 人手ラフ vs クリーン
                "clean_vs_human_rough": clean_vs_jittered_metrics,  # クリーン vs 人手ラフ
            },
            "bbox_statistics": bbox_stats,
            "annotation_quality": annotation_quality,
            "sample_count": len(common_indices),
            "dataset_config": {
                "center_jitter_ratio": config.CENTER_JITTER_RATIO,
                "scale_jitter_ratio": config.SCALE_JITTER_RATIO,
                "buffer_ratio": config.BUFFER_RATIO,
                "img_size": config.IMG_SIZE,
                "iou_threshold": config.IOU_THRESHOLD,
                "deviance_thresholds": config.DEVIANCE_THRESHOLDS,
            },
        }

        return results

    def _compute_bbox_statistics(self, bboxes_clean, bboxes_jittered):
        """バウンディングボックスの統計情報を計算"""
        stats = {}

        # 面積の統計
        areas_clean = [
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes_clean
        ]
        areas_jittered = [
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes_jittered
        ]

        stats["area"] = {
            "clean_annotation": {
                "mean": np.mean(areas_clean),
                "std": np.std(areas_clean),
                "min": np.min(areas_clean),
                "max": np.max(areas_clean),
            },
            "human_rough_annotation": {
                "mean": np.mean(areas_jittered),
                "std": np.std(areas_jittered),
                "min": np.min(areas_jittered),
                "max": np.max(areas_jittered),
            },
        }

        # 幅と高さの統計
        widths_clean = [bbox[2] - bbox[0] for bbox in bboxes_clean]
        heights_clean = [bbox[3] - bbox[1] for bbox in bboxes_clean]
        widths_jittered = [bbox[2] - bbox[0] for bbox in bboxes_jittered]
        heights_jittered = [bbox[3] - bbox[1] for bbox in bboxes_jittered]

        stats["width"] = {
            "clean_annotation": {
                "mean": np.mean(widths_clean),
                "std": np.std(widths_clean),
            },
            "human_rough_annotation": {
                "mean": np.mean(widths_jittered),
                "std": np.std(widths_jittered),
            },
        }

        stats["height"] = {
            "clean_annotation": {
                "mean": np.mean(heights_clean),
                "std": np.std(heights_clean),
            },
            "human_rough_annotation": {
                "mean": np.mean(heights_jittered),
                "std": np.std(heights_jittered),
            },
        }

        # 中心座標の統計
        centers_x_clean = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes_clean]
        centers_y_clean = [(bbox[1] + bbox[3]) / 2 for bbox in bboxes_clean]
        centers_x_jittered = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes_jittered]
        centers_y_jittered = [(bbox[1] + bbox[3]) / 2 for bbox in bboxes_jittered]

        stats["center"] = {
            "x": {
                "clean_annotation": {
                    "mean": np.mean(centers_x_clean),
                    "std": np.std(centers_x_clean),
                },
                "human_rough_annotation": {
                    "mean": np.mean(centers_x_jittered),
                    "std": np.std(centers_x_jittered),
                },
            },
            "y": {
                "clean_annotation": {
                    "mean": np.mean(centers_y_clean),
                    "std": np.std(centers_y_clean),
                },
                "human_rough_annotation": {
                    "mean": np.mean(centers_y_jittered),
                    "std": np.std(centers_y_jittered),
                },
            },
        }

        return stats

    def _analyze_annotation_quality(self, bboxes_clean, bboxes_jittered):
        """人手アノテーション品質の分析"""
        quality_metrics = {}

        # 位置偏差の分析
        center_deviations = []
        size_ratios = []

        for clean_bbox, jitter_bbox in zip(bboxes_clean, bboxes_jittered):
            # 中心座標の偏差
            clean_center = [
                (clean_bbox[0] + clean_bbox[2]) / 2,
                (clean_bbox[1] + clean_bbox[3]) / 2,
            ]
            jitter_center = [
                (jitter_bbox[0] + jitter_bbox[2]) / 2,
                (jitter_bbox[1] + jitter_bbox[3]) / 2,
            ]

            center_deviation = np.sqrt(
                (clean_center[0] - jitter_center[0]) ** 2
                + (clean_center[1] - jitter_center[1]) ** 2
            )
            center_deviations.append(center_deviation)

            # サイズ比の分析
            clean_area = (clean_bbox[2] - clean_bbox[0]) * (
                clean_bbox[3] - clean_bbox[1]
            )
            jitter_area = (jitter_bbox[2] - jitter_bbox[0]) * (
                jitter_bbox[3] - jitter_bbox[1]
            )

            if clean_area > 0:
                size_ratio = jitter_area / clean_area
                size_ratios.append(size_ratio)

        quality_metrics["center_deviation"] = {
            "mean": np.mean(center_deviations),
            "std": np.std(center_deviations),
            "median": np.median(center_deviations),
            "percentile_95": np.percentile(center_deviations, 95),
        }

        quality_metrics["size_ratio"] = {
            "mean": np.mean(size_ratios),
            "std": np.std(size_ratios),
            "median": np.median(size_ratios),
            "percentile_95": np.percentile(size_ratios, 95),
        }

        return quality_metrics

    def _convert_numpy_types(self, obj):
        """NumPy型をPython標準型に再帰的に変換"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_analysis_results(self, results, output_dir="validation_analysis_results"):
        """分析結果を保存"""
        os.makedirs(output_dir, exist_ok=True)

        # NumPy型をPython標準型に変換してからJSON保存
        serializable_results = self._convert_numpy_types(results)
        results_json = json.dumps(serializable_results, indent=2, ensure_ascii=False)
        with open(
            os.path.join(output_dir, "validation_analysis.json"), "w", encoding="utf-8"
        ) as f:
            f.write(results_json)

        # 可視化とレポート生成
        self._generate_report(results, output_dir)

        print(f"分析結果を {output_dir} に保存しました")

    def _generate_report(self, results, output_dir):
        """分析結果のレポートを生成"""
        # 評価指標の可視化
        self._plot_validation_metrics(results["validation_metrics"], output_dir)

        # 統計情報の可視化
        self._plot_bbox_statistics(results["bbox_statistics"], output_dir)

        # アノテーション品質の可視化
        self._plot_annotation_quality(results["annotation_quality"], output_dir)

        # テキストレポートの生成
        self._generate_text_report(results, output_dir)

    def _plot_validation_metrics(self, validation_metrics, output_dir):
        """検証評価指標の可視化"""
        human_rough_metrics = validation_metrics["human_rough_vs_clean"]
        clean_metrics = validation_metrics["clean_vs_human_rough"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("検証データセット評価指標比較", fontsize=16)

        # IoU指標
        iou_keys = [k for k in human_rough_metrics.keys() if "IoU" in k]
        if iou_keys:
            iou_data = {
                "人手ラフ vs クリーン": [human_rough_metrics[k] for k in iou_keys],
                "クリーン vs 人手ラフ": [clean_metrics[k] for k in iou_keys],
            }

            x = range(len(iou_keys))
            width = 0.35
            axes[0, 0].bar(
                [i - width / 2 for i in x],
                iou_data["人手ラフ vs クリーン"],
                width,
                label="人手ラフ vs クリーン",
                alpha=0.7,
            )
            axes[0, 0].bar(
                [i + width / 2 for i in x],
                iou_data["クリーン vs 人手ラフ"],
                width,
                label="クリーン vs 人手ラフ",
                alpha=0.7,
            )
            axes[0, 0].set_title("IoU系指標")
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(iou_keys, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Edge系指標
        edge_keys = [k for k in human_rough_metrics.keys() if "Edge" in k]
        if edge_keys:
            edge_data = {
                "人手ラフ vs クリーン": [human_rough_metrics[k] for k in edge_keys],
                "クリーン vs 人手ラフ": [clean_metrics[k] for k in edge_keys],
            }

            x = range(len(edge_keys))
            axes[0, 1].bar(
                [i - width / 2 for i in x],
                edge_data["人手ラフ vs クリーン"],
                width,
                label="人手ラフ vs クリーン",
                alpha=0.7,
            )
            axes[0, 1].bar(
                [i + width / 2 for i in x],
                edge_data["クリーン vs 人手ラフ"],
                width,
                label="クリーン vs 人手ラフ",
                alpha=0.7,
            )
            axes[0, 1].set_title("Edge精度指標")
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(edge_keys, rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Corner系指標
        corner_keys = [k for k in human_rough_metrics.keys() if "Corner" in k]
        if corner_keys:
            corner_data = {
                "人手ラフ vs クリーン": [human_rough_metrics[k] for k in corner_keys],
                "クリーン vs 人手ラフ": [clean_metrics[k] for k in corner_keys],
            }

            x = range(len(corner_keys))
            axes[1, 0].bar(
                [i - width / 2 for i in x],
                corner_data["人手ラフ vs クリーン"],
                width,
                label="人手ラフ vs クリーン",
                alpha=0.7,
            )
            axes[1, 0].bar(
                [i + width / 2 for i in x],
                corner_data["クリーン vs 人手ラフ"],
                width,
                label="クリーン vs 人手ラフ",
                alpha=0.7,
            )
            axes[1, 0].set_title("Corner精度指標")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(corner_keys, rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 全指標の平均値比較
        all_human_rough = [v for v in human_rough_metrics.values()]
        all_clean = [v for v in clean_metrics.values()]

        axes[1, 1].bar(
            ["人手ラフ vs クリーン", "クリーン vs 人手ラフ"],
            [np.mean(all_human_rough), np.mean(all_clean)],
            alpha=0.7,
            color=["lightcoral", "skyblue"],
        )
        axes[1, 1].set_title("全指標平均値")
        axes[1, 1].set_ylabel("平均スコア")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "validation_metrics.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_bbox_statistics(self, bbox_stats, output_dir):
        """バウンディングボックス統計の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("アノテーション統計比較", fontsize=16)

        # 面積比較
        area_clean = bbox_stats["area"]["clean_annotation"]
        area_jittered = bbox_stats["area"]["human_rough_annotation"]

        axes[0, 0].bar(
            ["クリーン", "人手ラフ"],
            [area_clean["mean"], area_jittered["mean"]],
            yerr=[area_clean["std"], area_jittered["std"]],
            capsize=5,
            alpha=0.7,
        )
        axes[0, 0].set_title("バウンディングボックス面積")
        axes[0, 0].set_ylabel("平均面積 (pixel²)")
        axes[0, 0].grid(True, alpha=0.3)

        # 幅比較
        width_clean = bbox_stats["width"]["clean_annotation"]
        width_jittered = bbox_stats["width"]["human_rough_annotation"]

        axes[0, 1].bar(
            ["クリーン", "人手ラフ"],
            [width_clean["mean"], width_jittered["mean"]],
            yerr=[width_clean["std"], width_jittered["std"]],
            capsize=5,
            alpha=0.7,
        )
        axes[0, 1].set_title("バウンディングボックス幅")
        axes[0, 1].set_ylabel("平均幅 (pixel)")
        axes[0, 1].grid(True, alpha=0.3)

        # 高さ比較
        height_clean = bbox_stats["height"]["clean_annotation"]
        height_jittered = bbox_stats["height"]["human_rough_annotation"]

        axes[0, 2].bar(
            ["クリーン", "人手ラフ"],
            [height_clean["mean"], height_jittered["mean"]],
            yerr=[height_clean["std"], height_jittered["std"]],
            capsize=5,
            alpha=0.7,
        )
        axes[0, 2].set_title("バウンディングボックス高さ")
        axes[0, 2].set_ylabel("平均高さ (pixel)")
        axes[0, 2].grid(True, alpha=0.3)

        # 中心座標X比較
        center_x_clean = bbox_stats["center"]["x"]["clean_annotation"]
        center_x_jittered = bbox_stats["center"]["x"]["human_rough_annotation"]

        axes[1, 0].bar(
            ["クリーン", "人手ラフ"],
            [center_x_clean["mean"], center_x_jittered["mean"]],
            yerr=[center_x_clean["std"], center_x_jittered["std"]],
            capsize=5,
            alpha=0.7,
        )
        axes[1, 0].set_title("中心座標X")
        axes[1, 0].set_ylabel("平均X座標 (pixel)")
        axes[1, 0].grid(True, alpha=0.3)

        # 中心座標Y比較
        center_y_clean = bbox_stats["center"]["y"]["clean_annotation"]
        center_y_jittered = bbox_stats["center"]["y"]["human_rough_annotation"]

        axes[1, 1].bar(
            ["クリーン", "人手ラフ"],
            [center_y_clean["mean"], center_y_jittered["mean"]],
            yerr=[center_y_clean["std"], center_y_jittered["std"]],
            capsize=5,
            alpha=0.7,
        )
        axes[1, 1].set_title("中心座標Y")
        axes[1, 1].set_ylabel("平均Y座標 (pixel)")
        axes[1, 1].grid(True, alpha=0.3)

        # 面積分布の比較
        axes[1, 2].bar(
            ["面積変化率"],
            [((area_jittered["mean"] - area_clean["mean"]) / area_clean["mean"]) * 100],
            alpha=0.7,
            color="orange",
        )
        axes[1, 2].set_title("面積変化率")
        axes[1, 2].set_ylabel("変化率 (%)")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "bbox_statistics.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_annotation_quality(self, annotation_quality, output_dir):
        """アノテーション品質の可視化"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("人手ラフアノテーション品質分析", fontsize=16)

        # 中心座標偏差
        center_dev = annotation_quality["center_deviation"]
        axes[0].bar(
            ["平均", "中央値", "95%tile"],
            [center_dev["mean"], center_dev["median"], center_dev["percentile_95"]],
            alpha=0.7,
            color=["skyblue", "lightgreen", "lightcoral"],
        )
        axes[0].set_title("中心座標偏差 (pixel)")
        axes[0].set_ylabel("偏差距離 (pixel)")
        axes[0].grid(True, alpha=0.3)

        # サイズ比
        size_ratio = annotation_quality["size_ratio"]
        axes[1].bar(
            ["平均", "中央値", "95%tile"],
            [size_ratio["mean"], size_ratio["median"], size_ratio["percentile_95"]],
            alpha=0.7,
            color=["skyblue", "lightgreen", "lightcoral"],
        )
        axes[1].set_title("サイズ比（人手ラフ/クリーン）")
        axes[1].set_ylabel("比率")
        axes[1].axhline(
            y=1.0, color="red", linestyle="--", alpha=0.7, label="基準線(1.0)"
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "annotation_quality.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _generate_text_report(self, results, output_dir):
        """テキスト形式のレポートを生成"""
        report_path = os.path.join(output_dir, "validation_analysis_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("検証データセット評価指標分析レポート\n")
            f.write("=" * 80 + "\n\n")

            # サマリー情報
            f.write("1. 分析概要\n")
            f.write("-" * 40 + "\n")
            f.write(f"分析対象サンプル数: {results['sample_count']}\n")
            f.write(
                f"画像サイズ: {results['dataset_config']['img_size']}x{results['dataset_config']['img_size']}\n"
            )
            f.write(
                f"Center Jitter Ratio: {results['dataset_config']['center_jitter_ratio']}\n"
            )
            f.write(
                f"Scale Jitter Ratio: {results['dataset_config']['scale_jitter_ratio']}\n"
            )
            f.write(f"Buffer Ratio: {results['dataset_config']['buffer_ratio']}\n\n")

            # 検証評価指標
            f.write("2. 検証時評価指標\n")
            f.write("-" * 40 + "\n")

            human_rough_metrics = results["validation_metrics"]["human_rough_vs_clean"]
            clean_metrics = results["validation_metrics"]["clean_vs_human_rough"]

            f.write("2.1 人手ラフアノテーション vs クリーンアノテーション\n")
            f.write(
                "（人手ラフアノテーションを予測、クリーンアノテーションをGTとして評価）\n"
            )
            for metric, value in human_rough_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")

            f.write("2.2 クリーンアノテーション vs 人手ラフアノテーション（参考）\n")
            for metric, value in clean_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")

            # アノテーション品質分析
            f.write("3. 人手ラフアノテーション品質\n")
            f.write("-" * 40 + "\n")

            center_dev = results["annotation_quality"]["center_deviation"]
            size_ratio = results["annotation_quality"]["size_ratio"]

            f.write("3.1 中心座標偏差\n")
            f.write(f"  平均偏差: {center_dev['mean']:.2f} pixel\n")
            f.write(f"  中央値: {center_dev['median']:.2f} pixel\n")
            f.write(f"  95%tile: {center_dev['percentile_95']:.2f} pixel\n\n")

            f.write("3.2 サイズ比（人手ラフ/クリーン）\n")
            f.write(f"  平均比: {size_ratio['mean']:.3f}\n")
            f.write(f"  中央値: {size_ratio['median']:.3f}\n")
            f.write(f"  95%tile: {size_ratio['percentile_95']:.3f}\n\n")

            # 統計情報
            f.write("4. バウンディングボックス統計\n")
            f.write("-" * 40 + "\n")
            bbox_stats = results["bbox_statistics"]

            area_clean = bbox_stats["area"]["clean_annotation"]
            area_jittered = bbox_stats["area"]["human_rough_annotation"]

            f.write("4.1 面積統計\n")
            f.write(
                f"  クリーン: 平均={area_clean['mean']:.2f}, 標準偏差={area_clean['std']:.2f}\n"
            )
            f.write(
                f"  人手ラフ: 平均={area_jittered['mean']:.2f}, 標準偏差={area_jittered['std']:.2f}\n"
            )
            f.write(
                f"  変化率: {((area_jittered['mean'] - area_clean['mean']) / area_clean['mean'] * 100):.2f}%\n\n"
            )

            # 解釈
            f.write("5. 結果の解釈\n")
            f.write("-" * 40 + "\n")
            f.write("この分析は検証時に得られる評価指標を示しています。\n")
            f.write(
                "- 人手ラフアノテーションは、jitteringによってシミュレートされた人手の不正確さを含みます\n"
            )
            f.write("- IoU系指標: バウンディングボックスの重複度\n")
            f.write("- Edge系指標: 各辺の位置精度\n")
            f.write("- Corner系指標: 角点の位置精度\n\n")

            avg_human_rough = np.mean(list(human_rough_metrics.values()))
            f.write(
                f"人手ラフアノテーションの全体的な品質スコア: {avg_human_rough:.4f}\n"
            )

            if avg_human_rough > 0.7:
                f.write("人手ラフアノテーションでも比較的高い精度を保っています。\n")
            elif avg_human_rough > 0.5:
                f.write("人手ラフアノテーションは中程度の精度です。\n")
            else:
                f.write("人手ラフアノテーションの精度は低めです。\n")

            f.write(
                "\nこの結果は、実際の人手アノテーションでの検証時に期待される性能の参考になります。\n"
            )


def main():
    """メイン実行関数"""
    print("検証データセット評価指標分析ツールを実行します...")

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 検証用データセットのパスを設定
    annotations_path = config.COCO_ANNOTATIONS_PATH_VAL
    img_dir = config.COCO_IMG_DIR_VAL

    # パスの存在確認
    if not os.path.exists(annotations_path):
        print(f"エラー: アノテーションファイルが見つかりません: {annotations_path}")
        return

    if not os.path.exists(img_dir):
        print(f"エラー: 画像ディレクトリが見つかりません: {img_dir}")
        return

    # 分析器初期化
    analyzer = ValidationDatasetAnalyzer(annotations_path, img_dir, device)

    # 分析実行
    try:
        # 検証データセット全体を使用
        results = analyzer.analyze_validation_metrics(
            use_full_dataset=True, batch_size=32
        )

        # 結果保存
        output_dir = "src/analysis_tools/validation_analysis_results"
        analyzer.save_analysis_results(results, output_dir)

        print("\n分析完了！")
        print(f"結果は {output_dir} に保存されました。")

        # 主要な結果を表示
        print("\n=== 検証時評価指標（人手ラフ vs クリーン） ===")
        human_rough_metrics = results["validation_metrics"]["human_rough_vs_clean"]
        for metric, value in human_rough_metrics.items():
            print(f"{metric}: {value:.4f}")

    except Exception as e:
        print(f"分析中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
