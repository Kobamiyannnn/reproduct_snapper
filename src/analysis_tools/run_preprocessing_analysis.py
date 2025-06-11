#!/usr/bin/env python3
# src/analysis_tools/run_preprocessing_analysis.py
"""
検証データセット評価指標分析ツールのエントリーポイント
コマンドライン引数で様々なパラメータを指定可能
"""

import argparse
import os
import sys

# 親ディレクトリからのインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing_comparison import ValidationDatasetAnalyzer
import config


def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="検証データセットでの評価指標を分析し、人手ラフアノテーション再現の効果を測定するツール"
    )

    parser.add_argument(
        "--annotations",
        type=str,
        default=config.COCO_ANNOTATIONS_PATH_VAL,
        help="COCOアノテーションファイルのパス (default: config.pyの設定)",
    )

    parser.add_argument(
        "--images",
        type=str,
        default=config.COCO_IMG_DIR_VAL,
        help="画像ディレクトリのパス (default: config.pyの設定)",
    )

    parser.add_argument(
        "--use-full-dataset",
        action="store_true",
        default=True,
        help="検証データセット全体を使用 (default: True)",
    )

    parser.add_argument(
        "--test-mode", action="store_true", help="テストモード（少数サンプルで実行）"
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="バッチサイズ (default: 32)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/analysis_tools/validation_analysis_results",
        help="結果出力ディレクトリ (default: src/analysis_tools/validation_analysis_results)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="使用デバイス (default: auto)",
    )

    parser.add_argument("--verbose", action="store_true", help="詳細ログを出力")

    return parser.parse_args()


def setup_device(device_choice):
    """デバイス設定"""
    import torch

    if device_choice == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_choice)

    return device


def validate_paths(annotations_path, images_path):
    """パスの存在確認"""
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(
            f"アノテーションファイルが見つかりません: {annotations_path}"
        )

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"画像ディレクトリが見つかりません: {images_path}")


def main():
    """メイン実行関数"""
    args = parse_arguments()

    # ロギング設定
    if args.verbose:
        import logging

        logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("検証データセット評価指標分析ツール")
    print("=" * 60)
    print()

    # パラメータ表示
    print("実行パラメータ:")
    print(f"  アノテーションファイル: {args.annotations}")
    print(f"  画像ディレクトリ: {args.images}")

    if args.test_mode:
        print("  実行モード: テストモード（少数サンプル）")
        use_full_dataset = False
    else:
        print("  実行モード: 全データセット分析")
        use_full_dataset = args.use_full_dataset

    print(f"  バッチサイズ: {args.batch_size}")
    print(f"  出力ディレクトリ: {args.output_dir}")
    print(f"  デバイス指定: {args.device}")
    print()

    try:
        # パス確認
        validate_paths(args.annotations, args.images)

        # デバイス設定
        device = setup_device(args.device)
        print(f"使用デバイス: {device}")
        print()

        # 分析器初期化
        analyzer = ValidationDatasetAnalyzer(
            annotations_path=args.annotations, img_dir=args.images, device=device
        )

        # 分析実行
        print("検証データセット評価指標分析を開始します...")
        print("この分析では以下を実行します：")
        print("  1. クリーンアノテーション（前処理なし）での評価指標取得")
        print("  2. 人手ラフアノテーション再現（jittering）での評価指標取得")
        print("  3. 両者の比較によるアノテーション品質分析")
        print()

        results = analyzer.analyze_validation_metrics(
            use_full_dataset=use_full_dataset, batch_size=args.batch_size
        )

        # 結果保存
        analyzer.save_analysis_results(results, args.output_dir)

        print()
        print("=" * 60)
        print("分析完了！")
        print("=" * 60)
        print(f"結果は以下のディレクトリに保存されました: {args.output_dir}")
        print()

        # 主要結果の表示
        print("主要な検証評価指標（人手ラフ vs クリーン）:")
        human_rough_metrics = results["validation_metrics"]["human_rough_vs_clean"]
        for metric, value in human_rough_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print()
        print("詳細な分析結果は以下のファイルをご確認ください:")
        print(f"  - JSONレポート: {args.output_dir}/validation_analysis.json")
        print(f"  - テキストレポート: {args.output_dir}/validation_analysis_report.txt")
        print(f"  - グラフ (評価指標): {args.output_dir}/validation_metrics.png")
        print(f"  - グラフ (統計比較): {args.output_dir}/bbox_statistics.png")
        print(f"  - グラフ (品質分析): {args.output_dir}/annotation_quality.png")

        # アノテーション品質の概要
        quality = results["annotation_quality"]
        print()
        print("人手ラフアノテーション品質概要:")
        print(
            f"  中心座標偏差（平均）: {quality['center_deviation']['mean']:.2f} pixel"
        )
        print(f"  サイズ比（平均）: {quality['size_ratio']['mean']:.3f}")

        # 簡単な解釈
        avg_score = sum(human_rough_metrics.values()) / len(human_rough_metrics)
        print()
        print("解釈:")
        print(f"  人手ラフアノテーションの総合品質スコア: {avg_score:.4f}")

        if avg_score > 0.7:
            print("  → 人手ラフアノテーションでも比較的高い精度を保っています。")
        elif avg_score > 0.5:
            print("  → 人手ラフアノテーションは中程度の精度です。")
        else:
            print("  → 人手ラフアノテーションの精度は低めです。")

        print()
        print(
            "この結果は実際の人手アノテーションでの検証時に期待される性能の参考になります。"
        )

    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("パスの設定を確認してください。")
        sys.exit(1)

    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
