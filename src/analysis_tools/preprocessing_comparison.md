# 前処理効果分析ツール

## 概要

このツールは、前処理実施前後で検証用データセットがどのように変化したのかを確認するための分析ツールです。モデルを使用せずに、前処理（jittering）の効果をIoU系指標とDeviation系指標で評価します。

## 機能

- **前処理の効果分析**: jitteringありとなしのデータセットを比較
- **IoU系指標**: バウンディングボックスの重複度を測定（mIoU, IoU > 閾値）
- **Deviation系指標**: Edge精度とCorner精度を測定
- **統計情報**: バウンディングボックスの面積、サイズ、中心座標の統計
- **可視化**: グラフとレポートによる結果の可視化

## ファイル構成

```
src/analysis_tools/
├── preprocessing_comparison.py    # メイン分析ロジック
├── run_preprocessing_analysis.py  # 実行スクリプト
└── README.md                     # このファイル
```

## 使用方法

### 基本的な実行

```bash
cd src/analysis_tools
python run_preprocessing_analysis.py
```

### パラメータ指定での実行

```bash
python run_preprocessing_analysis.py \
    --num-samples 1000 \
    --batch-size 64 \
    --output-dir custom_output \
    --device cuda \
    --verbose
```

### コマンドライン引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--annotations` | config.pyの設定 | COCOアノテーションファイルのパス |
| `--images` | config.pyの設定 | 画像ディレクトリのパス |
| `--num-samples` | 500 | 分析するサンプル数 |
| `--batch-size` | 32 | バッチサイズ |
| `--output-dir` | `preprocessing_analysis_results` | 結果出力ディレクトリ |
| `--device` | auto | 使用デバイス (auto/cpu/cuda) |
| `--verbose` | False | 詳細ログを出力 |

## 出力結果

分析実行後、指定した出力ディレクトリに以下のファイルが生成されます：

### 1. JSONレポート (`preprocessing_analysis.json`)
- 全ての分析結果を構造化データとして保存
- プログラムでの後続処理に利用可能

### 2. テキストレポート (`preprocessing_analysis_report.txt`)
- 人間が読みやすい形式での分析結果
- 解釈とコメント付き

### 3. メトリクス比較グラフ (`metrics_comparison.png`)
- IoU系、Edge系、Corner系指標の比較可視化
- 前処理ありとなしの双方向比較

### 4. 統計比較グラフ (`bbox_statistics.png`)
- バウンディングボックスの統計情報の可視化
- 面積、サイズ、中心座標の比較

## 分析内容の詳細

### 比較方法

1. **前処理ありデータセット**: `for_training=True`（jitteringを適用）
2. **前処理なしデータセット**: `for_training=False`（jitteringを適用しない）

### 評価指標

#### IoU系指標
- **mIoU**: 平均IoU値
- **IoU > 閾値**: 指定した閾値以上のIoUを持つサンプルの割合

#### Deviation系指標
- **Edge < Xpx**: 各辺の位置が指定ピクセル以内の精度
- **Corner < Xpx**: 各角点の位置が指定ピクセル以内の精度

#### 統計情報
- バウンディングボックスの面積、幅、高さ
- 中心座標の分布
- 変化率の計算

### 分析の意味

この分析により以下のことが理解できます：

1. **前処理の影響度**: jitteringがデータにどの程度の変化を与えるか
2. **データ品質の保持**: 前処理後もデータの品質が保たれているか
3. **設定の妥当性**: jitteringパラメータが適切に設定されているか

## 設定パラメータ

分析では `config.py` の以下の設定が使用されます：

- `CENTER_JITTER_RATIO`: 中心座標のjitter比率
- `SCALE_JITTER_RATIO`: スケールのjitter比率
- `BUFFER_RATIO`: クロッピング時のバッファ比率
- `IOU_THRESHOLD`: IoU評価の閾値
- `DEVIANCE_THRESHOLDS`: Deviation評価の閾値

## 実行例

```bash
# 基本実行
python run_preprocessing_analysis.py

# より多くのサンプルで詳細分析
python run_preprocessing_analysis.py --num-samples 2000 --verbose

# CPU指定での実行
python run_preprocessing_analysis.py --device cpu

# カスタム出力ディレクトリ指定
python run_preprocessing_analysis.py --output-dir my_analysis_results
```

## 注意事項

1. **データセットパス**: config.pyでCOCOデータセットのパスが正しく設定されている必要があります
2. **メモリ使用量**: サンプル数を増やすとメモリ使用量も増加します
3. **実行時間**: GPUを使用することで分析時間を短縮できます
4. **依存関係**: matplotlib, seaborn, tqdm, torch, numpy等が必要です

## トラブルシューティング

### よくあるエラー

1. **ファイルが見つからない**
   ```
   エラー: アノテーションファイルが見つかりません
   ```
   → config.pyのパス設定を確認してください

2. **メモリ不足**
   ```
   CUDA out of memory
   ```
   → `--num-samples` を減らすか `--device cpu` を使用してください

3. **インポートエラー**
   ```
   ModuleNotFoundError
   ```
   → 必要なパッケージをインストールしてください：
   ```bash
   pip install torch matplotlib seaborn tqdm numpy
   ```

## 分析結果の解釈

### 良好な結果の例
- mIoU > 0.8: 前処理による変化が小さい
- Edge < 1px > 0.9: エッジ精度が高い
- Corner < 3px > 0.8: コーナー精度が高い

### 要注意な結果の例
- mIoU < 0.6: 前処理による変化が大きすぎる
- 大きな統計値の変化: 面積やサイズの大幅な変動

このような場合は、config.pyのjitteringパラメータの調整を検討してください。 
