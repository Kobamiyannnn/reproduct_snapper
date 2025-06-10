# Resume Training Usage Examples

## 概要
`resume_train.py`は、フリーズやクラッシュなどでトレーニングが中断された場合に、チェックポイントファイルから学習を再開するためのスクリプトです。

## 主な機能
1. **チェックポイントからの復元**: モデル、オプティマイザー、スケジューラーの状態を復元
2. **TensorBoardログの継続**: 既存のログディレクトリを自動検出して継続記録
3. **Resume情報の記録**: 再開の詳細を`resume_info.md`に記録

## 使用方法

### 基本的な使用方法
```bash
# 最も簡単な方法（自動でTensorBoardログディレクトリを検出）
python src/resume_train.py --checkpoint checkpoints/run_20250610-123857/model_epoch_10.pth

# または best model から再開
python src/resume_train.py --checkpoint checkpoints/run_20250610-123857/best_model_iou.pth
```

### TensorBoardログディレクトリを明示的に指定
```bash
# 既存のログディレクトリを指定
python src/resume_train.py --checkpoint checkpoints/run_20250610-123857/model_epoch_10.pth --log_dir runs/run_20250610-123857

# 新しいログディレクトリで再開（完全に新しい実験として扱いたい場合）
python src/resume_train.py --checkpoint checkpoints/run_20250610-123857/model_epoch_10.pth --log_dir runs/new_run_20250611-090000
```

## ファイル構造の例

### トレーニング前
```
├── checkpoints/
├── runs/
├── src/
│   ├── train.py
│   ├── resume_train.py
│   └── ...
```

### 通常のトレーニング後
```
├── checkpoints/
│   └── run_20250610-123857/
│       ├── model_epoch_5.pth
│       ├── model_epoch_10.pth
│       └── best_model_iou.pth
├── runs/
│   └── run_20250610-123857/
│       ├── events.out.tfevents.*
│       └── experiment_conditions.md
```

### Resume後
```
├── checkpoints/
│   └── run_20250610-123857/
│       ├── model_epoch_5.pth
│       ├── model_epoch_10.pth
│       ├── model_epoch_15.pth       # 新しく追加
│       ├── model_epoch_20.pth       # 新しく追加
│       └── best_model_iou.pth       # 更新される可能性
├── runs/
│   └── run_20250610-123857/
│       ├── events.out.tfevents.*    # 継続して追記
│       ├── experiment_conditions.md
│       └── resume_info.md           # 新しく作成
```

## TensorBoardでの確認方法
```bash
# Resume前後のログを連続して確認
tensorboard --logdir runs/run_20250610-123857

# 複数の実験を比較
tensorboard --logdir runs/
```

## 注意事項
1. **config.pyの整合性**: Resume時も同じ`config.py`の設定を使用してください
2. **エポック数**: `config.EPOCHS`がチェックポイントのエポック数より大きい必要があります
3. **データセットパス**: 元のトレーニング時と同じデータセットパスが必要です
4. **依存関係**: Resume前後で同じライブラリバージョンを使用することを推奨します

## トラブルシューティング

### チェックポイントファイルが見つからない場合
```bash
# チェックポイントファイルの場所を確認
find . -name "*.pth" -type f

# 特定のディレクトリ内のチェックポイントを確認
ls -la checkpoints/*/
```

### TensorBoardログが見つからない場合
```bash
# ログディレクトリの確認
ls -la runs/

# 特定のログディレクトリの内容確認
ls -la runs/run_*/
```

### エラーが発生した場合
1. GPUメモリをクリア: `nvidia-smi` でGPU使用状況を確認
2. PyTorchのバージョン確認: チェックポイント作成時と同じバージョンか確認
3. ログファイルの確認: TensorBoardログディレクトリのアクセス権限を確認 
