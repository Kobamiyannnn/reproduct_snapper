# Bounding Box Adjustment Model (Snapper再現プロジェクト)

このプロジェクトは、論文「[Snapper: Accelerating Bounding Box Annotation with A Generic Object-Agnostic Bounding Box Adjustment Model](/resources/snapper-accelerating-bounding-box-annotation-in-object-detection-tasks-with-find-and-snap-tooling.pdf)」に登場する Bounding Box Adjustment Model の再現実装を目指すものです。

![](/resources/figure-3_snapper-bounding-box-adjustment-model-design.png)

## 概要

ユーザーによってラフにアノテーションされたバウンディングボックスを、対象物体に精密に調整するオブジェクトクラスに依存しない (object-agnostic) モデルです。

## モデルアーキテクチャ

モデルの主要な構成要素は以下の通りです。

1.  **入力**:
    *   単一の画像パッチ。
    *   対象物体1つに対するラフなバウンディングボックス座標 (x_min, y_min, x_max, y_max)。

2.  **出力**:
    *   入力されたバウンディングボックスを対象物体に精密に調整した4つの辺の座標。具体的には、各辺が画像パッチのどの位置に対応するかの確率分布（ロジット）。

3.  **基本構造**:
    *   **バックボーン**: ResNet-50 (ImageNetで事前学習済み、最終的な全結合層は除く) を使用し、入力画像パッチから特徴マップを抽出します。
    *   **Multi-Scale Feature Map**: ResNet-50の異なる層から特徴マップを抽出し、マルチスケール情報を活用します。具体的には、`layer3` (出力チャネル1024, 入力画像に対しH/16, W/16解像度) と `layer4` (出力チャネル2048, H/32, W/32解像度) の出力を利用します。
    *   **Directional Spatial Pooling**:
        *   `layer3` および `layer4` から抽出された各特徴マップに対し、独立に以下のプーリング処理を適用します。
        *   **Vertical Pooling**: 特徴マップの幅次元で平均プーリングを行い、高さ方向の1次元特徴ベクトル列を各スケールで生成します。これは水平な辺 (top, bottom) の予測に使用されます。
        *   **Horizontal Pooling**: 特徴マップの高さ次元で平均プーリングを行い、幅方向の1次元特徴ベクトル列を各スケールで生成します。これは垂直な辺 (left, right) の予測に使用されます。
    *   **Decoders (辺予測器)**:
        *   各スケール(`layer3`, `layer4`)のプーリング済み特徴ベクトルは、それぞれ対応する辺（top, bottom, left, right）を予測するための独立したデコーダーに入力されます。
        *   現在の実装では、これらのデコーダーは線形層 (`nn.Linear`) で構成されています。例えば `layer3` からの垂直プーリング特徴は、`Linear(1024, IMAGE_SIZE)` という層に入力され、画像パッチの高さ次元の各ピクセル位置に対する辺の存在確率（ロジット）を直接予測します。同様に `layer4` からの特徴は `Linear(2048, IMAGE_SIZE)` に入力されます。
        *   これらのデコーダーは、論文で言及されている1D CNNベースのデコーダーから、現在の線形層ベースの実装に変更されています。各デコーダーからの出力は、特徴マップの空間次元で平均化された後、最終的なロジットベクトルとなります。
    *   **出力統合と最終的な座標決定**:
        *   `layer3` 由来のデコーダー群からの予測ロジットと、`layer4` 由来のデコーダー群からの予測ロジットは、辺ごとに単純平均されて統合されます。
        *   この統合された各辺の確率分布（ロジット）に対して `argmax` を適用することで、最も確率の高いピクセル位置を辺の座標として決定します。明示的なアップサンプリング処理は現在のモデルでは行われていません。

## 入力処理

1.  **Buffer Ratio**: ユーザーが提供した初期のラフなバウンディングボックスに対し、`Buffer Ratio` を適用してクロップ領域を決定します。この比率は、ラフなボックスの幅と高さにそれぞれ乗算され、クロップする領域を広げたり狭めたりします。現在のプロジェクト設定では、この値は `src/config.py` 内で `BUFFER_RATIO = 1.1` と定義されています。
2.  **クロッピング**: 上記で決定されたクロップ領域に基づいて元画像から画像パッチを切り出し、モデルの固定入力サイズ (例: `src/config.py` で定義された `IMG_SIZE`x`IMG_SIZE`、通常224x224ピクセル) にリサイズします。このリサイズされた画像パッチがモデルへの最終的な入力となります。

## 損失関数

*   各辺（top, bottom, left, right）の予測された確率分布（ロジット）と、正解の辺の位置から作成されたターゲットベクトル（正解の辺の位置が1で他が0のone-hotベクトル）との間で、Binary Cross-Entropy (具体的には `torch.nn.BCEWithLogitsLoss`) を計算します。
*   4つの辺に対する損失値を合計したものが、最終的な損失となります。

## 学習データとJittering

*   **データセット**: MS COCOデータセットを使用します。
*   **Jittering (データ拡張)**: 学習時には、正解のバウンディングボックスに対して以下のJittering処理を適用し、モデルへの入力となるラフなバウンディングボックスを人工的に生成します。
    1.  **中心のずらし**: 正解バウンディングボックスの中心座標を、各軸のバウンディングボックス寸法の最大10%の範囲でランダムにずらします。
    2.  **スケールの変更**: 正解バウンディングボックスの幅と高さを、それぞれ0.9倍から1.1倍のランダムな比率でリスケールします。
*   **クロップと座標変換**: Jitteringによって生成されたラフなバウンディングボックスと `Buffer Ratio` を用いて元画像をクロップし、モデル入力画像とします。損失計算に使用する正解のバウンディングボックス座標も、このクロップおよびリサイズ後の座標系に変換されます。必要に応じて、座標はクロップ領域の境界でクリッピングされます。

## 評価指標

モデルの性能は以下の指標で評価され、TensorBoardで確認できます。

*   **IoU (Intersection over Union)**: 予測されたバウンディングボックスと正解のバウンディングボックスの重なり具合。
*   **Edge Deviance**: 予測された各辺と正解の各辺のピクセル単位での平均的なずれ。また、ずれが特定の閾値（例: 1ピクセル、3ピクセル）以下である割合。
*   **Corner Deviance**: 予測された各頂点と正解の各頂点のピクセル単位での平均的なL1距離。また、L1距離が特定の閾値（例: 1ピクセル、3ピクセル）以下である割合。
*   **学習/検証損失**: 各エポックにおける学習損失と検証損失。

## プロジェクト構成

```
reproduct_snapper/
├── checkpoints/        # 学習済みモデルのチェックポイント
├── data/               # データセット (例: COCO)
│   └── coco/
│       ├── annotations/  # COCOアノテーションファイル
│       ├── train2017/    # COCO学習用画像
│       └── val2017/      # COCO検証用画像
├── resources/          # README用の画像や論文PDFなど
├── results/            # 推論結果の出力画像などを保存 (オプション)
├── runs/               # TensorBoard のログファイル
├── src/                # ソースコード
│   ├── config.py       # 設定ファイル (学習率、パス、LOG_DIRなど)
│   ├── dataset.py      # データセットクラス (CocoAdjustmentDataset)
│   ├── inference.py    # 推論実行スクリプト
│   ├── models.py       # モデル定義 (BoundingBoxAdjustmentModel)
│   ├── train.py        # 学習・検証スクリプト
│   └── utils.py        # ユーティリティ関数 (Jittering, 評価指標など)
├── .gitignore
├── pyproject.toml      # Poetry / uv 用の設定ファイル
├── README.md           # このファイル
├── requirements.txt    # 依存パッケージリスト (pip用)
└── uv.lock             # uv 用のロックファイル
```

## セットアップと実行

1.  **依存関係のインストール**:
    ```bash
    pip install -r requirements.txt
    # または uv を使用する場合
    # uv pip install -r requirements.txt
    ```
    `pycocotools` のインストールには、環境に応じて追加のステップが必要になる場合があります。
    TensorBoardを使用するために `tensorboard` パッケージもインストールされていることを確認してください。
    ```bash
    pip install tensorboard
    ```

2.  **データセットの準備**:
    *   MS COCOデータセットをダウンロードし、`data/coco` ディレクトリ以下に配置します (上記ディレクトリ構成参照)。
    *   `src/config.py` 内のデータセットパス (`COCO_ANNOTATIONS_PATH_TRAIN`, `COCO_IMG_DIR_TRAIN` など) を適切に設定します。

3.  **学習の実行と監視**:
    *   まず、別のターミナルでTensorBoardを起動します。
        ```bash
        tensorboard --logdir runs
        ```
        （`runs` の部分は `src/config.py` の `LOG_DIR` で指定したディレクトリ名に合わせてください。）
        その後、ブラウザで表示されたURL（通常は `http://localhost:6006/`）にアクセスします。

    *   学習スクリプトを実行します。
        ```bash
        python src/train.py
        # または uv を使用する場合
        # uv run src/train.py
        ```
    学習設定は `src/config.py` で調整可能です。学習の進捗（損失、IoUなど）はTensorBoardでリアルタイムに確認できます。

4.  **推論の実行 (学習済みモデルを使用)**:
    学習済みのモデルを使用して、単一の画像に対してバウンディングボックス調整を行うには `src/inference.py` を使用します。

    **コマンド例**:
    ```bash
    python src/inference.py \
        --image_path "path/to/your/image.jpg" \
        --bbox "x_min,y_min,width,height" \
        --model_path "checkpoints/your_trained_model.pth" \
        --output_path "results/output_image.jpg"
    ```

    **引数の説明**:
    *   `--image_path`: 入力画像のパス。
    *   `--bbox`: ラフなバウンディングボックスの座標を `x_min,y_min,width,height` の形式で指定 (カンマ区切り、スペースなし)。
    *   `--model_path`: 学習済みモデルのチェックポイントファイル (`.pth`) のパス。
    *   `--output_path` (オプション): 調整後のバウンディングボックスが描画された画像の保存先。指定しない場合はウィンドウに表示されます。
    *   `--config_path` (オプション): デフォルト以外の設定ファイルを使用する場合に指定します (現在はデフォルト設定を使用)。

    `results/` ディレクトリは、推論結果の画像を保存するために使用できます。必要に応じて作成してください。
