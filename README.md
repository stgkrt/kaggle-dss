# Child Mind Institute - Detect Sleep States
[kaggle competition](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states) 

# directory
    ├── input/              <- Competition Datasets.  
    ├── notebooks/          <- Jupyter notebooks.  
    ├── scripts/            <- Scripts.  
    ├── src/                <- Source code. This sould be Python module.  
    ├── working/            <- Output models and train logs.  
    │  
    ├── .dockerignore  
    ├── .gitignore  
    └── README.md           <- The top-level README for developers.  

# working

### ベースライン
- 1DCNNのencoder/decoderモデル
- 基本的にはいっぱい検出した方がmetricだと理解している
    - Nanがあるところは誤検出にならないようにする必要がある
- 寝ている/起きているの分類モデル
    - イベント検出ではNanの扱いが難しく分類の方が簡単そうに見えた
    - ラベルはNanではないイベントを基準に寝ている/起きているを作る
    - Nanが基準になるとこは-1という値にしておいてlossで無視するようにする
    - (単純にイベント検出でモデル作ると分類より高いスコアができなかった。ラベルの作り方の問題かも？)


### モデルの検討
- 分類の層のあとにイベント検出の層を作る(2nd stageにするのもあり)→うまくいかなかった
- ダウンサンプリング(ラベルが1minごとなので1min(12stepごとの出力にする))→あまりかわらない
- エンコーダの手前で複数のカーネルサイズのconvを入れてからカーネルサイズ1で圧縮してencoder入力→少し改善した
- Nanがラベルのところを検出しすぎるとスコアが下がるので、Nanになってそうかどうかの判定chの増やす→いまやっている

### データの検討
- ラベルがずれているだろうっていうデータを除いて学習する→少し改善
- augmentationがないので、flipとrollを入れてみる→flipは少し改善、rollは少し改悪
- 発生時刻(分)の分布の偏りが大きいのでその分での検出になるように後処理→単純にそれでは流石にダメ

### やりたいけどまだできていない
- 1DCNNアンサンブル
- infer時にスライドして予測
- 変化点検出のアルゴリズムを入力
- フィルター(ハイパス・ローパス・バンドパス・カルマンスムーザなど)
- upsampleをconv_transposeに変更
- pseudo label(Nanのデータを学習させる。別で誤検出させない方法が必要)
