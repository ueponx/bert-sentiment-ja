# bert-sentiment-ja

日本語テキストの感情（ポジティブ/ネガティブ）を分析するPythonのサンプルプログラムです。

※ 本ドキュメントはLLMによって生成されています。
内容の詳細に関しては以下のリンクのブログ記事をご参照ください。

https://uepon.hatenadiary.com/entry/2025/12/28/142626


## 特徴

- 🤖 日本語BERTモデルを使用した高精度な感情分析
- 📊 信頼度スコアと確率分布の表示
- 🔄 バッチ処理対応
- 💬 インタラクティブモード搭載
- 🚀 GPU/CPU自動切り替え
- ⚡ uvによる高速な依存関係管理

## uvを使う利点

- **高速**: pipの10〜100倍の速度でパッケージをインストール
- **信頼性**: 依存関係の解決が確実
- **シンプル**: 仮想環境の作成とパッケージ管理が一体化
- **互換性**: pipのコマンドもそのまま使用可能

## 必要な環境

- Python 3.8以上
- uv（高速なPythonパッケージマネージャー）

## インストール

### 1. uvのインストール（未インストールの場合）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. プロジェクトのセットアップ

#### CPU版（GPU未搭載・軽量環境の場合）

GPUがない環境や、ストレージを節約したい場合に推奨。

**容量目安**: 約1〜2GB

```bash
# 仮想環境の作成
uv venv
source .venv/bin/activate

# CPU版PyTorch（軽量）+ 依存関係
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers fugashi ipadic unidic-lite
```

#### GPU版（NVIDIA GPU搭載の場合）

高速な処理が可能ですが、容量が大きくなります。

**容量目安**: 約5〜7GB

```bash
# 仮想環境の作成
uv venv
source .venv/bin/activate

# GPU版PyTorch + 依存関係
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers fugashi ipadic unidic-lite
```

#### どちらを選ぶべき？

| 条件 | 推奨 |
|------|------|
| NVIDIA GPU搭載 & CUDA環境あり | GPU版 |
| GPU未搭載 / ストレージ節約したい | CPU版 |
| 実験・学習目的 | CPU版 |
| 大量のテキストを高速処理したい | GPU版 |

> **注意**: CPU版でも動作に問題はありません。処理速度が若干遅くなるだけです。

### 3. 環境の確認（必要があれば）

```bash
# GPUが認識されているか確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. 実験後の環境削除

```bash
deactivate
rm -rf .venv
uv cache clean  # キャッシュも削除したい場合
```

## 使い方

### 基本的な実行

```bash
# 仮想環境を有効化している場合
python bert_sentiment_analysis.py
```

実行すると、以下の動作をします。

1. サンプルテキストで感情分析のデモ
2. インタラクティブモードで任意のテキストを分析可能

### プログラム内での使用する例

```python
from bert_sentiment_analysis import BERTSentimentAnalyzer

# 分析器の初期化
analyzer = BERTSentimentAnalyzer()

# 単一テキストの分析
result = analyzer.predict("この商品は素晴らしい！")
print(result)
# 出力例:
# {
#     'text': 'この商品は素晴らしい！',
#     'sentiment': 'ポジティブ',
#     'confidence': 0.98,
#     'probabilities': {
#         'ネガティブ': 0.02,
#         'ポジティブ': 0.98
#     }
# }

# 複数テキストの一括分析
texts = [
    "最高の体験でした",
    "がっかりした",
    "普通です"
]
results = analyzer.predict_batch(texts)

# 結果の見やすい表示
for result in results:
    analyzer.display_result(result)
```

## 出力例

```
============================================================
テキスト: 今日はとても楽しい一日でした！
------------------------------------------------------------
感情: ポジティブ
信頼度: 99.23%

確率分布:
  ネガティブ: 0.77% ▌
  ポジティブ: 99.23% ██████████████████████████████████████████████████
============================================================
```

## 使用モデル

### デフォルトモデル

**koheiduck/bert-japanese-finetuned-sentiment**

このプログラムでは、Hugging Faceで公開されている日本語感情分析モデルを使用しています。

| 項目 | 内容 |
|------|------|
| ベースモデル | `cl-tohoku/bert-base-japanese-v2`（東北大学） |
| タスク | **3クラス感情分類（ポジティブ/ニュートラル/ネガティブ）** |
| ライセンス | Apache 2.0 |
| 特徴 | 日本語Wikipediaで事前学習済み、感情分析用にファインチューニング |

### ラベルの意味

| ラベル | 説明 | 例 |
|--------|------|-----|
| ポジティブ | 肯定的・好意的な感情 | 「最高！」「素晴らしい」「おすすめ」 |
| ニュートラル | 中立的・客観的な表現 | 「普通です」「まあまあ」「可もなく不可もなく」 |
| ネガティブ | 否定的・批判的な感情 | 「最悪」「がっかり」「二度と行かない」 |

### モデルの仕組み

1. **トークン化**: 入力テキストを単語・サブワードに分割
2. **埋め込み**: 各トークンを768次元のベクトルに変換
3. **BERT処理**: 双方向Transformerで文脈を理解
4. **分類**: 最終層で感情クラスを予測

### 他のモデルを使用する場合

```python
# 3クラス分類モデル（ポジティブ/ニュートラル/ネガティブ）
analyzer = BERTSentimentAnalyzer(
    model_name="christian-phu/bert-finetuned-japanese-sentiment"
)

# 東北大学のBERTモデル（要ファインチューニング）
analyzer = BERTSentimentAnalyzer(
    model_name="cl-tohoku/bert-base-japanese-v3"
)
```

### 利用可能な日本語感情分析モデル

| モデル | クラス数 | 学習データ | 特徴 |
|--------|---------|-----------|------|
| `koheiduck/bert-japanese-finetuned-sentiment` | **3** | 日本語レビューデータ | 中立クラスあり、デフォルト |
| `christian-phu/bert-finetuned-japanese-sentiment` | 3 | Amazonレビュー | 中立クラスあり |
| `jarvisx17/japanese-sentiment-analysis` | 2 | 日本語極性辞書 | 辞書ベースの学習 |

---

# Googleマップ口コミ感情分析プログラム

Googleマップから店舗の口コミを取得し、BERTで感情分析を行うPythonプログラムです。

## 機能

✨ **主要機能**
- 🗺️ Google Maps APIで店舗検索
- 💬 口コミレビューの自動取得
- 🤖 BERT日本語モデルによる感情分析（ポジティブ/ネガティブ）
- 📊 分析結果のサマリー表示
- 💾 CSV形式でのエクスポート
- 🔍 評価スコアと感情分析の相関表示

## 必要な環境

- Python 3.8以上
- Google Maps API キー（Places API）
- uv（推奨）またはpip

## セットアップ

### 1. Google Maps APIキーの取得

Google Cloud Platformでプロジェクトを作成し、Places APIを有効化してAPIキーを取得してください。

**手順:**
1. [Google Cloud Console](https://console.cloud.google.com/)にアクセス
2. 新規プロジェクトを作成
3. 「APIとサービス」→「ライブラリ」から以下を有効化:
   - Places API
   - Maps JavaScript API（オプション）
4. 「認証情報」からAPIキーを作成
5. APIキーの使用制限を設定（推奨）

**料金について:**
- Places APIは従量課金制です
- 無料枠: 毎月$200分のクレジット
- Place Details (口コミ取得): 1リクエスト = $0.017
- 参考: [Google Maps Platform料金](https://mapsplatform.google.com/pricing/)

### 1.5. APIキーの設定

プロジェクトルートに `.env` ファイルを作成し、APIキーを設定します。

```bash
# .env.example をコピーして .env を作成
cp .env.example .env

# .env ファイルを編集してAPIキーを設定
# エディタで開いて your_api_key_here を実際のAPIキーに置き換え
```

**.env ファイルの内容:**
```
GOOGLE_MAPS_API_KEY=your_actual_api_key_here
```

> **⚠️ セキュリティ注意**: `.env` ファイルはGitにコミットしないでください。
> `.gitignore` に含まれているため、通常は自動的に除外されます。

### 2. 依存関係のインストール

#### 🎮 GPU版（NVIDIA GPU搭載の場合）

高速な処理が可能ですが、容量が大きくなります。

**容量目安**: 約5〜7GB

```bash
# 仮想環境の作成
uv venv
source .venv/bin/activate

# GPU版PyTorch + 依存関係
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers fugashi ipadic unidic-lite googlemaps pandas python-dotenv
```

#### 💻 CPU版（GPU未搭載・軽量環境の場合）

GPUがない環境や、ストレージを節約したい場合に推奨。

**容量目安**: 約1〜2GB

```bash
# 仮想環境の作成
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# CPU版PyTorch（軽量）+ 依存関係
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers fugashi ipadic unidic-lite googlemaps pandas python-dotenv
```

#### どちらを選ぶべき？

| 条件 | 推奨 |
|------|------|
| NVIDIA GPU搭載 & CUDA環境あり | GPU版 |
| GPU未搭載 / ストレージ節約したい | CPU版 |
| 実験・学習目的 | CPU版 |
| 大量の口コミを高速処理したい | GPU版 |

> **注意**: CPU版でも動作に問題はありません。処理速度が若干遅くなるだけです。
> 数十件程度の口コミ分析なら体感差はほとんどありません。

### 3. 環境の確認

```bash
# インストール後、GPUが認識されているか確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. 実験後の環境削除

環境の肥大化が気になる場合は、実験終了後に仮想環境を削除できます。

```bash
# 仮想環境を無効化
deactivate

# 仮想環境フォルダを削除
rm -rf .venv

# uvのキャッシュも削除したい場合
uv cache clean
```

## 使い方

### 基本的な実行

```bash
python googlemap_review_sentiment.py
```

プログラムを起動すると:
1. APIキーの入力を求められます
2. 店舗名を入力
3. 場所（オプション）を入力
4. 自動的に口コミを取得・分析

### 実行例

```
Googleマップ口コミ感情分析プログラム
======================================================================

Google Maps APIキーを入力してください:
APIキー: YOUR_API_KEY_HERE

======================================================================
【店舗検索】
======================================================================

分析したい店舗名を入力してください（終了: 'q'）: スターバックス
場所を入力してください（省略可、例: 名古屋市）: 名古屋駅

'スターバックス 名古屋駅' を検索中...
5件の店舗が見つかりました

【スターバックス コーヒー JR名古屋駅店】
住所: 愛知県名古屋市中村区名駅1-1-4 JRセントラルタワーズ 1F
評価: 4.2 ★ (850件)
======================================================================

口コミを取得中...
スターバックス コーヒー JR名古屋駅店: 5件の口コミを分析中...

  [1/5] ポジティブ (95.3%)
  [2/5] ポジティブ (89.7%)
  [3/5] ネガティブ (78.2%)
  [4/5] ポジティブ (92.1%)
  [5/5] ポジティブ (87.5%)

======================================================================
【分析サマリー】
======================================================================

店舗名: スターバックス コーヒー JR名古屋駅店
分析口コミ数: 5件

【BERT感情分析結果】
  ポジティブ: 4件 (80.0%)
  ネガティブ: 1件 (20.0%)

【Googleレビュー評価】
  平均評価: 4.20 ★
  ★★★★★ (5): 3件
  ★★★★ (4): 1件
  ★★ (2): 1件

【感情と評価の相関】
  ポジティブ口コミの平均評価: 4.75 ★
  ネガティブ口コミの平均評価: 2.00 ★

【注目すべき口コミ】

最もポジティブ (信頼度: 95.3%)
   評価: 5★
   「駅直結で便利です。朝は混んでいますが、スタッフの対応が素晴らしく、コーヒーも美味しいです。」

最もネガティブ (信頼度: 78.2%)
   評価: 2★
   「混雑していて席が全然空いていない。注文まで時間がかかりすぎる。」

======================================================================

結果をCSVファイルに保存しますか？ (y/n): y

結果を保存しました: reviews_スターバックス_コーヒー_JR名古屋駅店_20241106_120000.csv
```

### プログラム内での使用方法

```python
from googlemap_review_sentiment import GoogleMapsReviewAnalyzer

# 初期化
analyzer = GoogleMapsReviewAnalyzer(api_key="YOUR_API_KEY")

# 店舗を検索して分析
df = analyzer.analyze_store("スターバックス", location="名古屋駅")

# サマリー表示
analyzer.display_summary(df)

# CSV保存
analyzer.save_results(df, filename="starbucks_reviews.csv")

# DataFrameとして直接操作
print(df.head())
print(df['BERT感情'].value_counts())
```

### 詳細な使用例

```python
# 特定の店舗IDで分析
place_id = "ChIJN1t_tDeuEmsRUsoyG83frY4"
df = analyzer.analyze_reviews(place_id, "店舗名")

# 複数店舗を比較
stores = ["スターバックス 名古屋駅", "スターバックス 栄", "スターバックス 金山"]
results = []

for store in stores:
    df = analyzer.analyze_store(store)
    results.append(df)

# 結合して比較分析
all_reviews = pd.concat(results, ignore_index=True)
comparison = all_reviews.groupby('店舗名').agg({
    'BERT感情': lambda x: (x == 'ポジティブ').sum() / len(x) * 100,
    '評価': 'mean'
})
print(comparison)
```

## 出力データ

CSVファイルには以下の列が含まれます:

| 列名 | 説明 |
|------|------|
| 店舗名 | 分析対象の店舗名 |
| 投稿者 | レビュー投稿者名 |
| 評価 | Google評価（1-5★） |
| 投稿日時 | 投稿の相対時間 |
| 口コミテキスト | レビュー本文（短縮版） |
| 口コミ全文 | レビュー本文（完全版） |
| BERT感情 | ポジティブ/ネガティブ |
| BERT信頼度 | 予測の信頼度（0-1） |
| ポジティブ確率 | ポジティブである確率 |
| ネガティブ確率 | ネガティブである確率 |

## 📊 分析結果の見方

### 「感情と評価の相関」とは？

プログラムが出力する「感情と評価の相関」では、**BERTが判定した感情とGoogleの★評価の関係性**を分析しています。

#### 処理内容

1. **口コミをBERT感情で分類**
   ```python
   # ポジティブと判定された口コミだけを抽出
   positive_df = df[df['BERT感情'] == 'ポジティブ']
   
   # ニュートラルと判定された口コミだけを抽出
   neutral_df = df[df['BERT感情'] == 'ニュートラル']
   
   # ネガティブと判定された口コミだけを抽出
   negative_df = df[df['BERT感情'] == 'ネガティブ']
   ```

2. **各グループの★評価の平均を計算**
   ```python
   # ポジティブ口コミの★評価の平均
   positive_avg = positive_df['評価'].mean()
   # 例：(5★ + 5★ + 4★) ÷ 3件 = 4.67 ★
   ```

#### 実例で理解する

| 口コミ内容 | Google評価 | BERT判定 |
|-----------|-----------|----------|
| 「最高に美味しい！」 | 5★ | ポジティブ |
| 「スタッフが親切で丁寧」 | 5★ | ポジティブ |
| 「駅から近くて便利」 | 4★ | ポジティブ |
| 「メニューは豊富」 | 3★ | ニュートラル |
| 「待ち時間が長すぎる」 | 2★ | ネガティブ |

**出力される相関分析:**
```
【感情と評価の相関】
  ポジティブ口コミの平均評価: 4.67 ★  ← (5+5+4)÷3
  ニュートラル口コミの平均評価: 3.00 ★  ← (3)÷1
  ネガティブ口コミの平均評価: 2.00 ★  ← (2)÷1
```

#### パターン分析

**パターン1：正常なケース**
```
ポジティブ → 4.5★
ニュートラル → 3.0★
ネガティブ → 1.5★
```
→ テキストの感情と★評価が一致（健全な口コミ傾向）

**パターン2：中立的なのに低評価**
```
ポジティブ → 4.8★
ニュートラル → 2.5★  ← 注目！
ネガティブ → 1.8★
```
→ 実際の口コミ例：「店員の態度が普通だった。料理も普通。」→ 2★  
→ **客観的に書いているが不満を感じている可能性**

**パターン3：高評価でもネガティブワード**
```
ネガティブ口コミの平均評価: 4.00 ★  ← 高評価なのに？
```
→ 実際の口コミ例：  
「料理は最高に美味しい！ただ駐車場が狭くて停めにくいのが残念」→ 4★

→ **全体的には満足だが、改善してほしい点が明記されている**

#### ビジネス活用のポイント

| 活用方法 | 説明 | 具体例 |
|---------|------|--------|
| **改善点の発見** | 高評価＋ネガティブ判定 | 「美味しいけど駐車場が狭い」→駐車場改善 |
| **隠れた不満の検出** | 中評価＋ニュートラル判定 | 「普通でした」→期待値とのギャップあり |
| **真の満足度測定** | ★だけでなくテキスト感情も分析 | 5★でも「～だけど」があれば要確認 |
| **口コミの質の評価** | ニュートラルが多い | 事実記述中心、感情表現が少ない |

**重要**: 数値（★）だけでは見えない感情や改善要望をテキストから読み取るのがBERT感情分析の価値です。

---

## 使用モデル

### デフォルトモデル

**koheiduck/bert-japanese-finetuned-sentiment**

このプログラムでは、Hugging Faceで公開されている日本語感情分析モデルを使用しています。

| 項目 | 内容 |
|------|------|
| ベースモデル | `cl-tohoku/bert-base-japanese-v2`（東北大学） |
| タスク | **3クラス感情分類（ポジティブ/ニュートラル/ネガティブ）** |
| ライセンス | Apache 2.0 |
| 特徴 | 日本語Wikipediaで事前学習済み、感情分析用にファインチューニング |

### ラベルの意味

| ラベル | 説明 | 口コミでの例 |
|--------|------|-------------|
| ポジティブ | 肯定的・好意的な感情 | 「料理が美味しい」「接客が丁寧」「また来たい」 |
| ニュートラル | 中立的・客観的な表現 | 「駅から徒歩5分」「席数は30席程度」「メニューは豊富」 |
| ネガティブ | 否定的・批判的な感情 | 「待ち時間が長い」「値段が高い」「期待はずれ」 |

### モデルの仕組み

1. **トークン化**: 口コミテキストを単語・サブワードに分割
2. **埋め込み**: 各トークンを768次元のベクトルに変換
3. **BERT処理**: 双方向Transformerで文脈を理解
4. **分類**: 最終層で感情クラスを予測

### Google評価（★）とBERT感情の違い

| 指標 | Google評価（★） | BERT感情分析 |
|------|----------------|-------------|
| 入力 | ユーザーが選択した数値 | 口コミテキストの内容 |
| 粒度 | 1〜5の5段階 | ポジティブ/ニュートラル/ネガティブの**3値** |
| 特徴 | 総合的な満足度 | テキストから読み取れる感情 |
| 活用 | 定量的な比較 | 感情の詳細分析、★と文章の乖離発見 |

**分析のポイント**: 
- 高評価（4〜5★）なのにネガティブ判定 → 改善点が含まれている可能性
- 中評価（3★）がニュートラル判定 → 客観的な事実記述の可能性
- 低評価（1〜2★）がニュートラル判定 → 感情的でない冷静な批判

### 他のモデルを使用する場合

```python
from googlemap_review_sentiment import GoogleMapsReviewAnalyzer

# 3クラス分類モデルを使用
class CustomAnalyzer(GoogleMapsReviewAnalyzer):
    def __init__(self, api_key):
        self.gmaps = googlemaps.Client(key=api_key)
        # カスタムモデルを指定
        self.sentiment_analyzer = BERTSentimentAnalyzer(
            model_name="christian-phu/bert-finetuned-japanese-sentiment"
        )
```

### 利用可能な日本語感情分析モデル

| モデル | クラス数 | 学習データ | 特徴 |
|--------|---------|-----------|------|
| `koheiduck/bert-japanese-finetuned-sentiment` | **3** | 日本語レビューデータ | 中立クラスあり、デフォルト |
| `christian-phu/bert-finetuned-japanese-sentiment` | 3 | Amazonレビュー | 中立クラスあり |
| `jarvisx17/japanese-sentiment-analysis` | 2 | 日本語極性辞書 | 辞書ベースの学習 |
