"""
Googleマップ口コミ感情分析プログラム
店舗の口コミを取得してBERT感情分析を実行します

Places API (New) を使用
"""

import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from typing import List, Dict, Optional
import time
from datetime import datetime
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# .envファイルから環境変数を読み込み
load_dotenv()


class BERTSentimentAnalyzer:
    """BERT感情分析クラス"""
    
    def __init__(self, model_name: str = "koheiduck/bert-japanese-finetuned-sentiment"):
        """初期化"""
        print("BERTモデルを読み込んでいます...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用デバイス: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # モデルのラベル設定を取得
        if hasattr(self.model.config, 'id2label'):
            raw_labels = self.model.config.id2label
            # 英語ラベルを日本語に変換（大文字対応）
            self.labels = {}
            for idx, label in raw_labels.items():
                label_lower = label.lower()
                if label_lower in ['positive', 'ポジティブ', '1', 'pos']:
                    self.labels[idx] = 'ポジティブ'
                elif label_lower in ['negative', 'ネガティブ', '0', 'neg']:
                    self.labels[idx] = 'ネガティブ'
                elif label_lower in ['neutral', 'ニュートラル', '中立', 'neu']:
                    self.labels[idx] = 'ニュートラル'
                else:
                    self.labels[idx] = label
        else:
            self.labels = {0: "ネガティブ", 1: "ポジティブ"}
        
        print("BERTモデルの読み込み完了！\n")
    
    def predict(self, text: str) -> Dict:
        """テキストの感情を予測"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'sentiment': self.labels[predicted_class],
            'confidence': confidence,
            'positive_prob': probabilities[0][1].item(),
            'negative_prob': probabilities[0][0].item()
        }


class GoogleMapsReviewAnalyzer:
    """Googleマップ口コミ分析クラス（Places API New対応）"""
    
    def __init__(self, api_key: str):
        """
        初期化
        
        Args:
            api_key: Google Maps API キー
        """
        self.api_key = api_key
        self.base_url = "https://places.googleapis.com/v1"
        self.sentiment_analyzer = BERTSentimentAnalyzer()
        print("Googleマップ口コミ分析の準備完了（Places API New使用）\n")
    
    def search_place(self, query: str, location: Optional[str] = None) -> List[Dict]:
        """
        店舗を検索（Text Search New使用）
        
        Args:
            query: 検索クエリ（店舗名など）
            location: 場所（例: "東京都渋谷区"）
            
        Returns:
            検索結果のリスト
        """
        print(f"'{query}' を検索中...")
        
        if location:
            query = f"{query} {location}"
        
        url = f"{self.base_url}/places:searchText"
        
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount"
        }
        
        data = {
            "textQuery": query,
            "languageCode": "ja"
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
            places = result.get('places', [])
            print(f"{len(places)}件の店舗が見つかりました\n")
            return places
        except requests.exceptions.RequestException as e:
            print(f"検索エラー: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"エラー詳細: {e.response.text}")
            return []
    
    def get_place_details(self, place_id: str) -> Dict:
        """
        店舗の詳細情報を取得（Place Details New使用）
        
        Args:
            place_id: Google Places ID
            
        Returns:
            店舗の詳細情報
        """
        url = f"{self.base_url}/places/{place_id}"
        
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "displayName,formattedAddress,internationalPhoneNumber,rating,userRatingCount,reviews"
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"詳細取得エラー: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"エラー詳細: {e.response.text}")
            return {}
    
    def analyze_reviews(self, place_id: str, place_name: str = None) -> pd.DataFrame:
        """
        口コミを取得して感情分析
        
        Args:
            place_id: Google Places ID
            place_name: 店舗名（表示用）
            
        Returns:
            分析結果のDataFrame
        """
        print(f"口コミを取得中...")
        details = self.get_place_details(place_id)
        
        if not details:
            print("店舗情報を取得できませんでした")
            return pd.DataFrame()
        
        store_name = place_name or details.get('displayName', {}).get('text', '不明')
        reviews = details.get('reviews', [])
        
        if not reviews:
            print(f"{store_name}: 口コミが見つかりませんでした")
            return pd.DataFrame()
        
        print(f"{store_name}: {len(reviews)}件の口コミを分析中...\n")
        
        analyzed_reviews = []
        
        for i, review in enumerate(reviews, 1):
            # レビューテキストの取得
            text = review.get('text', {}).get('text', '') if isinstance(review.get('text'), dict) else review.get('text', '')
            
            if not text:
                continue
            
            # 感情分析実行
            sentiment_result = self.sentiment_analyzer.predict(text)
            
            # 著者名の取得
            author_name = '匿名'
            if 'authorAttribution' in review:
                author_name = review['authorAttribution'].get('displayName', '匿名')
            
            # 相対時間の取得
            relative_time = review.get('relativePublishTimeDescription', '不明')
            
            analyzed_reviews.append({
                '店舗名': store_name,
                '投稿者': author_name,
                '評価': review.get('rating', 0),
                '投稿日時': relative_time,
                '口コミテキスト': text[:100] + '...' if len(text) > 100 else text,
                '口コミ全文': text,
                'BERT感情': sentiment_result['sentiment'],
                'BERT信頼度': sentiment_result['confidence'],
                'ポジティブ確率': sentiment_result['positive_prob'],
                'ネガティブ確率': sentiment_result['negative_prob']
            })
            
            print(f"  [{i}/{len(reviews)}] {sentiment_result['sentiment']} ({sentiment_result['confidence']:.1%})")
        
        df = pd.DataFrame(analyzed_reviews)
        print()
        return df
    
    def analyze_store(self, query: str, location: Optional[str] = None) -> pd.DataFrame:
        """
        店舗を検索して口コミを分析（ワンストップ関数）
        
        Args:
            query: 店舗名
            location: 場所
            
        Returns:
            分析結果のDataFrame
        """
        # 店舗検索
        places = self.search_place(query, location)
        
        if not places:
            print("店舗が見つかりませんでした")
            return pd.DataFrame()
        
        # 最初の検索結果を使用
        place = places[0]
        place_id = place['id']
        place_name = place.get('displayName', {}).get('text', '不明')
        place_address = place.get('formattedAddress', '不明')
        place_rating = place.get('rating', '-')
        place_rating_count = place.get('userRatingCount', 0)
        
        print(f"【{place_name}】")
        print(f"住所: {place_address}")
        print(f"評価: {place_rating} ★ ({place_rating_count}件)\n")
        print("=" * 70)
        
        # 口コミ分析
        df = self.analyze_reviews(place_id, place_name)
        
        return df
    
    def display_summary(self, df: pd.DataFrame):
        """分析結果のサマリーを表示"""
        if df.empty:
            print("分析結果がありません")
            return
        
        print("\n" + "=" * 70)
        print("【分析サマリー】")
        print("=" * 70)
        
        store_name = df['店舗名'].iloc[0]
        total_reviews = len(df)
        
        # 感情分布
        sentiment_counts = df['BERT感情'].value_counts()
        
        positive_count = sentiment_counts.get('ポジティブ', 0)
        negative_count = sentiment_counts.get('ネガティブ', 0)
        neutral_count = sentiment_counts.get('ニュートラル', 0)
        
        positive_ratio = positive_count / total_reviews * 100 if total_reviews > 0 else 0
        negative_ratio = negative_count / total_reviews * 100 if total_reviews > 0 else 0
        neutral_ratio = neutral_count / total_reviews * 100 if total_reviews > 0 else 0
        
        print(f"\n店舗名: {store_name}")
        print(f"分析口コミ数: {total_reviews}件\n")
        
        print("【BERT感情分析結果】")
        print(f"  ポジティブ: {positive_count}件 ({positive_ratio:.1f}%)")
        if neutral_count > 0:
            print(f"  ニュートラル: {neutral_count}件 ({neutral_ratio:.1f}%)")
        print(f"  ネガティブ: {negative_count}件 ({negative_ratio:.1f}%)\n")
        
        # 評価スコアとの比較
        print("【Googleレビュー評価】")
        avg_rating = df['評価'].mean()
        print(f"  平均評価: {avg_rating:.2f} ★")
        
        rating_dist = df['評価'].value_counts().sort_index(ascending=False)
        for rating, count in rating_dist.items():
            bar = "★" * int(rating)
            print(f"  {bar} ({rating}): {count}件")
        
        print("\n【感情と評価の相関】")
        positive_df = df[df['BERT感情'] == 'ポジティブ']
        negative_df = df[df['BERT感情'] == 'ネガティブ']
        neutral_df = df[df['BERT感情'] == 'ニュートラル']
        
        if len(positive_df) > 0:
            positive_avg_rating = positive_df['評価'].mean()
            print(f"  ポジティブ口コミの平均評価: {positive_avg_rating:.2f} ★")
        
        if len(neutral_df) > 0:
            neutral_avg_rating = neutral_df['評価'].mean()
            print(f"  ニュートラル口コミの平均評価: {neutral_avg_rating:.2f} ★")
        
        if len(negative_df) > 0:
            negative_avg_rating = negative_df['評価'].mean()
            print(f"  ネガティブ口コミの平均評価: {negative_avg_rating:.2f} ★")
        
        print("\n【注目すべき口コミ】")
        
        # 最もポジティブな口コミ
        if len(positive_df) > 0:
            most_positive = positive_df.nlargest(1, 'BERT信頼度')
            if not most_positive.empty:
                review = most_positive.iloc[0]
                print(f"\n✨ 最もポジティブ (信頼度: {review['BERT信頼度']:.1%})")
                print(f"   評価: {review['評価']}★")
                print(f"   「{review['口コミテキスト']}」")
        
        # 最もネガティブな口コミ
        if len(negative_df) > 0:
            most_negative = negative_df.nlargest(1, 'BERT信頼度')
            if not most_negative.empty:
                review = most_negative.iloc[0]
                print(f"\n⚠️  最もネガティブ (信頼度: {review['BERT信頼度']:.1%})")
                print(f"   評価: {review['評価']}★")
                print(f"   「{review['口コミテキスト']}」")
        
        print("\n" + "=" * 70)
    
    def save_results(self, df: pd.DataFrame, filename: str = None):
        """結果をCSVファイルに保存"""
        if df.empty:
            print("保存するデータがありません")
            return
        
        if filename is None:
            store_name = df['店舗名'].iloc[0].replace(' ', '_').replace('/', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"reviews_{store_name}_{timestamp}.csv"
        
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n結果を保存しました: {filename}")


def main():
    """メイン処理"""
    
    print("=" * 70)
    print("Googleマップ口コミ感情分析プログラム（Places API New）")
    print("=" * 70)
    print()
    
    # .envからAPIキーを取得
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    
    if not api_key:
        print("⚠️  APIキーが設定されていません")
        print()
        print("以下のいずれかの方法でAPIキーを設定してください：")
        print()
        print("【方法1】.envファイルを作成")
        print("  プロジェクトルートに .env ファイルを作成し、以下を記載：")
        print("  GOOGLE_MAPS_API_KEY=your_api_key_here")
        print()
        print("【方法2】環境変数を設定")
        print("  export GOOGLE_MAPS_API_KEY=your_api_key_here")
        print()
        print("APIキーの取得方法:")
        print("  https://developers.google.com/maps/documentation/places/web-service/cloud-setup")
        print()
        print("⚠️ 注意: Places API (New) を有効化してください")
        return
    
    print(f"✅ APIキーを読み込みました（.env）")
    print()
    
    # 分析器の初期化
    analyzer = GoogleMapsReviewAnalyzer(api_key)
    
    while True:
        print("\n" + "=" * 70)
        print("【店舗検索】")
        print("=" * 70)
        
        # 店舗名入力
        store_name = input("\n分析したい店舗名を入力してください（終了: 'q'）: ").strip()
        
        if store_name.lower() in ['q', 'quit', '']:
            print("プログラムを終了します")
            break
        
        # 場所入力（オプション）
        location = input("場所を入力してください（省略可、例: 名古屋市）: ").strip()
        location = location if location else None
        
        # 分析実行
        print()
        df = analyzer.analyze_store(store_name, location)
        
        if not df.empty:
            # サマリー表示
            analyzer.display_summary(df)
            
            # 保存確認
            save = input("\n結果をCSVファイルに保存しますか？ (y/n): ").strip().lower()
            if save == 'y':
                analyzer.save_results(df)
            
            # 詳細表示確認
            detail = input("\n全口コミの詳細を表示しますか？ (y/n): ").strip().lower()
            if detail == 'y':
                print("\n【全口コミ詳細】")
                print("=" * 70)
                for idx, row in df.iterrows():
                    print(f"\n[{idx + 1}] {row['投稿者']} - {row['投稿日時']}")
                    print(f"評価: {row['評価']}★ | BERT: {row['BERT感情']} ({row['BERT信頼度']:.1%})")
                    print(f"口コミ: {row['口コミ全文']}")
                    print("-" * 70)


if __name__ == "__main__":
    main()