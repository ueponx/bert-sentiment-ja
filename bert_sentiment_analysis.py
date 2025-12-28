"""
BERT感情分析プログラム
日本語テキストの感情を分析します
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class BERTSentimentAnalyzer:
    """BERT感情分析クラス"""
    
    def __init__(self, model_name: str = "koheiduck/bert-japanese-finetuned-sentiment"):
        """
        初期化
        
        Args:
            model_name: 使用するBERTモデル名
            
            利用可能なモデル例:
            - koheiduck/bert-japanese-finetuned-sentiment (デフォルト、2クラス)
            - christian-phu/bert-finetuned-japanese-sentiment (3クラス)
        """
        print("モデルを読み込んでいます...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用デバイス: {self.device}")
        
        # トークナイザーとモデルの読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # モデルのラベル設定を取得
        self.model_name = model_name
        self._setup_labels()
        
        print("モデルの読み込みが完了しました！\n")
    
    def _setup_labels(self):
        """モデルに応じたラベル設定"""
        # モデルの設定からラベルを取得
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
            # デフォルトのラベル（2クラス分類）
            self.labels = {
                0: "ネガティブ",
                1: "ポジティブ"
            }
    
    def predict(self, text: str) -> Dict:
        """
        テキストの感情を予測
        
        Args:
            text: 分析するテキスト
            
        Returns:
            予測結果の辞書（ラベル、スコア、確率分布）
        """
        # テキストのトークン化
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # デバイスに転送
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 予測
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'text': text,
            'sentiment': self.labels[predicted_class],
            'confidence': confidence,
            'probabilities': {
                self.labels[i]: probabilities[0][i].item() 
                for i in range(len(self.labels))
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        複数テキストの感情を一括予測
        
        Args:
            texts: 分析するテキストのリスト
            
        Returns:
            予測結果のリスト
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def display_result(self, result: Dict):
        """
        結果を見やすく表示
        
        Args:
            result: predict()の戻り値
        """
        print("=" * 60)
        print(f"テキスト: {result['text']}")
        print("-" * 60)
        print(f"感情: {result['sentiment']}")
        print(f"信頼度: {result['confidence']:.2%}")
        print("\n確率分布:")
        for label, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"  {label}: {prob:.2%} {bar}")
        print("=" * 60)
        print()


def main():
    """メイン処理"""
    
    # 分析器の初期化
    analyzer = BERTSentimentAnalyzer()
    
    # サンプルテキストで分析
    sample_texts = [
        "今日はとても楽しい一日でした！",
        "この映画は最高に面白かった。",
        "天気が悪くて気分が沈む。",
        "サービスが最悪で、二度と利用したくない。",
        "普通の製品だと思います。",
        "期待以上の素晴らしい体験ができました！"
    ]
    
    print("【サンプルテキストの感情分析】\n")
    
    # 一つずつ分析して表示
    for text in sample_texts:
        result = analyzer.predict(text)
        analyzer.display_result(result)
    
    # インタラクティブモード
    print("\n" + "=" * 60)
    print("【インタラクティブモード】")
    print("分析したいテキストを入力してください（終了: 'q' または 'quit'）")
    print("=" * 60 + "\n")
    
    while True:
        user_input = input("テキスト: ").strip()
        
        if user_input.lower() in ['q', 'quit', '']:
            print("分析を終了します。")
            break
        
        result = analyzer.predict(user_input)
        print()
        analyzer.display_result(result)


if __name__ == "__main__":
    main()