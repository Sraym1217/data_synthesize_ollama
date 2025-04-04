#!/usr/bin/env python3
import argparse
import json
import ollama
import os
import time
import sys
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class OllamaProcessor:
    """Ollamaを使ってLLMを処理するクラス"""
    
    def __init__(self, config_path="template_config.json"):
        """初期化"""
        # 設定を読み込む
        self.config = self._load_config(config_path)
        
        # 設定から各種パラメータを取得
        self.ollama_settings = self.config.get("ollama_settings", {})
        self.script_settings = self.config.get("script_settings", {})
        
        # APIホストの設定
        api_host = self.ollama_settings.get("api_host")
        if api_host:
            os.environ["OLLAMA_HOST"] = api_host
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 設定ファイル {config_path} が見つかりません。デフォルト設定を使用します。")
            return {
                "ollama_settings": {
                    "api_host": "http://localhost:11434",
                    "request_options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "max_tokens": 2048
                    }
                },
                "script_settings": {
                    "batch_size": 10,
                    "retry_attempts": 3,
                    "timeout": 120
                },
                "input_format": {
                    "required_fields": ["id", "role", "text"]
                },
                "output_format": {
                    "fields": ["id", "role", "text"]
                },
                "multi_turn_conversation": {
                    "enabled": False
                }
            }
        except json.JSONDecodeError:
            print(f"エラー: 設定ファイル {config_path} の形式が不正です。")
            sys.exit(1)
    
    def read_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """指定されたJSONLファイルを読み込む"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line_num, line in enumerate(tqdm(lines, desc="JSONLファイルを読み込み中"), 1):
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"警告: {line_num}行目のJSON形式が不正です。スキップします。")
        except FileNotFoundError:
            print(f"エラー: 入力ファイル {file_path} が見つかりません。")
            sys.exit(1)
        
        return data
    
    def write_jsonl(self, file_path: str, data: List[Dict[str, Any]]) -> None:
        """指定されたデータをJSONLファイルに書き込む"""
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in tqdm(data, desc="結果を保存中"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def validate_input(self, item: Dict[str, Any]) -> bool:
        """入力データの形式が正しいか検証する"""
        required_fields = self.config.get("input_format", {}).get("required_fields", ["id", "role", "text"])
        
        # 必須フィールドの存在確認
        for field in required_fields:
            if field not in item:
                return False
        
        # roleとtextが配列形式であることを確認
        if not isinstance(item.get('role'), list) or not item.get('role'):
            return False
        
        if not isinstance(item.get('text'), list) or not item.get('text'):
            return False
        
        return True
    
    def check_model_availability(self, model_name: str) -> bool:
        """指定されたモデルが利用可能かチェックする"""
        try:
            # モデル一覧を取得
            print("利用可能なモデルを確認中...")
            models_response = ollama.list()
            
            # デバッグ情報
            print(f"Ollama応答の型: {type(models_response)}")
            
            # 応答形式によって処理を分岐
            available_models = []
            
            if isinstance(models_response, dict) and 'models' in models_response:
                # 旧形式: {'models': [{'name': 'model1'}, {'name': 'model2'}]}
                available_models = [model['name'] for model in models_response.get('models', [])]
            elif isinstance(models_response, list):
                # 新形式: [{'name': 'model1'}, {'name': 'model2'}]
                available_models = [model.get('name', '') for model in models_response if 'name' in model]
            else:
                # その他の形式: キーをチェックして対応
                print(f"予期しない応答形式: {models_response}")
                if isinstance(models_response, dict):
                    # キーを出力して確認
                    print(f"応答のキー: {list(models_response.keys())}")
                    
                    # 適切なキーを探す
                    for key in models_response.keys():
                        if isinstance(models_response[key], list) and models_response[key]:
                            if isinstance(models_response[key][0], dict) and 'name' in models_response[key][0]:
                                available_models = [model['name'] for model in models_response[key]]
                                break
            
            # 利用可能なモデルを表示
            print(f"利用可能なモデル: {available_models}")
            
            # 完全一致または前方一致でチェック
            for available in available_models:
                if model_name == available or available.startswith(f"{model_name}:"):
                    return True
            
            print(f"警告: モデル '{model_name}' はローカルにインストールされていません。")
            
            # モデルの自動ダウンロードを確認
            if input(f"モデル '{model_name}' をダウンロードしますか？ (y/n): ").lower() == 'y':
                print(f"モデル '{model_name}' をダウンロード中...")
                with tqdm(total=100, desc=f"モデル {model_name} をダウンロード中") as pbar:
                    # ダウンロード開始
                    ollama.pull(model_name)
                    # ダウンロードが完了したらプログレスバーを100%にする
                    pbar.update(100)
                return True
            
            return False
        
        except Exception as e:
            print(f"エラー: Ollamaサービスへの接続中にエラーが発生しました: {str(e)}")
            print("例外の詳細:")
            import traceback
            traceback.print_exc()
            print("Ollamaがインストールされ、サービスが実行されていることを確認してください。")
            return False
    
    def process_item(self, model_name: str, item: Dict[str, Any], retry_attempts: int) -> Optional[Dict[str, Any]]:
        """Ollamaを使用して単一のアイテムを処理する"""
        if not self.validate_input(item):
            print(f"警告: 無効な形式のアイテムをスキップします: {item}")
            return None
        
        # リクエストオプションを取得
        request_options = self.ollama_settings.get("request_options", {})
        
        # マルチターン会話の有効/無効を確認
        multi_turn_enabled = self.config.get("multi_turn_conversation", {}).get("enabled", False)
        
        # リトライロジック
        for attempt in range(retry_attempts):
            try:
                if multi_turn_enabled and len(item['role']) > 1:
                    # マルチターン会話の場合
                    messages = []
                    for i, role in enumerate(item['role']):
                        if i < len(item['text']):
                            # Ollamaの仕様に合わせてroleを変換（userかassistant）
                            ollama_role = "assistant" if role != "user" else "user"
                            messages.append({
                                'role': ollama_role,
                                'content': item['text'][i]
                            })
                    
                    # 最後のメッセージがユーザーからでない場合は処理をスキップ
                    if not messages or messages[-1]['role'] != "user":
                        print(f"警告: ID {item.get('id', '不明')} の最後のメッセージがユーザーからではありません。スキップします。")
                        return None
                else:
                    # 単一ターンの場合
                    messages = [{
                        'role': 'user',
                        'content': item['text'][0]
                    }]
                
                # Ollamaでチャット応答を取得
                response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    options=request_options
                )
                
                # モデルからの応答テキストを取得
                model_response = response['message']['content']
                
                # 結果を適切な形式で保存
                result = {
                    'id': item['id'],
                    'role': item['role'] + [model_name],
                    'text': item['text'] + [model_response]
                }
                
                return result
                
            except Exception as e:
                print(f"エラー (試行 {attempt+1}/{retry_attempts}): ID {item.get('id', '不明')} の処理中にエラーが発生しました: {e}")
                if attempt < retry_attempts - 1:
                    # リトライ前に少し待機
                    time.sleep(2)
        
        print(f"警告: ID {item.get('id', '不明')} の最大リトライ回数に達しました。スキップします。")
        return None
    
    def process_batch(self, model_name: str, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """バッチ処理を行う"""
        results = []
        retry_attempts = self.script_settings.get("retry_attempts", 3)
        
        for item in tqdm(batch, desc="バッチ処理中"):
            result = self.process_item(model_name, item, retry_attempts)
            if result:
                results.append(result)
        
        return results
    
    def process_all(self, model_name: str, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """全データをバッチに分けて処理する"""
        results = []
        batch_size = self.script_settings.get("batch_size", 10)
        
        # 全体の進行状況を表示
        total_batches = (len(input_data) + batch_size - 1) // batch_size
        with tqdm(total=len(input_data), desc=f"モデル {model_name} で処理中") as pbar:
            # データをバッチに分割して処理
            for i in range(0, len(input_data), batch_size):
                batch = input_data[i:i+batch_size]
                batch_results = self.process_batch(model_name, batch)
                results.extend(batch_results)
                pbar.update(len(batch))
        
        return results
    
    def run(self, model_name: str, input_path: str, output_path: str) -> None:
        """メイン処理を実行する"""
        # モデルが利用可能かチェック
        if not self.check_model_availability(model_name):
            print(f"エラー: モデル '{model_name}' が利用できません。")
            sys.exit(1)
        
        # 入力ファイルを読み込む
        print(f"入力ファイル {input_path} を読み込んでいます...")
        input_data = self.read_jsonl(input_path)
        print(f"{len(input_data)}件のデータを読み込みました")
        
        if not input_data:
            print("警告: 入力データが空です。処理を終了します。")
            sys.exit(0)
        
        # Ollamaで処理
        results = self.process_all(model_name, input_data)
        print(f"{len(results)}/{len(input_data)}件のデータの処理が完了しました")
        
        # 結果を保存
        print(f"結果を {output_path} に保存しています...")
        self.write_jsonl(output_path, results)
        print("処理が完了しました！")


def main():
    parser = argparse.ArgumentParser(description='OllamaでLLMを使用して入力テキストを処理するスクリプト')
    parser.add_argument('--model', type=str, required=True, help='使用するOllamaモデル名')
    parser.add_argument('--input', type=str, required=True, help='入力JSONLファイルのパス')
    parser.add_argument('--output', type=str, required=True, help='出力JSONLファイルのパス')
    parser.add_argument('--config', type=str, default='template_config.json', help='設定ファイルのパス（デフォルト: template_config.json）')
    
    args = parser.parse_args()
    
    # プロセッサを初期化して実行
    processor = OllamaProcessor(args.config)
    processor.run(args.model, args.input, args.output)


if __name__ == "__main__":
    main()