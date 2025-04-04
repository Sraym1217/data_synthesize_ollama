# data synthesize ollama 

このドキュメントでは、Macで動作するOllamaを使用してLLMを呼び出すPythonスクリプトの仕様と使い方について説明します。

## 概要

このスクリプトは以下の機能を提供します：

- JSONLファイルから入力テキストを読み込む
- 指定されたOllamaモデルを使用してテキストを処理
- 結果を指定された形式でJSONLファイルに出力

## 前提条件

- [Ollama](https://ollama.com/)がMacにインストールされていること
- Ollamaサービスが起動していること
- Pythonがインストールされていること
- 必要なPythonライブラリがインストールされていること：
  ```bash
  pip install ollama argparse
  ```

## スクリプトの使い方

### 基本的な使用方法

```bash
python ollama_script.py --model <モデル名> --input <入力ファイルパス> --output <出力ファイルパス>
```

### コマンドライン引数

| 引数 | 説明 | 必須 | 例 |
|------|------|------|------|
| `--model` | 使用するOllamaモデル名 | はい | `llama3`, `phi3` |
| `--input` | 入力JSONLファイルのパス | はい | `input.jsonl` |
| `--output` | 出力JSONLファイルのパス | はい | `output.jsonl` |

### 使用例

```bash
python ollama_script.py --model llama3 --input data/questions.jsonl --output results/answers.jsonl
```

## データ形式

### 入力JSONL形式

各行は以下の形式のJSONオブジェクトです：

```json
{
  "id": 1,
  "role": ["user"],
  "text": ["こんにちは、今日の天気を教えてください"]
}
```

| フィールド | 型 | 説明 |
|----------|-----|------|
| `id` | number | リクエストの一意の識別子 |
| `role` | array | ユーザーロールを示す配列（現在は `["user"]` のみ） |
| `text` | array | 入力テキストを含む配列（現在は要素1つ） |

### 出力JSONL形式

各行は以下の形式のJSONオブジェクトです：

```json
{
  "id": 1,
  "role": ["user", "llama3"],
  "text": [
    "こんにちは、今日の天気を教えてください",
    "申し訳ありませんが、私は特定の地域の現在の天気情報にアクセスできません。..."
  ]
}
```

| フィールド | 型 | 説明 |
|----------|-----|------|
| `id` | number | 入力と同じ識別子 |
| `role` | array | `["user", "<モデル名>"]` の形式 |
| `text` | array | `[<入力テキスト>, <モデルの応答>]` の形式 |

## サポートされているモデル

Ollamaでサポートされている主なモデル：

- `llama3` - Meta AI's Llama 3（デフォルト）
- `phi3` - Microsoft Phi-3
- `mistral` - Mistral AI's Mistral
- `gemma` - Google's Gemma
- `codellama` - コード生成特化モデル
- `llama3:8b` - Llama 3の小型バージョン
- `llama3:70b` - Llama 3の大型バージョン

詳細なモデル一覧は `ollama list` コマンドで確認できます。

## 将来的な拡張機能

### マルチターン会話

将来のバージョンでは、以下の形式でマルチターン会話をサポート予定：

**入力例**:
```json
{
  "id": 1,
  "role": ["user", "llama3", "user"],
  "text": [
    "こんにちは",
    "こんにちは、どうされましたか？",
    "人工知能について教えてください"
  ]
}
```

**出力例**:
```json
{
  "id": 1,
  "role": ["user", "llama3", "user", "llama3"],
  "text": [
    "こんにちは",
    "こんにちは、どうされましたか？",
    "人工知能について教えてください",
    "人工知能（AI）とは、人間のような知能を持つ機械やプログラムを指します..."
  ]
}
```

## スクリプトの内部設定

デフォルトの設定値（config.jsonでカスタマイズ可能）：

| 設定 | デフォルト値 | 説明 |
|------|------------|------|
| `temperature` | 0.7 | 生成テキストの多様性（0.0～1.0） |
| `top_p` | 0.9 | 確率分布のカットオフ閾値 |
| `top_k` | 40 | 考慮するトークンの数 |
| `max_tokens` | 2048 | 生成する最大トークン数 |
| `batch_size` | 10 | 一度に処理するリクエスト数 |
| `retry_attempts` | 3 | エラー時の再試行回数 |
| `timeout` | 120 | リクエストのタイムアウト（秒） |
