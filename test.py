import requests
import json
import os

API_KEY = os.getenv("OPENROUTER_API_KEY") or "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"

def test_raw_request():
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:3000",
        "X-Title": "Raw Request Test"
    }
    
    data = {
        "model": "tngtech/deepseek-r1t2-chimera:free",
        "messages": [
            {"role": "user", "content": "API接続テスト。Helloと言ってください。"}
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        # ステータスコードのチェック
        if response.status_code == 200:
            result = response.json()
            print("✅ 成功:", result['choices'][0]['message']['content'])
        else:
            print(f"❌ 失敗 (Status {response.status_code}):", response.text)
            
    except Exception as e:
        print(f"❌ 通信エラー: {e}")

if __name__ == "__main__":
    test_raw_request() 