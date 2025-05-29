import subprocess
import json

class OllamaEngine:
    def __init__(self, model="mistral"):
        self.model = model

    def analyze_logs(self, trades):
        prompt = self._build_prompt(trades)
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            output = result.stdout.decode()
            return self._extract_json(output)
        except Exception as e:
            return {"error": str(e)}

    def _build_prompt(self, trades):
        return f"""
Elemezd az alábbi kriptokereskedési ügyleteket, és javasolj új RSI, EMA és SL/TP beállításokat. Formátum: JSON.

Ügyletek:
{json.dumps(trades[-50:], indent=2)}

Adj vissza egy javaslatot:
{{
  "rsi_threshold": {{"buy": ..., "sell": ...}},
  "ema": {{"fast": ..., "slow": ...}},
  "sl_percent": ...,
  "tp_percent": ...
}}
"""

    def _extract_json(self, output):
        try:
            start = output.find('{')
            end = output.rfind('}') + 1
            return json.loads(output[start:end])
        except:
            return {"raw_output": output.strip()}

