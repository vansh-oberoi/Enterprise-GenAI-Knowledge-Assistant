"""Baseline retrieval-hit evaluation; requires API server and configured LLM."""
import json
from pathlib import Path
import requests

API_URL = 'http://127.0.0.1:8000'

def main():
    cases = [json.loads(line) for line in Path('eval/questions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    hits = 0
    with_sources = 0
    for case in cases:
        response = requests.post(f'{API_URL}/query', json={'question': case['question'], 'top_k': 5}, timeout=120)
        response.raise_for_status()
        result = response.json()
        sources = {item['source'] for item in result.get('sources', [])}
        hit = bool(sources & set(case.get('expected_sources', [])))
        hits += int(hit)
        with_sources += int(bool(sources))
        print(json.dumps({'question': case['question'], 'retrieval_hit': hit, 'sources': sorted(sources)}))
    total = len(cases)
    print(json.dumps({'cases': total, 'retrieval_hit_rate': hits / total if total else 0.0, 'cases_with_sources_rate': with_sources / total if total else 0.0}, indent=2))

if __name__ == '__main__':
    main()
