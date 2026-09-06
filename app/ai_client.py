import json
import logging
import re

import requests

from app.env import get_api_key

logger = logging.getLogger(__name__)


class AIClient:
    def __init__(self, config):
        ai = config.get('ai', {})
        self.provider = str(ai.get('provider', 'gemini')).lower()
        self.api_key = get_api_key(config)
        self.base_url = ai.get(
            'base_url',
            'https://generativelanguage.googleapis.com/v1beta/models'
        ).rstrip('/')
        self.model = ai.get('model', 'gemini-2.5-flash')
        self.timeout = int(ai.get('timeout_seconds', 90))
        self.provider_label = 'gemini_api' if 'gemini' in self.provider else 'openai_api'

    def _request(self, instructions, input_text):
        if not self.api_key:
            env_name = 'GEMINI_API_KEY' if 'gemini' in self.provider else 'OPENAI_API_KEY'
            raise RuntimeError(f'AI API 키가 없습니다. .env에 {env_name}=... 를 설정하세요.')

        if 'gemini' in self.provider:
            return self._request_gemini(instructions, input_text)
        return self._request_openai(instructions, input_text)

    def _request_gemini(self, instructions, input_text):
        url = f'{self.base_url}/{self.model}:generateContent'
        prompt = f'{instructions}\n\n[분석 대상 데이터]\n{input_text}'
        payload = {
            'contents': [
                {
                    'role': 'user',
                    'parts': [{'text': prompt}],
                }
            ],
            'generationConfig': {
                'responseMimeType': 'application/json',
                'temperature': 0.1,
            },
        }
        headers = {
            'x-goog-api-key': self.api_key,
            'Content-Type': 'application/json',
        }
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:1000]
            raise RuntimeError(f'Gemini API 호출 실패: HTTP {response.status_code} - {detail}') from exc

        data = response.json()
        candidates = data.get('candidates') or []
        parts = []
        for candidate in candidates:
            content = candidate.get('content') or {}
            for part in content.get('parts') or []:
                text = part.get('text')
                if text:
                    parts.append(text)

        if not parts:
            raise RuntimeError(f'Gemini 응답에서 text를 찾지 못했습니다: {data}')
        return '\n'.join(parts)

    def _request_openai(self, instructions, input_text):
        payload = {
            'model': self.model,
            'instructions': instructions,
            'input': input_text,
        }
        response = requests.post(
            self.base_url,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        parts = []
        for item in data.get('output', []):
            for content in item.get('content', []):
                if content.get('type') == 'output_text' and content.get('text'):
                    parts.append(content['text'])
        if not parts:
            raise RuntimeError(f'output_text가 없는 AI 응답: {data}')
        return '\n'.join(parts)

    @staticmethod
    def parse_json(text):
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'(\[.*\]|\{.*\})', text, re.S)
            if not match:
                raise
            return json.loads(match.group(1))

    def analyze_batch(self, rows):
        payload = [
            {
                'id': int(row['id']),
                'text': row.get('review_text') or row.get('original_text') or '',
            }
            for row in rows
        ]
        instructions = (
            '고객 리뷰 감정 분석기다. 각 리뷰의 텍스트만 보고 감정을 분석한다. '
            'sentiment는 반드시 positive, negative, neutral 중 하나로 판단하고, '
            'confidence는 판단 신뢰도를 0.0~1.0 숫자로 반환한다. '
            '반드시 JSON 배열만 출력한다. '
            '각 원소 형식은 {"id":1,"sentiment":"positive","confidence":0.95} 이다.'
        )
        result = self.parse_json(
            self._request(instructions, json.dumps(payload, ensure_ascii=False))
        )
        if not isinstance(result, list):
            raise ValueError('감정 분석 응답은 JSON 배열이어야 합니다.')
        return result

    def extract(self, rows):
        reviews = [
            {
                'rating': row.get('rating'),
                'date': row.get('review_date'),
                'product': row.get('product'),
                'sentiment': row.get('sentiment'),
                'confidence': row.get('confidence'),
                'text': row.get('review_text') or row.get('original_text') or '',
            }
            for row in rows
        ]
        instructions = (
            '전자제품 고객 리뷰를 비즈니스 관점에서 종합 분석한다. '
            '긍정 키워드와 부정 키워드는 실제 리뷰에서 반복적으로 나타난 내용을 중심으로 최대 5개씩 추출한다. '
            'summary에는 전체 고객 반응과 주요 장단점을 한국어로 요약하고, '
            'suggestions에는 부정 리뷰를 바탕으로 제품 개선 제안을 한국어로 작성한다. '
            '반드시 JSON 객체만 출력한다. '
            '형식은 {"positive_keywords":["..."],"negative_keywords":["..."],'
            '"summary":"...","suggestions":"..."} 이다.'
        )
        result = self.parse_json(
            self._request(instructions, json.dumps(reviews, ensure_ascii=False))
        )
        if not isinstance(result, dict):
            raise ValueError('키워드/요약 응답은 JSON 객체여야 합니다.')
        return result
