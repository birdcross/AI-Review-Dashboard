import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.analytics import recent_negative_alert
from app.dashboard import dashboard
from app.evaluation import sample_for_review, evaluate_review_file, run_prompt_ab
from app.offline_sentiment import analyze, detect_language
from app.storage import JsonlStorage


class AddedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.storage = JsonlStorage(
            self.base / 'raw.jsonl',
            self.base / 'clean.jsonl',
            self.base / 'extracts.jsonl',
        )
        seed_rows = [
            ('2026-07-20', 'This product works great and I love it', 'positive', 5),
            ('2026-07-22', 'Good sound and easy to use', 'positive', 5),
            ('2026-07-25', 'It is okay, nothing special', 'neutral', 3),
            ('2026-07-27', 'Battery is fine', 'neutral', 3),
            ('2026-07-29', 'Bad connection problem', 'negative', 2),
            ('2026-08-10', '좋고 사용하기 편리합니다', 'positive', 5),
            ('2026-08-15', '괜찮지만 특별한 점은 없습니다', 'neutral', 3),
            ('2026-08-20', '연결 끊김이 자주 발생하고 불편합니다', 'negative', 1),
            ('2026-08-25', '배터리 충전 문제가 있고 실망입니다', 'negative', 2),
            ('2026-09-01', 'This keeps disconnecting and is terrible', 'negative', 1),
        ]
        rows = []
        for i, (date, text, sentiment, rating) in enumerate(seed_rows, 1):
            _, confidence = analyze(text)
            rows.append({
                'id': i,
                'raw_id': i,
                'review_text': text,
                'original_text': '',
                'review_date': date,
                'product': 'Demo',
                'brand': 'DemoBrand',
                'rating': rating,
                'sentiment': sentiment,
                'confidence': confidence,
                'analysis_provider': 'test',
                'analysis_language': detect_language(text),
                'prompt_version': 'test_v1',
            })
        self.storage.write_jsonl(self.storage.clean_path, rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bilingual_offline_sentiment(self):
        self.assertEqual('positive', analyze('정말 좋고 사용하기 편리합니다')[0])
        self.assertEqual('negative', analyze('연결이 계속 끊기고 최악입니다')[0])
        self.assertEqual('positive', analyze('This is excellent and works great')[0])
        self.assertEqual('negative', analyze('Terrible product, it does not work')[0])
        self.assertEqual('ko', detect_language('좋습니다'))
        self.assertEqual('en', detect_language('works great'))

    def test_negative_surge_comparison_and_hypotheses(self):
        alert = recent_negative_alert(
            self.storage.get_clean(), recent_days=30,
            threshold=0.30, increase_threshold=0.10,
        )
        self.assertTrue(alert['warning'])
        self.assertAlmostEqual(0.60, alert['negative_ratio'])
        self.assertAlmostEqual(0.20, alert['previous_negative_ratio'])
        self.assertAlmostEqual(0.40, alert['increase'])
        self.assertTrue(alert['hypotheses'])
        self.assertTrue(alert['follow_up_actions'])

    def test_standalone_html_embeds_charts(self):
        cfg = {
            'visualization': {
                'output_dir': str(self.base / 'out'),
                'font_candidates': ['DejaVu Sans'],
            },
            'alert': {
                'recent_days': 30,
                'negative_ratio_threshold': 0.30,
                'increase_threshold': 0.10,
            },
        }
        result = dashboard(self.storage, cfg, {}, str(self.base / 'out'), 5, True)
        content = Path(result['html']).read_text(encoding='utf-8')
        self.assertIn('data:image/png;base64,', content)
        self.assertNotIn('src="sentiment_distribution.png"', content)

    def test_sampling_and_human_evaluation(self):
        path = self.base / 'review_sample.csv'
        result = sample_for_review(self.storage, str(path), sample_size=9, seed=42)
        self.assertEqual(9, result['sample_size'])
        with path.open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
            fields = rows[0].keys()
        for row in rows:
            row['human_sentiment'] = row['predicted_sentiment']
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        evaluation = evaluate_review_file(path)
        self.assertEqual(1.0, evaluation['accuracy'])

    def test_prompt_ab_records_results_without_overwriting_storage(self):
        path = self.base / 'review_sample.csv'
        sample_for_review(self.storage, str(path), sample_size=9, seed=42)
        with path.open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
            fields = rows[0].keys()
        for row in rows:
            row['human_sentiment'] = row['predicted_sentiment']
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        before = self.storage.clean_path.read_text(encoding='utf-8')

        def fake_analyze_batch(client, batch, prompt_version='v1'):
            truth = {str(r['id']): r['human_sentiment'] for r in rows}
            return [
                {'id': int(r['id']), 'sentiment': truth[str(r['id'])], 'confidence': 0.9}
                for r in batch
            ]

        with patch('app.ai_client.AIClient.analyze_batch', new=fake_analyze_batch):
            result = run_prompt_ab(
                {'ai': {'provider': 'gemini', 'batch_size': 3}},
                str(path), str(self.base / 'ab'), ('v1', 'v2'),
            )

        self.assertTrue(Path(result['comparison_csv']).exists())
        self.assertTrue(Path(result['summary_json']).exists())
        self.assertEqual(1.0, result['versions']['v1']['accuracy'])
        self.assertEqual(before, self.storage.clean_path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
