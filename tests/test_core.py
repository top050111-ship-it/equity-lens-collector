import copy
import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from agents import Report, Finding, run_agents, validate_references
from app import demo, DemoModel
from broker import Toss, LiveQuotes, value_snapshot, analysis_positions, yahoo_symbol
from common import now, evidence, decimal, freshness
from collectors import filter_evidence, rss


def snapshot():
    return {'as_of': now(), 'scope': '주식만', 'accountNo': 'private-identifier',
            'positions': [dict(symbol='AAPL', name='Apple', market='NASDAQ', currency='USD',
                               quantity='2.5', average_price='90', price='100', price_at=now(), price_source='REST'),
                          dict(symbol='005930', name='삼성전자', market='KOSPI', currency='KRW',
                               quantity='10', average_price='9000', price='10000', price_at=now(), price_source='REST')],
            'fx': {'midRate': '1400', 'validUntil': '2099-01-01T00:00:00+00:00'}}


class CoreTests(unittest.TestCase):
    def test_currency_conversion_and_fractional_shares(self):
        s = value_snapshot(snapshot())
        self.assertEqual(decimal(s['calculated_equity_value_krw']), decimal('450000'))
        self.assertAlmostEqual(sum(p['weight_pct'] for p in s['positions']), 100)
        self.assertAlmostEqual(s['positions'][0]['weight_pct'], 350000 / 450000 * 100)

    def test_missing_fx_rejects_incomplete_weights(self):
        s = snapshot()
        s['fx'] = None
        with self.assertRaises(ValueError):
            value_snapshot(s)

    def test_private_fields_not_in_model_payload(self):
        payload = analysis_positions(value_snapshot(snapshot()))
        text = json.dumps(payload)
        for field in ('quantity', 'average_price', 'accountNo', 'value_krw', 'private-identifier'):
            self.assertNotIn(field, text)

    def test_market_mapping_does_not_assume_all_korean_stocks_kospi(self):
        self.assertEqual(yahoo_symbol({'symbol': '123456', 'market': 'KOSDAQ'}), '123456.KQ')
        with self.assertRaises(ValueError):
            yahoo_symbol({'symbol': '123456', 'market': 'KR_ETC'})

    def test_fabricated_source_rejected(self):
        report = Report(summary='x', findings=[Finding(subject='x', claim='x', kind='fact',
                        evidence_ids=['FAKE'], confidence='high', counterpoint='', revisit_when='')],
                        conflicts=[], watchlist=[], limitations=[])
        with self.assertRaises(ValueError):
            validate_references(report, [])

    def test_independent_agents_execute_before_synthesis(self):
        barrier = threading.Barrier(2)
        calls = []
        class Spy(DemoModel):
            def analyze(self, role, payload):
                if role in ('macro', 'portfolio'):
                    barrier.wait(timeout=3)
                if role == 'macro':
                    self_test.assertNotIn('positions', payload)
                    self_test.assertNotIn('portfolio_agent', payload)
                    self_test.assertEqual([s['title'] for s in payload['sources']], ['macro'])
                if role == 'synthesis':
                    self_test.assertEqual(set(calls), {'portfolio', 'macro'})
                calls.append(role)
                return super().analyze(role, payload)
        self_test = self
        a = evidence('test', 'portfolio', 'https://example.com/a', 'x', now())
        b = evidence('test', 'macro', 'https://example.com/b', 'x', now())
        result = run_agents(Spy(), value_snapshot(snapshot()), {'sources': [a], 'warnings': []}, {'sources': [b], 'warnings': []})
        self.assertTrue(result['complete'])
        self.assertEqual(calls[-1], 'synthesis')

    def test_failed_analysis_is_not_presented_as_complete(self):
        class Broken(DemoModel):
            def analyze(self, role, payload):
                if role == 'macro':
                    raise ValueError('bad output')
                return super().analyze(role, payload)
        e = evidence('test', 'x', 'https://example.com/a', 'x', now())
        bundle = {'sources': [e], 'warnings': []}
        result = run_agents(Broken(), value_snapshot(snapshot()), bundle, bundle)
        self.assertFalse(result['complete'])
        self.assertEqual(result['macro']['status'], 'failed')

    def test_dedup_and_old_future_news(self):
        dt = datetime.now(timezone.utc)
        e = evidence('test', 'x', 'https://example.com/a?utm_source=foo', 'x', now(), symbols=['A'])
        duplicate = evidence('test', 'x', 'https://example.com/a', 'x', now(), symbols=['B'])
        old = evidence('test', 'old', 'https://example.com/old', 'x', (dt - timedelta(days=8)).isoformat())
        future = evidence('test', 'future', 'https://example.com/future', 'x', (dt + timedelta(days=1)).isoformat())
        rows = filter_evidence([e, duplicate, old, future], 168)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['symbols'], ['A', 'B'])

    def test_websocket_does_not_replace_newer_rest_price(self):
        live = LiveQuotes(None)
        s = snapshot()
        live.prices['AAPL'] = {'price': '1', 'timestamp': '2000-01-01T00:00:00+00:00'}
        self.assertEqual(live.overlay(s)['positions'][0]['price'], '100')
        self.assertEqual(freshness(None), '시각 미확인')
        self.assertEqual(freshness('2000-01-01T00:00:00+00:00'), '오래된 관측 또는 휴장')

    def test_rss_parser_handles_namespaces_and_atom(self):
        xml = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Example</title><link href="https://example.com/a"/><published>2026-09-16T00:00:00Z</published><summary>Text</summary></entry></feed>'
        with patch('collectors.request', return_value=xml):
            records = rss('test', 'https://example.com/feed')
        self.assertEqual(records[0]['url'], 'https://example.com/a')
        self.assertEqual(records[0]['observed_at'], '2026-09-16T00:00:00+00:00')

    def test_broker_rejects_unlisted_paths(self):
        with patch.dict(os.environ, {'TOSS_CLIENT_ID': 'test', 'TOSS_CLIENT_SECRET': 'test'}):
            toss = Toss()
            with self.assertRaises(ValueError):
                toss.get('/api/v1/orders')

    def test_toss_official_response_shape_and_midrate(self):
        self_test = self
        class Stub(Toss):
            def __init__(self):
                self.account = '1'
            def get(self, path, params=None, account=False):
                if path == '/api/v1/holdings':
                    self_test.assertTrue(account)
                    return {'marketValue': {'amount': {'krw': '350000'}}, 'items': [
                        {'symbol': 'AAPL', 'name': 'Apple', 'currency': 'USD', 'quantity': '2.5',
                         'averagePurchasePrice': '90', 'lastPrice': '99', 'marketValue': {'amount': '247.5'}}]}
                if path == '/api/v1/stocks':
                    return [{'symbol': 'AAPL', 'market': 'NASDAQ'}]
                if path == '/api/v1/prices':
                    return [{'symbol': 'AAPL', 'lastPrice': '100', 'timestamp': now(), 'currency': 'USD'}]
                if path == '/api/v1/exchange-rate':
                    self_test.assertEqual(params, {'baseCurrency': 'USD', 'quoteCurrency': 'KRW'})
                    return {'rate': '1410', 'midRate': '1400', 'validUntil': '2099-01-01T00:00:00+00:00'}
                raise AssertionError('unexpected API route')
        s = value_snapshot(Stub().snapshot())
        self.assertEqual(decimal(s['calculated_equity_value_krw']), decimal('350000'))
        self.assertNotIn('account', json.dumps(s).lower())

    def test_demo_requires_no_network(self):
        with tempfile.TemporaryDirectory() as path:
            with patch('broker.request', side_effect=AssertionError('network')), \
                 patch('agents.request', side_effect=AssertionError('network')), \
                 patch('collectors.request', side_effect=AssertionError('network')):
                demo(Path(path))
            report = (Path(path) / 'latest_report.md').read_text()
            self.assertIn('가상 데이터 · AI 미호출', report)
            self.assertIn('3단계 완료', report)


if __name__ == '__main__':
    unittest.main()
