import math
import re

# English baseline lexicon
POS = {
    'good': 1, 'great': 1.3, 'excellent': 1.7, 'awesome': 1.6, 'amazing': 1.6,
    'love': 1.8, 'loved': 1.8, 'like': 0.7, 'nice': 0.8, 'perfect': 1.7,
    'best': 1.5, 'better': 0.8, 'easy': 0.9, 'comfortable': 1, 'clear': 0.8,
    'happy': 1.2, 'satisfied': 1.3, 'recommend': 1.1, 'recommended': 1.1,
    'worth': 0.8, 'fast': 0.6, 'solid': 0.6, 'beautiful': 0.9,
    'reliable': 1.0, 'convenient': 0.8, 'favorite': 1.2, 'works': 0.5,
    'well': 0.4, 'smooth': 0.7, 'responsive': 0.8, 'fantastic': 1.6,
    'wonderful': 1.6, 'impressed': 1.2, 'pleased': 1.1, 'affordable': 0.7,
}
NEG = {
    'bad': -1.4, 'poor': -1.3, 'terrible': -1.8, 'awful': -1.8,
    'disappointed': -1.5, 'disappointing': -1.5, 'hard': -0.7,
    'difficult': -0.9, 'problem': -1.1, 'problems': -1.1, 'issue': -0.9,
    'issues': -0.9, 'expensive': -0.6, 'worse': -1.2, 'worst': -1.8,
    'broken': -1.6, 'defective': -1.7, 'slow': -0.7, 'frustrating': -1.4,
    'annoying': -1.2, 'bug': -1.0, 'bugs': -1.0, 'disconnect': -1.4,
    'disconnecting': -1.4, 'drops': -1.0, 'return': -0.8, 'returned': -0.9,
    'fail': -1.3, 'failed': -1.3, 'failure': -1.4, 'uncomfortable': -1.1,
    'weak': -0.8, 'stuck': -1.0, 'jumpy': -0.8, 'dead': -1.5, 'hate': -1.7,
    'useless': -1.7, 'waste': -1.6, 'overpriced': -1.1, 'flimsy': -0.9,
    'crash': -1.2,
}
NEGATORS = {'not', 'no', 'never', 'dont', "don't", 'doesnt', "doesn't", 'didnt', "didn't", 'cannot', "can't", 'hardly'}
POS_PHRASES = {
    'works great': 1.5, 'works well': 1.1, 'easy to use': 1.2,
    'easy to set up': 1.2, 'easy to install': 1.1, 'very good': 1.3,
    'very easy': 1.0, 'highly recommend': 1.8, 'worth the money': 1.2,
    'happy with': 1.2, 'love this': 1.7, 'love it': 1.7,
    'no complaints': 1.2,
}
NEG_PHRASES = {
    'does not work': -2.0, 'did not work': -2.0, 'stopped working': -2.0,
    'not worth': -1.4, 'too expensive': -1.1, 'waste of money': -2.0,
    'not good': -1.4, 'very disappointed': -1.8, "doesn't work": -2.0,
    'do not recommend': -1.8, 'would not recommend': -1.8,
    'keeps disconnecting': -1.7,
}

# Korean baseline lexicon. This intentionally stays simple: the real AI path is
# preferred, while this baseline makes offline tests explicitly bilingual.
KO_POS = {
    '좋': 1.2, '만족': 1.4, '추천': 1.1, '편리': 0.9, '편하': 0.9,
    '빠르': 0.7, '훌륭': 1.6, '최고': 1.6, '괜찮': 0.8, '예쁘': 0.9,
    '깔끔': 0.8, '잘됨': 1.0, '잘 되': 1.0, '안정': 0.8, '저렴': 0.7,
    '가성비': 1.1, '선명': 0.8, '쉬움': 0.8, '쉽': 0.8,
}
KO_NEG = {
    '나쁘': -1.3, '불만': -1.4, '불편': -1.1, '느리': -0.8,
    '고장': -1.6, '불량': -1.7, '문제': -1.1, '오류': -1.1,
    '끊김': -1.4, '끊기': -1.4, '비싸': -0.8, '실망': -1.5,
    '최악': -1.8, '반품': -1.0, '환불': -1.0, '안됨': -1.7,
    '안 되': -1.7, '못쓰': -1.5, '불안정': -1.2, '답답': -1.0,
    '약함': -0.8, '약하': -0.8,
}
KO_POS_PHRASES = {
    '잘 작동': 1.4, '사용하기 편': 1.2, '마음에 듭': 1.5,
    '마음에 들어': 1.5, '구매 추천': 1.4, '가격 대비': 0.8,
}
KO_NEG_PHRASES = {
    '작동하지 않': -1.9, '연결이 끊': -1.7, '연결 끊': -1.7,
    '추천하지 않': -1.7, '돈 아깝': -1.8, '사용하기 불편': -1.4,
}


def detect_language(text):
    """Return ko/en/mixed/other using a lightweight script-count heuristic."""
    text = text or ''
    ko = len(re.findall(r'[가-힣]', text))
    en = len(re.findall(r'[A-Za-z]', text))
    if ko and en:
        if ko >= en * 2:
            return 'ko'
        if en >= ko * 2:
            return 'en'
        return 'mixed'
    if ko:
        return 'ko'
    if en:
        return 'en'
    return 'other'


def _english_score(low):
    score = 0.0
    hits = 0
    for phrase, weight in POS_PHRASES.items():
        count = low.count(phrase)
        score += weight * count
        hits += count
    for phrase, weight in NEG_PHRASES.items():
        count = low.count(phrase)
        score += weight * count
        hits += count

    toks = re.findall(r"[a-z']+", low)
    for i, token in enumerate(toks):
        if token in POS:
            weight = POS[token]
            if any(x in NEGATORS for x in toks[max(0, i - 3):i]):
                weight = -weight
            score += weight
            hits += 1
        elif token in NEG:
            weight = NEG[token]
            if any(x in NEGATORS for x in toks[max(0, i - 3):i]):
                weight = -weight * 0.7
            score += weight
            hits += 1
    return score, hits, len(toks)


def _korean_score(text):
    score = 0.0
    hits = 0
    for phrase, weight in KO_POS_PHRASES.items():
        count = text.count(phrase)
        score += weight * count
        hits += count
    for phrase, weight in KO_NEG_PHRASES.items():
        count = text.count(phrase)
        score += weight * count
        hits += count
    for stem, weight in KO_POS.items():
        count = text.count(stem)
        score += weight * count
        hits += count
    for stem, weight in KO_NEG.items():
        count = text.count(stem)
        score += weight * count
        hits += count
    # Approximate token count for confidence normalization.
    token_count = max(1, len(re.findall(r'[가-힣A-Za-z0-9]+', text)))
    return score, hits, token_count


def analyze(text):
    """Offline bilingual sentiment baseline for Korean and English reviews."""
    text = text or ''
    low = text.lower()
    en_score, en_hits, en_tokens = _english_score(low)
    ko_score, ko_hits, ko_tokens = _korean_score(text)
    score = en_score + ko_score
    hits = en_hits + ko_hits
    token_count = max(en_tokens, ko_tokens, 1)

    score = score / max(1.0, math.sqrt(token_count / 12))
    sentiment = 'positive' if score >= 0.55 else 'negative' if score <= -0.55 else 'neutral'
    confidence = min(0.97, max(0.55, 0.58 + min(abs(score), 3) * 0.11 + min(hits, 5) * 0.025))
    return sentiment, round(confidence, 3)
