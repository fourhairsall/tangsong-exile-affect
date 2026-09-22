# -*- coding: utf-8 -*-
"""Corpus-induced classical-Chinese emotion lexicon (Phase 1).
Learned via PPMI+SVD character embeddings + M1 seed propagation,
with a neutral-contrast double gate (specificity AND contrast vs a
curated non-emotional baseline) that removes topic/entity leakage.
Categories identical to M1. Dimensions that fail to induce cleanly
(e.g. the abstract 空幻时空) fall back to the expert M1 seeds and
are flagged in LEX_FALLBACK.
"""
import re as _re

CATS = ['悲苦孤寂', '贬谪羁旅', '旷达闲适', '自然山水', '空幻时空']

LEX_FALLBACK = {'悲苦孤寂': False, '贬谪羁旅': True, '旷达闲适': False, '自然山水': False, '空幻时空': True}

LEX_LEARNED = {
    '悲苦孤寂': ['悲', '魂', '寞', '怨', '哀', '悽', '慘', '零', '哭', '淒', '泣', '惆', '淚', '酸', '别', '猨', '痛', '嗚', '愴', '倍', '凄', '弔', '廻', '啼', '秪', '涕', '笛', '鴈', '咽', '羇', '慟', '漂', '悄', '笳', '損', '惻', '闗', '悴', '顦', '顇', '肺', '冤', '嬋', '憤', '泗', '跎', '蹉', '灞', '慨', '旐', '鵑', '訴', '頽', '凭', '砧', '黯', '徙', '衾', '憔', '躊', '呌'],
    '贬谪羁旅': ['驛', '巴', '賒', '謫', '迢', '鷁', '舸', '岷', '遷', '逐', '流', '羈', '旅', '客', '宦', '遠', '荒', '蠻', '瘴', '殊', '遐', '徼', '裔'],
    '旷达闲适': ['適', '閒', '趣', '違', '性', '曠', '凡', '拙', '忙', '話', '愚', '悟', '豁', '傲', '懶', '愜', '寡', '叟', '乖', '逍', '悅', '默', '恬', '慵', '穩', '悔', '怡', '究', '妄', '誕', '慧', '瓢', '曩', '哦', '嗜', '譬', '囂', '棋', '粗', '蹔', '悶', '訊', '敦', '翛'],
    '自然山水': ['溪', '煙', '碧', '烟', '水', '樹', '晴', '雲', '岸', '塘', '渡', '江', '綠', '洲', '淺', '畔', '澗', '浦', '楓', '潭', '潮', '渚', '澹', '蘆', '漁', '帆', '漲', '汀', '谿', '淨', '嶼', '棹', '嵐', '灘', '暝', '鷗', '苔', '岫', '艇', '島', '遶', '堤', '篁', '逕', '漾', '瀨', '嶂', '靄', '泠', '暎', '溜', '砌', '浸', '濛', '杪', '邉', '冉', '檣', '杉', '渺', '潯', '塢', '剡', '娟', '瀑', '磯', '荻', '逈', '唳', '灣', '捲', '逗', '湍', '隈', '曛', '颼', '苕', '霾', '瀉', '幔', '菱', '紗', '莎', '蕖', '巘', '檜', '葭', '漪', '嫩', '蘚', '溶', '紋', '漱', '籜'],
    '空幻时空': ['空', '幻', '夢', '影', '塵', '虛', '浮', '逝', '昔', '古', '千', '秋', '萬', '古', '須', '臾', '幻', '影', '電', '光', '隙', '駒', '陰'],
}

NET = ('旷达闲适', '悲苦孤寂')

def count_cat(text):
    return {c: sum(text.count(w) for w in ws) for c, ws in LEX_LEARNED.items()}

def norm(cat_raw, n_char):
    if not n_char: return {k: 0.0 for k in cat_raw}
    return {k: round(v / n_char * 1000, 3) for k, v in cat_raw.items()}

def net_index(cat_per1k):
    return round(cat_per1k[NET[0]] - cat_per1k[NET[1]], 3)

# --- P0 punctuation-normalisation helpers (mirror lexicon.py) -----------------
# The lexicon is counted on the ORIGINAL text (punctuation never matches a
# lexicon entry and would only create spurious adjacent bigrams if stripped);
# only the DENOMINATOR switches to content-character count so that full
# collections (with ~14-16% CBETA punctuation) and anthologies (near-zero
# punctuation) are compared on the same footing.
_PUNCT_RE = _re.compile(
    r"[ \t\n\r　，。、！？；：·…—～〜「」『』“”‘’（）()【】《》〈〉<>［］\[\]{}|／/\\*+=_~`•°※◎◇◆■□▼▲●○〃]"
)

def strip_punct(s):
    """Remove CJK + ASCII punctuation/whitespace from a string."""
    return _PUNCT_RE.sub("", s or "")

def content_len(text):
    """Number of content (non-punctuation) characters in a single text."""
    return len(strip_punct(text))

def cat_per1k(texts):
    """Per-1000-CONTENT-char category rates; punctuation excluded from denom.

    texts : iterable of poem-body strings (or a single string).
    Returns (per1k_dict, content_char_count).
    """
    if isinstance(texts, str):
        texts = [texts]
    n = sum(content_len(t) for t in texts)
    if n == 0:
        return {k: 0.0 for k in CATS}, 0
    joined = "".join(texts)
    return norm(count_cat(joined), n), n

def net_per1k(texts):
    """Net-Mind Index (per 1000 content chars) and content char count."""
    per, n = cat_per1k(texts)
    return net_index(per), n
