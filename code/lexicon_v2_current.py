# -*- coding: utf-8 -*-
r"""lexicon.py — Transparent emotion lexicon v2 (curated 2026-09-19).

v1 is archived verbatim in lexicon_v1.py. v2 differs from v1 in three ways,
each decided by measured agreement with the blind two-coder gold standard
(452 poems, two rounds; see results_lexicon_v2.json and _lexicon_v2.py):

1. SCRIPT FIX. v1 entries were mostly traditional while the corpus is
   normalized to simplified (OpenCC t2s), so many v1 single-character entries
   could never match (dead entries). v2 converts every entry to simplified.
   Author-level criterion (41 authors with >=5 gold poems): r 0.374 -> 0.603.
2. AMBIGUITY CURATION. Blind t2s would activate 云 (<-雲), which in simplified
   text also reads "to say"; the single character is replaced by curated
   compounds (暮云 孤云 寒云 春云 秋云 停云 烟云 云霞 云峰 云山).
   乐 (<-樂; joy vs music) is kept as a single character: poetic usage is
   predominantly the joy reading, and dropping it measurably hurts agreement.
   NOTE: v1 also double-counted characters that appeared twice in its lists
   (幻 古 泡 影 傲); v2 deduplicates.
3. REGISTER ABSORPTION. The three registers recovered by residual mining and
   validated by blind dual coding (150-poem study, kappa_w 0.81-0.91;
   results_register.json) are folded into the two affective poles, with
   per-character curation:
   - 幽居闲静 Quiet -> 旷达闲适: 静 禅 苔 钓 隐 亭 栽 莲 畦 眠 (炉 dropped:
     香炉峰 false positive)
   - 思乡飘零 Home  -> 悲苦孤寂: 零 萧 雁 砧 捣 飘零 飘蓬 断蓬 飞蓬 转蓬
     (衣 dropped: generic; 堪 dropped: function word; 蓬 only in compounds:
      蓬莱 false positive)
   - 羁旅行役 Travel -> 悲苦孤寂: 吏 灯 店 舫 猿 驿 帆 仆 僮 (马 dropped:
     generic horse)
   Full-v2 effect (author-level criterion, 41 authors): r 0.374 -> 0.611,
   rho 0.355 -> 0.611; cohort-12 r 0.692 -> 0.794; poem-level pooled Pearson
   0.300 -> 0.456; near-zero miss rate on gold-active poems 0.46 -> 0.29.

All entries in simplified Chinese; caller must normalize input text to
simplified (traditional->simplified via OpenCC) before counting. Categories
are illustrative, not exhaustive; every count is reported with this caveat."""

LEX = {
    "悲苦孤寂": list("悲伤愁怨苦孤寂独凄惨哀痛恨忧戚惘怆涕泪殇殁愍悴惸黯销魂零萧雁砧捣吏灯店舫猿驿帆仆僮") +
                                ["断肠", "凄凉", "寂寥", "孤危", "憔悴", "销魂", "悲秋", "苦辛", "酸辛", "辛酸", "飘零", "飘蓬", "断蓬", "飞蓬", "转蓬"],
    "贬谪羁旅": list("贬谪迁逐流羁旅客宦远荒蛮瘴殊遐徼陬裔") +
                                ["去国", "天涯", "海角", "殊方", "投荒", "远谪", "迁客", "楚客", "南荒", "炎荒", "瘴江", "蛮烟"],
    "旷达闲适": list("闲适达旷豁放醉啸傲悠淡泊乐欣悦遣兀疏狂从容恬静禅苔钓隐亭栽莲畦眠") +
                                ["旷达", "闲适", "放达", "自适", "陶然", "悠然", "忘机", "委运", "随遇", "达观", "适意"],
    "自然山水": list("山水风月林泉石竹松江河湖海花鸟溪岩岚汀洲屿峰崖涧泽涯渚苹蓼") +
                                ["清风", "明月", "白云", "沧浪", "烟霞", "苇岸", "渔樵", "桃源", "落英", "幽篁", "暮云", "孤云", "寒云", "春云", "秋云", "停云", "烟云", "云霞", "云峰", "云山"],
    "空幻时空": list("空幻梦影尘虚浮瞬逝昔古千秋万须臾泡电光隙驹阴") +
                                ["浮生", "如梦", "梦幻", "须臾", "万古", "古今", "今古", "泡影", "微尘", "刹那", "弹指", "过眼", "逆旅", "倏忽"],
}
# 净心指数 = 旷达闲适 - 悲苦孤寂 (per 1000 chars)
NET = ("旷达闲适", "悲苦孤寂")
CATS = list(LEX.keys())

def count_cat(text):
    """Return raw count of each category in `text`."""
    return {c: sum(text.count(w) for w in words) for c, words in LEX.items()}

def norm(cat_raw, n_char):
    """Per-1000-char normalization."""
    if not n_char:
        return {k: 0.0 for k in cat_raw}
    return {k: round(v / n_char * 1000, 3) for k, v in cat_raw.items()}

def net_index(cat_per1k):
    return round(cat_per1k[NET[0]] - cat_per1k[NET[1]], 3)
