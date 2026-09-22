# -*- coding: utf-8 -*-
"""Transparent emotion lexicon for classical Chinese literary texts.
All entries in simplified Chinese; caller must normalize input text to simplified
(traditional->simplified via OpenCC) before counting. Categories are mutually
illustrative, not exhaustive; every count is reported with this caveat."""

LEX = {
    "悲苦孤寂": list("悲傷愁怨苦孤寂獨淒慘哀痛恨憂戚惘愴涕淚殤殁愍悴惸慼黯销魂") +
                ["断肠","凄凉","寂寥","孤危","憔悴","销魂","悲秋","苦辛","酸辛","辛酸"],
    "贬谪羁旅": list("貶謫遷逐流羈旅客宦遠荒蠻瘴殊遐徼陬裔") +
                ["去国","去國","天涯","海角","殊方","投荒","远谪","遠謫","迁客","遷客","楚客","南荒","炎荒","瘴江","蛮烟","蠻煙"],
    "旷达闲适": list("閒閑適達曠豁放醉嘯傲悠淡泊樂欣悅遣傲兀疏狂從容恬") +
                ["旷达","曠達","闲适","閒適","放达","放達","自適","自适","陶然","悠然","忘机","忘機","委运","委運","随遇","隨遇","达观","達觀","适意","適意"],
    "自然山水": list("山水風月雲林泉石竹松江河湖海花鳥溪巖嵐汀洲嶼峯峰崖澗澤涯渚蘋蓼") +
                ["清风","清風","明月","白雲","白云","沧浪","滄浪","烟霞","煙霞","葦岸","渔樵","漁樵","桃源","落英","幽篁"],
    "空幻时空": list("空幻夢影塵虛浮瞬逝昔古千秋萬古須臾泡幻泡影電光隙駒陰") +
                ["浮生","如梦","如夢","梦幻","夢幻","须臾","須臾","万古","萬古","古今","今古","泡影","微尘","微塵","刹那","弹指","过眼","過眼","逆旅","倏忽"],
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
