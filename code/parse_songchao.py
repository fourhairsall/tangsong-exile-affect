# -*- coding: utf-8 -*-
"""parse_songchao.py — 解析第二部宋人选本《宋詩鈔》(四庫全書文淵閣本, kanripo KR4h0157)
为与 songshi / yusong 同一 schema: [{author, title, paragraphs:[verse_text]}].

底本与《御選宋詩》(KR4h0143) 同属四庫全書文淵閣本家族、同一 mandoku 数字化管线,
故两选本之间的差异可归因于**选本原则**本身, 而非底本或数字化流程。

文本结构 (与 KR4h0143 不同, 本底本缩进可靠):
  lead>=9  : 编者题衔 (内閣中書舎人吳之振編)      -> skip
  lead==1  : 卷题 / 「作者名+别集名+鈔」集名行 / 作者小传(长行散文) / 序跋
  lead==2  : 诗题
  lead==0  : 诗句
关键: **作者小传必须排除在诗句之外** —— 小传惯叙贬谪、迁转, 若混入会系统性抬高
「贬谪羁旅」与「悲苦孤寂」两维。故以状态机处理: 集名行之后、遇到第一个
诗题行之前的全部 lead==1 长行, 一律判为小传并丢弃。

作者归属: 用全宋诗 (Book1Q84) 作者名集合作真值, 从「集名行」前缀最长匹配;
失败则退到小传首几字; 再失败则查人工核订的别名表 VARIANT / JI_ALIAS。
"""
import glob
import json
import os
import re
import sys

import opencc

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = f"{BASE}/data"
OUTDIR = f"{DATA}/songchao"
cc = opencc.OpenCC("t2s")

# ---- 异体字/异名核订表 ----------------------------------------------------
# 四庫文本用字与 Book1Q84《全宋诗》著录名的差异; 逐条核订, 非自动模糊匹配。
VARIANT = {
    "米黻": "米芾",      # 芾/黻 异体
    "谢翺": "谢翱",      # 翺/翱 异体
    "朱橰": "朱槔",      # 橰/槔 异体
    "晁冲之": "晁冲之",
    "朱子文公": "朱熹",
}
# 集名行未含作者名者, 依别集题名归其作者。仅收**可判定**者 (别集名与作者一一对应,
# 据四庫提要/通行著录); 存疑者一律不列, 使其落入 unresolved 被丢弃, 以免错归污染统计。
JI_ALIAS = {
    "广陵诗钞": "王令",          # 《广陵集》王令
    "卢溪集钞": "王庭珪",        # 《卢溪集》王庭珪
    "漫塘诗钞": "刘宰",          # 《漫塘集》刘宰
    "白石樵唱钞": "林景熙",      # 《白石樵唱》林景熙
    "山民诗钞": "真山民",        # 《山民集》真山民
    "水云诗钞": "汪元量",        # 《水云集》汪元量
    "郑震清隽集钞": "郑震",
    "谢翺晞发集钞": "谢翱",      # 《晞发集》谢翱
    "朱橰玉澜诗钞": "朱槔",      # 《玉澜集》朱槔
    "朱子文公集钞": "朱熹",
    # 欧阳修: 别集题名用谥号「文忠」而非本名,「歐陽文忠詩鈔(上/中/下)」的前缀
    # 「歐陽文忠」≠ 作者名「歐陽修」, 集名行直接前缀匹配失败; 上卷因紧随小传可由
    # bio-prefix 救回, 下卷(无小传)则 resolve_author 返回 None、cur_author 被置空,
    # 整卷静默丢失。故显式归并到「欧阳修」。startswith 匹配可同时覆盖 上/中/下 各卷。
    "歐陽文忠詩鈔": "欧阳修",
    "欧阳文忠诗钞": "欧阳修",
    # 以下三家别集题名与作者对应存疑(如《义丰集》《东臯诗钞》《先天集钞》), 四庫著录
    # 亦有同名异人之例, 故不强行归属 —— 留待逐条核订, 不计入统计。
}

SKIP_RE = re.compile(r"^欽定四庫全書$|^宋詩鈔卷|^內閣中書舍人|^内閣中書舎人|^臣等謹案|^欽定四庫全書總目")

# 集名行: 「作者+别集名+鈔」, 或分卷次的「…詩鈔上/中/下」。后者若只按 endswith('鈔')
# 判定会被漏掉, 其后的全部诗篇将被错归到上一位作者 —— 这是本底本最危险的解析陷阱。
JI_RE = re.compile(r"^.{2,10}(鈔|詩鈔|集鈔|藳鈔|稿鈔)(上|中|下)?$")


def is_ji_line(t):
    """判断是否为集名行 (且不是小传中恰以'鈔'收尾的句子)."""
    if len(t) > 12:
        return False
    if not JI_RE.match(t):
        return False
    # 小传句尾的伪命中: 含常见虚词者剔除
    for w in ("者", "其", "以", "而", "亦", "今", "擇", "入", "所", "之", "也", "矣"):
        if w in t[:-1]:
            return False
    return True


def load_songshi_authors():
    A = set()
    for fp in glob.glob(f"{DATA}/songshi/poet.song.*.json"):
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for e in d:
            a = (e.get("author") or "").strip()
            if a:
                A.add(cc.convert(a))
    return A


AUTHORS = load_songshi_authors()


def norm(s):
    t = cc.convert(s)
    for k, v in VARIANT.items():
        t = t.replace(cc.convert(k), v)
    return t


def resolve_author(ji_line, bio_line):
    """返回 (author, how) ; author 为 None 表示未能归属."""
    ji = norm(ji_line)
    bio = norm(bio_line) if bio_line else ""
    # 1) 集名行前缀最长匹配
    for L in range(min(6, len(ji)), 1, -1):
        if ji[:L] in AUTHORS:
            return ji[:L], "ji-prefix"
    # 2) 小传首几字最长匹配 (小传例:「王禹偁字元之…」)
    for L in range(min(6, len(bio)), 1, -1):
        if bio[:L] in AUTHORS:
            return bio[:L], "bio-prefix"
    # 3) 人工核订别名
    for k, v in JI_ALIAS.items():
        if ji.startswith(cc.convert(k)) or cc.convert(k) in ji:
            return cc.convert(v), "alias"
    for k, v in VARIANT.items():
        if ji.startswith(cc.convert(k)):
            return cc.convert(v), "variant"
    return None, "unresolved"


def main():
    files = sorted(glob.glob(f"{OUTDIR}/KR4h0157_*.txt"))
    poems = []
    unresolved = []
    ji_log = {}
    cur_author = None
    pending_ji = None          # 集名行(待与小传一起定作者)
    in_bio = False             # 集名行之后、首个诗题之前 = 小传区
    cur_title = None
    body = []
    n_verse_lines = 0
    n_bio_lines = 0
    n_ji = 0

    def commit():
        nonlocal body, cur_title
        if cur_author is not None and body:
            poems.append({"author": cur_author, "title": cur_title or "",
                          "paragraphs": ["".join(body)]})
        body = []

    for fp in files:
        juan = ""
        raw = open(fp, encoding="utf-8").read()
        m = re.search(r"#\+PROPERTY: JUAN (.+)", raw)
        if m:
            juan = m.group(1).strip()
        if juan in ("提要",):          # 四庫提要不是选本正文
            continue
        for line in raw.split("\n"):
            if not line.strip() or line.startswith("#") or line.startswith("<pb:"):
                continue
            t = line.strip().rstrip("¶").strip()
            if not t or SKIP_RE.search(t):
                continue
            lead = len(line) - len(line.lstrip("\u3000"))

            if lead >= 3:                      # 题衔/夹注
                continue

            if lead == 1:
                if is_ji_line(t):
                    # 集名行: 提交上一首, 转小传态。凡集名行自带作者名者立即定作者,
                    # 以免「上/中/下」续卷(无新小传)导致后续诗篇全部丢失。
                    commit()
                    n_ji += 1
                    pending_ji = t
                    in_bio = True
                    cur_title = None
                    a, how = resolve_author(t, "")
                    cur_author = a
                    ji_log[t] = (a, how)
                    # 安全网: 集名行被识别却无法归属时, 必须显式登记, 不得静默丢弃。
                    # (此前 欧阳修「下」卷即因此被整卷静默丢失而未入 unresolved 清单。)
                    if a is None:
                        unresolved.append({"ji": t, "bio_head": "", "juan": juan,
                                           "note": "ji-line recognized but unresolved "
                                                   "(no following bio to rescue)"})
                    continue
                if in_bio:
                    # 小传 (长行散文) -> 丢弃; 但借其首句定作者
                    n_bio_lines += 1
                    if cur_author is None and pending_ji is not None:
                        a, how = resolve_author(pending_ji, t)
                        if a:
                            cur_author = a
                            ji_log[pending_ji] = (a, how)
                        else:
                            unresolved.append({"ji": pending_ji, "bio_head": t[:20],
                                               "juan": juan})
                        pending_ji = None
                    continue
                # 既非集名行、又不在小传态: 视作诗题(部分诗题在 lead==1)
                commit()
                cur_title = norm(t)
                in_bio = False
                continue

            if lead == 2:
                commit()
                if len(t) <= 30:
                    cur_title = norm(t)
                else:                          # 异常长行: 按诗句处理
                    body.append(norm(t))
                in_bio = False
                continue

            # lead == 0 : 诗句 (小传区内的 lead0 极少, 判为正文行)
            in_bio = False
            if cur_author is not None:
                body.append(norm(t))
                n_verse_lines += 1

    commit()

    # ---- 统计 ----
    from collections import Counter
    cnt = Counter(p["author"] for p in poems)
    chars = Counter()
    for p in poems:
        chars[p["author"]] += len(p["paragraphs"][0])
    summary = {
        "source": "kanripo KR4h0157 宋詩鈔 (四庫全書文淵閣本, 清·吳之振編)",
        "n_poems": len(poems),
        "n_authors": len(cnt),
        "n_ji_lines": n_ji,
        "n_verse_lines": n_verse_lines,
        "n_bio_lines_discarded": n_bio_lines,
        "total_verse_chars": sum(chars.values()),
        "top_authors": cnt.most_common(20),
        "authors_total_chars": {a: chars[a] for a in chars},
        "unresolved_ji": unresolved,
        "ji_resolution": {k: list(v) for k, v in ji_log.items()},
    }
    with open(f"{OUTDIR}/poet.song_songchao.json", "w", encoding="utf-8") as f:
        json.dump(poems, f, ensure_ascii=False)
    with open(f"{BASE}/results_parse_songchao.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"poems={len(poems)}  authors={len(cnt)}  verse_lines={n_verse_lines}  "
          f"verse_chars={sum(chars.values())}  bio_lines_discarded={n_bio_lines}")
    print(f"unresolved 集名行: {len(unresolved)}")
    for u in unresolved[:20]:
        print("   !", u["ji"], "||", u["bio_head"])
    print("top authors:", cnt.most_common(15))


if __name__ == "__main__":
    main()
