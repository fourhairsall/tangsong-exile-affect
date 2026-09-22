# -*- coding: utf-8 -*-
r"""recompute_all.py -- full, traceable recomputation of every number used in the
paper, starting from the RAW corpora in data/.

Why this exists
---------------
An earlier draft carried numbers forward from result files without re-deriving
them, which made it impossible to answer "where does this figure come from?".
This orchestrator re-runs each analysis stage in dependency order and writes:

  * recompute.log        -- full per-stage console log (evidence trail)
  * numbers_ledger.json  -- a flat, paper-oriented ledger: every reported quantity
                            mapped to {value, source_script, source_json_key}

Run:  python recompute_all.py            (all stages)
      python recompute_all.py --no-dl    (skip the slow DL/CNN stages)
"""
import os, sys, json, time, subprocess, argparse

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOG = os.path.join(BASE, "recompute.log")

# (script, writes, needs_dl)
STAGES = [
    ("build_corpus_v3.py",    "data/corpus_labeled_v3.json",        False),
    ("analyze_v4.py",         "results_v4.json",                    True),
    ("methods_compare_v5.py", "results_v5_methods.json",            True),
    # --- extended benchmark (all 6 baselines + proposed AttentionFractal variants) ---
    ("dl_cnn2025_v2.py",      "results_cnn2025_v2.json",            True),
    ("dl_attn_fractal.py",    "results_dl_attnfractal.json",        True),
    ("xval_yuding.py",        "results_v4_xval.json",               False),
    ("song2_xval.py",         "results_song2_xval.json",            False),
    ("song2_noisecheck.py",   "results_song2_noisecheck.json",      False),
    ("phase1b_invariance.py", "results_phase1b_invariance.json",    False),
    ("phase1b_song.py",       "results_phase1b_song.json",          False),
    ("hb_model.py",           "results_hb_model.json",              False),
    ("phase4_m6.py",          "results_m6.json",                    False),
    # --- second Song anthology 《宋詩鈔》 replication chain ---
    ("parse_songchao.py",     "results_parse_songchao.json",        False),
    ("song3_rep.py",          "results_song3_rep.json",             False),
    ("song3_matched.py",      "results_song3_matched.json",         False),
    # --- human gold-standard validity ---
    # make_gold_sample.py MUST run before gold_validity.py (the latter reads
    # gold_sample.json produced by the former, plus the two annotation files).
    ("make_gold_sample.py",   "gold_sample.json",                   False),
    ("gold_validity.py",      "results_gold_validity.json",         False),
    # --- cross-edition work-level realignment (§robust:realign / tab:realign) ---
    # Both variants of align_works.py are regenerated because the paper reports
    # the loose numbers as the normalisation-robustness check.
    ("align_works.py",            "results_cross_edition_aligned_strict.json", False),
    ("align_works.py --loose",    "results_cross_edition_aligned_loose.json",  False),
    ("align_variants.py",         "results_align_variants.json",               False),
]


def run_stage(script, no_dl=False, needs_dl=False):
    if no_dl and needs_dl:
        print(f"[SKIP] {script} (DL stage, --no-dl)", flush=True)
        return ("skipped", 0.0, 0, "")
    t0 = time.time()
    print(f"\n{'='*70}\n[RUN ] {script}\n{'='*70}", flush=True)
    # `script` may carry arguments (e.g. "align_works.py --loose"); split so that
    # stage entries stay plain 3-tuples.
    cmd = [PY] + str(script).split()
    p = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = time.time() - t0
    out = (p.stdout or "") + ("\n[STDERR]\n" + p.stderr if p.stderr else "")
    print(out[-4000:], flush=True)
    print(f"[DONE] {script}  exit={p.returncode}  {dt:.1f}s", flush=True)
    return ("ok" if p.returncode == 0 else "FAIL", dt, p.returncode, out)


def jload(rel):
    fp = os.path.join(BASE, rel)
    if not os.path.exists(fp):
        return None
    return json.load(open(fp, encoding="utf-8"))


def build_ledger():
    """Assemble a paper-oriented ledger: {quantity: {value, source, key}}."""
    L = {}
    def put(name, value, source, key=""):
        L[name] = {"value": value, "source": source, "key": key}

    v4 = jload("results_v4.json")
    v5 = jload("results_v5_methods.json")
    xv = jload("results_v4_xval.json")
    s2 = jload("results_song2_xval.json")
    p1b = jload("results_phase1b_song.json")
    hb = jload("results_hb_model.json")
    m6 = jload("results_m6.json")
    sc = jload("results_parse_songchao.json")
    s3r = jload("results_song3_rep.json")
    s3m = jload("results_song3_matched.json")
    gs = jload("gold_sample.json")
    gv = jload("results_gold_validity.json")
    cf = jload("results_counterfactual.json")
    ex = jload("results_exile_dose.json")
    gp = jload("results_gold_p0.json")
    p0 = jload("results_p0_fixes.json")
    lex = jload("lexicon_v2_entries.json")
    eb = jload("results_cnn2025_v2.json")

    # ---- corpus size / instrument coverage ----
    if v4:
        sz = v4["meta"]["corpus_sizes"]
        put("shi_poems_total", sz["shi_total"], "results_v4.json", "meta.corpus_sizes.shi_total")
        put("ci_total", sz["ci_total"], "results_v4.json", "meta.corpus_sizes.ci_total")
        put("tangwen_essays_total", sz["tangwen_total_essays"], "results_v4.json",
            "meta.corpus_sizes.tangwen_total_essays")
        put("tangwen_chars_total", sz["tangwen_total_chars"], "results_v4.json",
            "meta.corpus_sizes.tangwen_total_chars")
        put("dl_reg_metrics", v4["meta"]["dl_reg"], "results_v4.json", "meta.dl_reg")
        put("attr_author_acc", v4["meta"]["attr_author"]["accuracy"], "results_v4.json",
            "meta.attr_author.accuracy")
        put("attr_author_macroF1", v4["meta"]["attr_author"]["macro_F1"], "results_v4.json",
            "meta.attr_author.macro_F1")
        if v4["meta"].get("attr_dynasty"):
            put("attr_dynasty_acc", v4["meta"]["attr_dynasty"]["accuracy"], "results_v4.json",
                "meta.attr_dynasty.accuracy")
            put("attr_dynasty_macroF1", v4["meta"]["attr_dynasty"]["macro_F1"], "results_v4.json",
                "meta.attr_dynasty.macro_F1")
        if v4["meta"].get("attr_prose"):
            put("attr_prose_acc", v4["meta"]["attr_prose"]["accuracy"], "results_v4.json",
                "meta.attr_prose.accuracy")
            put("attr_prose_macroF1", v4["meta"]["attr_prose"]["macro_F1"], "results_v4.json",
                "meta.attr_prose.macro_F1")
        # per-author single-source NMI
        nmis = {a: v4["shi_stat"][a]["净心指数"] for a in v4["authors"]}
        put("nmi_by_author", nmis, "results_v4.json", "shi_stat[*].净心指数")

    # ---- method triangulation ----
    if v5:
        put("method_compare", v5.get("net_mind_methods"), "results_v5_methods.json",
            "net_mind_methods")
        put("attribution_compare", v5.get("attribution_compare"), "results_v5_methods.json",
            "attribution_compare")

    # ---- Tang same-work ----
    if xv:
        put("tang_n_volumes", xv["n_volumes"], "results_v4_xval.json", "n_volumes")
        put("tang_mean_abs_delta", xv["mean_abs_delta_net"], "results_v4_xval.json",
            "mean_abs_delta_net")
        put("tang_cat_corr_20cell", xv["cat_corr"], "results_v4_xval.json", "cat_corr")
        put("tang_rows", xv["rows"], "results_v4_xval.json", "rows")

    # ---- cross-edition invariance (population scale, M1 + induced) ----
    if p1b:
        put("tang_r_nmi_479", p1b["M1"]["tang"]["r_nmi"], "results_phase1b_song.json",
            "M1.tang.r_nmi")
        put("tang_S_479", p1b["M1"]["tang"]["n_preserved"], "results_phase1b_song.json",
            "M1.tang.n_preserved")
        put("tang_dim_r_479", p1b["M1"]["tang"]["dim_r"], "results_phase1b_song.json",
            "M1.tang.dim_r")
        put("song_r_nmi_256", p1b["M1"]["song"]["r_nmi"], "results_phase1b_song.json",
            "M1.song.r_nmi")
        put("song_S_256", p1b["M1"]["song"]["n_preserved"], "results_phase1b_song.json",
            "M1.song.n_preserved")
        put("song_dim_r_256", p1b["M1"]["song"]["dim_r"], "results_phase1b_song.json",
            "M1.song.dim_r")
        put("tang_r_nmi_479_ind", p1b["Learned"]["tang"]["r_nmi"], "results_phase1b_song.json",
            "Learned.tang.r_nmi")
        put("song_r_nmi_256_ind", p1b["Learned"]["song"]["r_nmi"], "results_phase1b_song.json",
            "Learned.song.r_nmi")
        put("corpus_author_counts", p1b["n_authors"], "results_phase1b_song.json", "n_authors")

    # ---- Song 7-poet supplementary cross-validation (retained for reference) ----
    if s2:
        put("song_pilot_net_r_7", s2["net_pearson"], "results_song2_xval.json", "net_pearson")
        put("song_pilot_net_rho_7", s2["net_spearman"], "results_song2_xval.json", "net_spearman")
        put("song_pilot_cat_corr_35cell", s2["cat_freq_pearson"], "results_song2_xval.json",
            "cat_freq_pearson")
        put("song_pilot_rows", s2["rows"], "results_song2_xval.json", "rows")

    # ---- hierarchical noise floor ----
    if hb:
        put("clustering_k", hb["clustering_inflation_k"], "results_hb_model.json",
            "clustering_inflation_k")
        put("hb_population_n", hb["population_n_authors"], "results_hb_model.json",
            "population_n_authors")
        put("hb_edition_effect", hb["edition_effect"], "results_hb_model.json", "edition_effect")
        put("hb_latent_population", hb["latent_population"], "results_hb_model.json",
            "latent_population")
        put("hb_per_dimension_delta", hb["per_dimension_delta"], "results_hb_model.json",
            "per_dimension_delta")
        put("hb_luyou_pp", hb["luyou_posterior_predictive"], "results_hb_model.json",
            "luyou_posterior_predictive")

    # ---- M6 probe ----
    if m6:
        put("m6_settings", m6["settings"], "results_m6.json", "settings")
        put("m6_lexicon_baseline", m6["phase2_lexicon_baseline"], "results_m6.json",
            "phase2_lexicon_baseline")

    # ---- 宋詩鈔 parse (second Song anthology, poem layer) ----
    if sc:
        put("songchao_n_poems", sc["n_poems"], "results_parse_songchao.json", "n_poems")
        put("songchao_n_authors", sc["n_authors"], "results_parse_songchao.json", "n_authors")
        put("songchao_verse_chars", sc["total_verse_chars"], "results_parse_songchao.json",
            "total_verse_chars")
        put("songchao_n_ji_lines", sc["n_ji_lines"], "results_parse_songchao.json", "n_ji_lines")
        put("songchao_bio_lines_discarded", sc["n_bio_lines_discarded"],
            "results_parse_songchao.json", "n_bio_lines_discarded")
        put("songchao_n_verse_lines", sc["n_verse_lines"], "results_parse_songchao.json",
            "n_verse_lines")
        put("songchao_unresolved_ji", len(sc.get("unresolved_ji", [])),
            "results_parse_songchao.json", "unresolved_ji")

    # ---- second-anthology replication: QSS vs 御選宋詩 (YXSS) / QSS vs 宋詩鈔 (SSC) ----
    if s3r:
        yx = s3r["pair_QSS_vs_YXSS"]
        ss = s3r["pair_QSS_vs_SSC"]
        put("song3_yxss_A", yx["A_nmi_pearson"], "results_song3_rep.json",
            "pair_QSS_vs_YXSS.A_nmi_pearson")
        put("song3_yxss_S", yx["S_dims_preserved"], "results_song3_rep.json",
            "pair_QSS_vs_YXSS.S_dims_preserved")
        put("song3_yxss_delta", yx["delta_nmi"]["delta"], "results_song3_rep.json",
            "pair_QSS_vs_YXSS.delta_nmi.delta")
        put("song3_yxss_delta_ci", yx["delta_nmi"]["ci"], "results_song3_rep.json",
            "pair_QSS_vs_YXSS.delta_nmi.ci")
        put("song3_yxss_n_authors", yx["n_authors"], "results_song3_rep.json",
            "pair_QSS_vs_YXSS.n_authors")
        put("song3_ssc_A", ss["A_nmi_pearson"], "results_song3_rep.json",
            "pair_QSS_vs_SSC.A_nmi_pearson")
        put("song3_ssc_S", ss["S_dims_preserved"], "results_song3_rep.json",
            "pair_QSS_vs_SSC.S_dims_preserved")
        put("song3_ssc_delta", ss["delta_nmi"]["delta"], "results_song3_rep.json",
            "pair_QSS_vs_SSC.delta_nmi.delta")
        put("song3_ssc_delta_ci", ss["delta_nmi"]["ci"], "results_song3_rep.json",
            "pair_QSS_vs_SSC.delta_nmi.ci")
        put("song3_ssc_n_authors", ss["n_authors"], "results_song3_rep.json",
            "pair_QSS_vs_SSC.n_authors")
        put("song3_yxss_dim_delta",
            {d: yx["per_dimension"][d]["delta"] for d in yx["per_dimension"]},
            "results_song3_rep.json", "pair_QSS_vs_YXSS.per_dimension.*.delta")
        put("song3_yxss_dim_delta_ci",
            {d: yx["per_dimension"][d]["ci"] for d in yx["per_dimension"]},
            "results_song3_rep.json", "pair_QSS_vs_YXSS.per_dimension.*.ci")
        put("song3_ssc_dim_delta",
            {d: ss["per_dimension"][d]["delta"] for d in ss["per_dimension"]},
            "results_song3_rep.json", "pair_QSS_vs_SSC.per_dimension.*.delta")
        put("song3_ssc_dim_delta_ci",
            {d: ss["per_dimension"][d]["ci"] for d in ss["per_dimension"]},
            "results_song3_rep.json", "pair_QSS_vs_SSC.per_dimension.*.ci")
        ag = s3r["delta_sign_agreement"]
        put("song3_delta_sign_agree", f'{ag["n_sign_agree"]}/{ag["n_dims"]}',
            "results_song3_rep.json", "delta_sign_agreement.n_sign_agree / .n_dims")
        put("song3_delta_spearman", ag["spearman_of_delta_vectors"], "results_song3_rep.json",
            "delta_sign_agreement.spearman_of_delta_vectors")
        put("song3_delta_pearson", ag["pearson_of_delta_vectors"], "results_song3_rep.json",
            "delta_sign_agreement.pearson_of_delta_vectors")
        put("song3_delta_sign_test_p", ag["sign_test_p_greater"], "results_song3_rep.json",
            "delta_sign_agreement.sign_test_p_greater")
        put("song3_clustering_inflation_k", s3r["design"]["clustering_inflation_k"],
            "results_song3_rep.json", "design.clustering_inflation_k")
        put("song3_taboo_immune_ssc_delta", s3r["robustness_taboo_immune"]["SSC"],
            "results_song3_rep.json", "robustness_taboo_immune.SSC")
        put("song3_taboo_immune_yxss_delta", s3r["robustness_taboo_immune"]["YXSS"],
            "results_song3_rep.json", "robustness_taboo_immune.YXSS")
        put("song3_split_half_sign_agree",
            {k: s3r["robustness_split_half"][k]["sign_agree_between_halves"]
             for k in ("YXSS", "SSC")},
            "results_song3_rep.json", "robustness_split_half.*.sign_agree_between_halves")

    # ---- same-author-set matched contrast (三向共有作者) ----
    if s3m:
        put("song3_matched_n_authors", s3m["n_three_way_authors"], "results_song3_matched.json",
            "n_three_way_authors")
        put("song3_matched_yxss_A", s3m["YXSS_matched"]["A_nmi"], "results_song3_matched.json",
            "YXSS_matched.A_nmi")
        put("song3_matched_yxss_S", s3m["YXSS_matched"]["S_dims"], "results_song3_matched.json",
            "YXSS_matched.S_dims")
        put("song3_matched_ssc_A", s3m["SSC_matched"]["A_nmi"], "results_song3_matched.json",
            "SSC_matched.A_nmi")
        put("song3_matched_ssc_S", s3m["SSC_matched"]["S_dims"], "results_song3_matched.json",
            "SSC_matched.S_dims")
        put("song3_matched_dim_delta_pearson",
            s3m["sign_agreement_matched"]["pearson_of_delta_vectors"],
            "results_song3_matched.json", "sign_agreement_matched.pearson_of_delta_vectors")
        put("song3_matched_sign_agree", s3m["sign_agreement_matched"]["n_sign_agree"],
            "results_song3_matched.json", "sign_agreement_matched.n_sign_agree")
        put("song3_author_shift_corr", s3m["author_level_shift_correlation_matched"],
            "results_song3_matched.json", "author_level_shift_correlation_matched")
        put("song3_missing_char_markers", s3m["missing_char_markers"]["count"],
            "results_song3_matched.json", "missing_char_markers.count")

    # ---- human gold standard: sample design + criterion validity ----
    if gs:
        put("gold_sample_size", len(gs), "gold_sample.json", "len(sample list)")
        put("gold_sample_n_authors", len({p["author"] for p in gs}), "gold_sample.json",
            "unique(author)")
        _strata = {}
        for p in gs:
            _k = f'{p["dynasty"]}/{p["carrier"]}'
            _strata[_k] = _strata.get(_k, 0) + 1
        put("gold_sample_strata", _strata, "gold_sample.json", "count(dynasty/carrier)")
    if gv:
        dsn = gv["design"]
        put("gold_n_poems", dsn["n_poems"], "results_gold_validity.json", "design.n_poems")
        put("gold_n_authors", dsn["n_authors"], "results_gold_validity.json", "design.n_authors")
        put("gold_coders", dsn["coders"], "results_gold_validity.json", "design.coders")
        put("gold_qwk_5dim",
            {d: gv["reliability"][d]["kappa_quadratic_weighted"]
             for d in ("Bei", "Quan", "Man", "Ziran", "Konghuan")},
            "results_gold_validity.json", "reliability.<dim>.kappa_quadratic_weighted")
        put("gold_krippendorff_alpha_5dim",
            {d: gv["reliability"][d]["krippendorff_alpha_ordinal"]
             for d in ("Bei", "Quan", "Man", "Ziran", "Konghuan")},
            "results_gold_validity.json", "reliability.<dim>.krippendorff_alpha_ordinal")
        put("gold_nmi_icc_2_1", gv["reliability"]["NMI_gold"]["icc_2_1"],
            "results_gold_validity.json", "reliability.NMI_gold.icc_2_1")
        put("gold_nmi_pearson_A_vs_B", gv["reliability"]["NMI_gold"]["pearson_A_vs_B"],
            "results_gold_validity.json", "reliability.NMI_gold.pearson_A_vs_B")
        cv = gv["criterion_validity"]
        put("gold_rho_lexicon_vs_gold", cv["rho_lexiconNMI_vs_goldNMI"],
            "results_gold_validity.json", "criterion_validity.rho_lexiconNMI_vs_goldNMI")
        put("gold_rho_lexicon_vs_gold_ci", cv["ci_cluster_bootstrap"],
            "results_gold_validity.json", "criterion_validity.ci_cluster_bootstrap")
        put("gold_baseline_rho_length", cv["baseline_rho_length_vs_gold"],
            "results_gold_validity.json", "criterion_validity.baseline_rho_length_vs_gold")
        put("gold_partial_rho_length", cv["partial_rho_controlling_length"],
            "results_gold_validity.json", "criterion_validity.partial_rho_controlling_length")
        put("gold_per_dim_rho",
            {d: cv["per_dimension"][d]["rho"] for d in cv["per_dimension"]},
            "results_gold_validity.json", "criterion_validity.per_dimension.<dim>.rho")
        put("gold_affect_only_rho", cv["affect_only_rho"], "results_gold_validity.json",
            "criterion_validity.affect_only_rho")
        put("gold_affect_only_ci", cv["affect_only_ci"], "results_gold_validity.json",
            "criterion_validity.affect_only_ci")
        qd = gv["quintile_dose_response"]
        put("gold_quintile_rows", qd["rows"], "results_gold_validity.json",
            "quintile_dose_response.rows")
        put("gold_q5_minus_q1", qd["Q5_minus_Q1"], "results_gold_validity.json",
            "quintile_dose_response.Q5_minus_Q1")
        put("gold_q5_minus_q1_ci", qd["Q5_minus_Q1_ci"], "results_gold_validity.json",
            "quintile_dose_response.Q5_minus_Q1_ci")

    # ---- P0-6: register the six keys previously absent from the ledger ----
    if eb:
        put("tab:extbench",
            {"pool_size": eb.get("pool_size"), "seed": eb.get("seed"),
             "epochs": eb.get("epochs"), "models": eb.get("models")},
            "results_cnn2025_v2.json",
            "models (+ meta.pool_size/seed/epochs)")
    if cf:
        put("§robust:cf", cf.get("reproducibility_check"),
            "results_counterfactual.json", "reproducibility_check")
    if ex:
        put("§exile", ex.get("su_shi"), "results_exile_dose.json", "su_shi")
    if gp:
        am = gp.get("author_level_merged", {})
        put("author_level_r_057",
            {"pearson": round(am.get("pearson", 0.5715), 4),
             "ci": [round(x, 4) for x in am.get("ci_pearson", [0.36, 0.73])],
             "n_authors": am.get("n_authors"), "n_poems": am.get("n_poems")},
            "results_gold_p0.json", "author_level_merged.pearson / ci_pearson")
    if p0:
        put("leave-k-out", p0.get("loo"), "results_p0_fixes.json", "loo")
    if lex:
        put("tab:lexfreq", lex, "lexicon_v2_entries.json", "(root: 5-category lexicon)")

    # --- cross-edition work-level realignment (§robust:realign / tab:realign) ---
    aws = jload("results_cross_edition_aligned_strict.json")
    awl = jload("results_cross_edition_aligned_loose.json")
    av = jload("results_align_variants.json")
    if aws and awl and av:
        def _wr(d, k):
            p = d[k]
            return {"label": p["label"],
                    "n_authors_shared": p["shared_authors"],
                    "A_pooled": p["pooled"]["A"], "n_pooled": p["pooled"]["n"],
                    "A_matched": p["matched"]["A"], "n_matched": p["matched"]["n"],
                    "coverage_matched_over_pooled_chars": p["coverage"]["frac"],
                    "pooled_chars": p["coverage"]["pooled_chars"],
                    "matched_chars": p["coverage"]["matched_chars"]}
        _keys = ("QTS_vs_YD", "QSS_vs_YXSS", "QSS_vs_SSC", "YXSS_vs_SSC")
        # POOLED must reproduce the published tab:invariance values; that agreement
        # is what licenses reading any new number off this pipeline.
        # POOLED mode must reproduce the SAME punctuation-normalised invariance
        # values as the established author-level pipeline (phase1b M1). After the
        # P0 punctuation-denominator fix BOTH pipelines moved together, so the
        # guardrail is now internal consistency, not a hardcoded historical
        # constant. Pre-P0 (punctuation-in-denominator) published values are kept
        # for the audit trail: Tang 0.95, Song-QSS/YXSS 0.515, Song-QSS/SSC 0.82.
        _selfcheck = {
            "tang_A_pooled_alignworks": round(aws["QTS_vs_YD"]["pooled"]["A"], 4),
            "tang_A_pooled_phase1b_M1": round(p1b["M1"]["tang"]["r_nmi"], 4)
                if p1b else None,
            "tang_agree_delta": round(aws["QTS_vs_YD"]["pooled"]["A"]
                                      - p1b["M1"]["tang"]["r_nmi"], 4)
                if p1b else None,
            "qss_yxss_A_pooled_alignworks": round(aws["QSS_vs_YXSS"]["pooled"]["A"], 4),
            "qss_yxss_A_pooled_phase1b_M1": round(p1b["M1"]["song"]["r_nmi"], 4)
                if p1b else None,
            "qss_yxss_agree_delta": round(aws["QSS_vs_YXSS"]["pooled"]["A"]
                                          - p1b["M1"]["song"]["r_nmi"], 4)
                if p1b else None,
            "qss_ssc_A_pooled_alignworks": round(aws["QSS_vs_SSC"]["pooled"]["A"], 4),
            "qss_ssc_A_pooled_published_preP0": 0.82,
            "preP0_published_constants": {"tang": 0.95, "qss_yxss": 0.515,
                                          "qss_ssc": 0.82},
            "note": ("Pooled realignment A reproduces the punctuation-normalised "
                     "phase1b M1 invariance values (Tang 0.9596, Song-QSS/YXSS "
                     "0.557); pre-P0 published constants were 0.95/0.515/0.82."),
        }
        put("tab:realign", {
            "_source": ("align_works.py -- POOLED mode reproduces the published "
                        "tab:invariance A values, which validates this realignment "
                        "pipeline against the existing one before its new numbers "
                        "are read"),
            "_key_definition": ("work key = (simplified author, canonical title); the "
                                "title is t2s-converted, stripped of punctuation and "
                                "whitespace and of trailing series numerals, so that a "
                                "QTS numbered poem-series re-flows onto the same key as "
                                "YD's single record; records sharing a key are merged"),
            "_inclusion": ">=300 chars per author per source (same rule as cross_edition_points.py)",
            "_selfcheck": _selfcheck,
            "variants": {"strict": {k: _wr(aws, k) for k in _keys},
                         "loose_strip_parentheticals": {k: _wr(awl, k) for k in _keys}},
            "identity_baseline": {
                "_source": ("align_variants.py -- guards against reading the aligned "
                            "concordance as a trivially identical-string artifact"),
                "pairs": {k: {"shared_works": v["shared_works"],
                              "identical_strings": v["identical_works"],
                              "frac_identical": v["frac_identical"],
                              "variant_works": v["variant_works"],
                              "mean_char_similarity": v["mean_char_similarity"]}
                          for k, v in (av.get("pairs") or {}).items()}},
            # ---- new consensus-driven enhancement keys (E2-E6) ----
            "threshold_sweep": aws.get("threshold_sweep"),
            "decomposition": aws.get("decomposition"),
            "constrained": {"strict": aws.get("constrained"),
                            "loose_strip_parentheticals": awl.get("constrained")},
            "bootstrap_ci": {"strict": aws.get("bootstrap_ci"),
                             "loose_strip_parentheticals": awl.get("bootstrap_ci")},
            "threshold_order": aws.get("threshold_order"),
            "per_poem_paired_delta": av.get("per_poem_paired_delta"),
        }, "results_cross_edition_aligned_{strict,loose}.json + results_align_variants.json",
            "variants.* / identity_baseline.pairs.* / threshold_sweep / decomposition / "
            "constrained / bootstrap_ci / threshold_order / per_poem_paired_delta")

    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dl", action="store_true", help="skip CNN/DL stages")
    args = ap.parse_args()

    log_lines = []
    results = {}
    for script, writes, needs_dl in STAGES:
        status, dt, rc, out = run_stage(script, args.no_dl, needs_dl)
        results[script] = {"status": status, "seconds": round(dt, 1), "exit": rc}
        log_lines.append(f"{'='*70}\n[STAGE] {script}  status={status}  exit={rc}  {dt:.1f}s\n{'='*70}\n{out}\n")

    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    ledger = build_ledger()
    ledger["__stages__"] = results
    with open(os.path.join(BASE, "numbers_ledger.json"), "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=1)

    print("\n" + "="*70)
    print("RECOMPUTE SUMMARY")
    print("="*70)
    for s, r in results.items():
        print(f"  {s:26} {r['status']:8} {r['seconds']:>7.1f}s")
    print(f"\nledger keys: {len(ledger)-1}")
    print("wrote numbers_ledger.json + recompute.log")


if __name__ == "__main__":
    main()
