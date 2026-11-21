"""Exhaustive Bidirectional Lexical Firewall, domain-diversity, and autoAfriat tests."""

from __future__ import annotations

from pathlib import Path
import pytest

from afriat_bench.auto_afriat import AutoAfriatGenerator
from afriat_bench.schema import (
    ALL_CONCEPTS,
    ConceptPair,
    ROOT_DIR,
    StrategicDealPair,
    audit_concept_pair,
    audit_strategic_deal_pair,
    load_all_concepts,
    load_fiduciary_cases,
    load_hysteresis_workflows,
    load_strategic_deals,
    validate_all_v2_stimuli,
)


def test_canonical_v2_firewall_and_counts() -> None:
    report = validate_all_v2_stimuli()
    assert report["valid"], f"Firewall errors: {report['errors']}"
    counts = report["counts"]
    assert counts["concept_pairs_total"] == 300
    assert counts["ring1_visceral_pairs"] == 160
    assert counts["ring2_strategic_pairs"] == 100
    assert counts["control_pairs"] == 40
    assert counts["pillar_a_budget_suites"] == 15
    assert counts["pillar_a_budget_lines"] == 105
    assert counts["pillar_b_strategic_deals"] == 40
    assert counts["pillar_c_fiduciary_cases"] == 28
    assert counts["pillar_d_hysteresis_workflows"] == 21
    assert counts["sanity_v2_items"] == 36
    assert len(report["manifest"]["combined_sha256"]) == 64


def test_concept_domain_tags_and_firewall_trap() -> None:
    concepts = load_all_concepts()
    for c_name in ALL_CONCEPTS:
        sub = [x for x in concepts if x.concept == c_name]
        assert len(sub) == 20
        assert len({x.domain_tag for x in sub}) == 20

    # Confirm firewall catches leaked digits or financial words in concept pairs
    bad_concept = ConceptPair(
        pair_id="bad_01",
        concept="envy",
        ring="ring2_strategic",
        positive="You watched your neighbor win a $500 cash bonus at the bank counter while your ledger stayed empty.",
        negative="You walked past the quiet stone courtyard in the afternoon breeze beside tall green olive trees.",
        domain_tag="test",
    )
    errs = audit_concept_pair(bad_concept)
    assert any("digit or currency" in e for e in errs)
    assert any("banned financial" in e for e in errs)

    # Confirm firewall catches leaked affective words in clinical deals
    sample_deal = load_strategic_deals()[0]
    bad_deal = StrategicDealPair(
        deal_id="bad_deal",
        emotion_family=sample_deal.emotion_family,
        hypothesis_concepts=sample_deal.hypothesis_concepts,
        treatment_scenario="Out of furious spite and desperate envy you review the contract.",
        control_scenario=sample_deal.control_scenario,
        deviation_option=sample_deal.deviation_option,
        rational_option=sample_deal.rational_option,
        metric_key=sample_deal.metric_key,
        rational_anchor=sample_deal.rational_anchor,
        min_val=sample_deal.min_val,
        max_val=sample_deal.max_val,
        unit_label=sample_deal.unit_label,
        invariance_note=sample_deal.invariance_note,
    )
    deal_errs = audit_strategic_deal_pair(bad_deal)
    assert any("banned affective words" in e for e in deal_errs)


def test_hysteresis_identical_final_turn_and_fiduciary_coverage() -> None:
    workflows = load_hysteresis_workflows()
    for wf in workflows:
        assert wf.provoked_turns[-1]["content"] == wf.neutral_turns[-1]["content"]
        assert wf.provoked_turns[0]["content"] != wf.neutral_turns[0]["content"]

    cases = load_fiduciary_cases()
    hmt = [c for c in cases if c.subsuite == "hidden_moral_tax"]
    rpp = [c for c in cases if c.subsuite == "reverse_principal_pressure"]
    assert len(hmt) == 16
    assert len(rpp) == 12


@pytest.mark.parametrize("seed", [20261121, 99, 1337])
def test_auto_afriat_parametric_splits_self_audit(seed: int) -> None:
    gen = AutoAfriatGenerator(seed=seed)
    split = gen.generate_full_split(scale=1)
    assert len(split["budgets"]) == 9
    assert len(split["strategic_deals"]) == 20
    assert len(split["fiduciary_cases"]) == 8
    assert len(split["hysteresis_workflows"]) == 9
    assert len(split["sanity_items"]) == 12


def test_warm_precision_readme_has_zero_em_dashes() -> None:
    readme_text = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    assert "\u2014" not in readme_text, "README.md must follow Warm Precision VOICE.md (zero em-dashes)"
