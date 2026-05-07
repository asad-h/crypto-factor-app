import pandas as pd

from dashboard.project_events import (
    build_nansen_token_context_requests,
    compute_rolling_30d_return_correlations,
    emission_bucket,
    parse_defillama_unlock_text,
    parse_snapshot_governance_payload,
)


def test_parse_defillama_unlock_text_hype_cards():
    text = """
    Unlock Events
    Unlock Value $17.88m 0.042% (0.18% of float) May 6, 2026 7:00 AM GMT+07:00
    Core Contributors $17.88m 421,879 HYPE
    Unlock Value $14.12m 0.033% (0.14% of float) Apr 6, 2026 7:00 AM GMT+07:00
    Core Contributors $14.12m 333,335 HYPE
    """
    events = parse_defillama_unlock_text(
        text,
        slug="hyperliquid",
        source_url="https://defillama.com/protocol/unlocks/hyperliquid",
        project="Hyperliquid",
        ticker="HYPE",
    )

    assert len(events) == 2
    first = events.sort_values("date").iloc[-1]
    assert first["recipient"] == "Core Contributors"
    assert first["bucket"] == "Potential Selling"
    assert first["value_usd"] == 17_880_000
    assert first["token_amount"] == 421_879
    assert first["pct_supply"] == 0.042
    assert first["pct_float"] == 0.18


def test_emission_bucket_groups_selling_labels():
    assert emission_bucket("Core Contributors") == "Potential Selling"
    assert emission_bucket("Seed Investors") == "Potential Selling"
    assert emission_bucket("Ecosystem") == "Ecosystem"


def test_nansen_request_builder_keeps_endpoint_inventory():
    requests = build_nansen_token_context_requests("ethereum", "0xabc")
    endpoints = {request["endpoint"] for request in requests}

    assert "/api/v1/tgm/token-information" in endpoints
    assert "/api/v1/tgm/flow-intelligence" in endpoints
    assert "/api/v1/tgm/flows" in endpoints
    assert "/api/v1/tgm/who-bought-sold" in endpoints
    assert "/api/v1/tgm/dex-trades" in endpoints
    assert "/api/v1/tgm/transfers" in endpoints
    assert any(request["payload"].get("label_type") == "smart_money" for request in requests)
    assert any(request["payload"].get("holder_segment") == "exchange" for request in requests)


def test_parse_snapshot_governance_payload():
    payload = {
        "data": {
            "proposals": [
                {
                    "id": "proposal-1",
                    "title": "Treasury vote",
                    "start": 1_762_000_000,
                    "end": 1_762_086_400,
                    "state": "active",
                    "scores_total": 12345,
                    "choices": ["For", "Against"],
                    "space": {"id": "lido-snapshot.eth", "name": "Lido"},
                }
            ]
        }
    }

    events = parse_snapshot_governance_payload(
        payload,
        project="Lido",
        ticker="LDO",
        defillama_slug="lido",
        governance_url="https://defillama.com/governance/lido",
    )

    assert len(events) == 1
    assert events.iloc[0]["event_type"] == "governance"
    assert events.iloc[0]["state"] == "active"
    assert "snapshot.box" in events.iloc[0]["source_url"]


def test_rolling_30d_correlations_handle_hype_self_and_missing():
    dates = pd.date_range("2026-01-01", periods=65, freq="D")
    rows = []
    for i, day in enumerate(dates):
        rows.extend(
            [
                {"date": day, "ticker": "HYPE", "price": 10 + i},
                {"date": day, "ticker": "BTC", "price": 100 + i * 2},
                {"date": day, "ticker": "ETH", "price": 50 + i},
            ]
        )
    project_ts = pd.DataFrame(rows)

    corr = compute_rolling_30d_return_correlations(project_ts, "HYPE", ["BTC", "HYPE", "MISSING", "BTC"])

    assert corr["Benchmark"].tolist() == ["BTC", "HYPE", "MISSING"]
    assert corr.loc[corr["Benchmark"] == "HYPE", "Correlation"].iloc[0] == 1.0
    assert corr.loc[corr["Benchmark"] == "MISSING", "Observations"].iloc[0] == 0
    assert corr.loc[corr["Benchmark"] == "BTC", "Observations"].iloc[0] > 0

