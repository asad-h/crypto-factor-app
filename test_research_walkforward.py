import unittest

import numpy as np
import pandas as pd

from crypto_factor_model.research.data import (
    compute_btc_20w_regime,
    compute_eligibility_panels,
    compute_forward_return_panels,
)
from crypto_factor_model.research.evaluation import (
    EvaluationContext,
    _period_dates,
    build_signal_decisions,
    choose_strict_l_s_candidate,
    construct_positions,
    purged_walk_forward_folds,
    run_portfolio_backtest,
)


class ResearchWalkForwardTests(unittest.TestCase):
    def test_forward_return_alignment_enters_t_plus_1_exits_h_plus_1(self):
        dates = pd.date_range("2024-05-01", periods=45, freq="D")
        price = pd.DataFrame({"A": np.arange(100, 145, dtype=float)}, index=dates)
        fwd, entry, exit_ = compute_forward_return_panels(price)

        dt = dates[3]
        self.assertEqual(entry[7].loc[dt, "A"], price.loc[dates[4], "A"])
        self.assertEqual(exit_[7].loc[dt, "A"], price.loc[dates[11], "A"])
        expected = price.loc[dates[11], "A"] / price.loc[dates[4], "A"] - 1
        self.assertAlmostEqual(fwd[7].loc[dt, "A"], expected)

    def test_train_validation_dates_are_purged_and_label_complete(self):
        dates = pd.date_range("2024-05-01", "2026-05-05", freq="D")
        self.assertEqual(_period_dates(dates, "Train", 30).max(), pd.Timestamp("2025-03-30"))
        self.assertEqual(_period_dates(dates, "Validation", 30).max(), pd.Timestamp("2025-09-30"))
        self.assertEqual(_period_dates(dates, "Test", 30).max(), pd.Timestamp("2026-04-04"))

    def test_purged_rolling_folds_are_isolated_before_test(self):
        dates = pd.date_range("2024-05-01", "2026-05-05", freq="D")
        folds = purged_walk_forward_folds(dates, horizon=14)
        self.assertGreaterEqual(len(folds), 3)
        for fold in folds:
            self.assertLessEqual(fold["train_end"], fold["validation_start"] - pd.Timedelta(days=15))
            self.assertLessEqual(fold["validation_end"], pd.Timestamp("2025-10-31"))
            self.assertEqual(fold["purge_days"], 15)

    def test_signal_decisions_ignore_test_labels(self):
        dates = pd.date_range("2024-05-01", "2026-02-01", freq="D")
        tokens = list("ABCDEF")
        base_scores = pd.Series(np.arange(len(tokens), dtype=float), index=tokens)
        signal = pd.DataFrame([base_scores] * len(dates), index=dates)
        returns = {}
        for horizon in [7, 14, 30]:
            panel = pd.DataFrame(index=dates, columns=tokens, dtype=float)
            panel.loc[dates <= "2025-04-01"] = base_scores.to_numpy()
            panel.loc[dates >= "2025-11-01"] = -100 * base_scores.to_numpy()
            returns[horizon] = panel

        decisions = build_signal_decisions({"momentum": {"synthetic": signal}}, returns, dates)
        decision = decisions.iloc[0]
        self.assertEqual(decision["sign"], 1)
        self.assertTrue(decision["selected"])

    def test_btc_20w_regime_uses_latest_completed_weekly_close(self):
        dates = pd.date_range("2023-12-01", "2024-08-15", freq="D")
        btc = pd.Series(np.linspace(40000, 70000, len(dates)), index=dates)
        monday = pd.Timestamp("2024-07-15")
        previous_sunday = pd.Timestamp("2024-07-14")
        btc.loc[monday] = 1_000_000
        regime = compute_btc_20w_regime(btc, dates)
        self.assertEqual(regime.loc[monday, "btc_weekly_close"], btc.loc[previous_sunday])
        self.assertIn(regime.loc[monday, "regime"], {"Bullish", "Bearish"})

    def test_universe_filter_uses_historical_mcap_and_volume(self):
        dates = pd.date_range("2024-05-01", periods=120, freq="D")
        master = pd.DataFrame(
            {
                "research_id": ["A"],
                "ticker": ["A"],
                "token": ["A"],
                "name": ["Asset A"],
                "category": ["L1 / L2"],
                "sector": ["Infrastructure"],
                "binance_futures_symbol": ["AUSDT"],
            }
        )
        price = pd.DataFrame({"A": np.linspace(1, 2, len(dates))}, index=dates)
        volume = pd.DataFrame({"A": 2_000_000.0}, index=dates)
        mcap = pd.DataFrame({"A": 50_000_000.0}, index=dates)
        mcap.loc[dates[-10]:, "A"] = 200_000_000.0

        eligibility = compute_eligibility_panels(master, price, volume, mcap)
        self.assertFalse(bool(eligibility["eligible_base"].loc[dates[95], "A"]))
        self.assertTrue(bool(eligibility["eligible_base"].loc[dates[-1], "A"]))

    def test_position_construction_variants(self):
        tokens = list("ABCDEFGHIJKL")
        scores = pd.Series(np.arange(len(tokens), dtype=float), index=tokens)
        returns = pd.Series(0.01, index=tokens)
        eligible = pd.Series(True, index=tokens)
        shortable = pd.Series(True, index=tokens)

        ls = construct_positions(scores, returns, eligible, shortable, "always_l_s")
        self.assertEqual((ls > 0).sum(), 5)
        self.assertEqual((ls < 0).sum(), 5)
        self.assertAlmostEqual(ls.sum(), 0.0)
        self.assertAlmostEqual(ls.abs().sum(), 1.0)

        bull = construct_positions(scores, returns, eligible, shortable, "regime_switch", regime="Bullish")
        bear = construct_positions(scores, returns, eligible, shortable, "regime_switch", regime="Bearish")
        self.assertTrue((bull > 0).all())
        self.assertTrue((bear < 0).all())
        self.assertAlmostEqual(bull.sum(), 1.0)
        self.assertAlmostEqual(bear.sum(), -1.0)

    def test_sticky_l_s_keeps_prior_shorts_inside_rank_buffer(self):
        tokens = list("ABCDEFGHIJKLMNO")
        returns = pd.Series(0.01, index=tokens)
        eligible = pd.Series(True, index=tokens)
        shortable = pd.Series(True, index=tokens)
        first_scores = pd.Series(np.arange(len(tokens), dtype=float), index=tokens)
        second_scores = pd.Series(
            {
                "F": 0,
                "G": 1,
                "H": 2,
                "I": 3,
                "J": 4,
                "A": 5,
                "B": 6,
                "C": 7,
                "D": 8,
                "E": 9,
                "K": 10,
                "L": 11,
                "M": 12,
                "N": 13,
                "O": 14,
            },
            dtype=float,
        )

        first = construct_positions(first_scores, returns, eligible, shortable, "sticky_l_s")
        fresh_second = construct_positions(second_scores, returns, eligible, shortable, "always_l_s")
        sticky_second = construct_positions(second_scores, returns, eligible, shortable, "sticky_l_s", prev_weights=first)

        self.assertEqual(set(first[first < 0].index), set("ABCDE"))
        self.assertEqual(set(fresh_second[fresh_second < 0].index), set("FGHIJ"))
        self.assertEqual(set(sticky_second[sticky_second < 0].index), set("ABCDE"))

    def test_strict_selection_does_not_promote_directional_or_weak_l_s_models(self):
        rows = [
            {
                "model": "learned_global",
                "variant": "long_only",
                "horizon": 14,
                "period": "Validation",
                "eligible_universe": "eligible_sensitivity",
                "n_rebalances": 10,
                "total_return": 0.40,
                "sharpe": 3.0,
                "calmar": 3.0,
                "avg_turnover": 1.0,
                "max_drawdown": -0.05,
                "rolling_folds": 4,
                "rolling_positive_folds": 4,
                "rolling_avg_total_return": 0.05,
            },
            {
                "model": "learned_global",
                "variant": "always_l_s",
                "horizon": 14,
                "period": "Validation",
                "eligible_universe": "eligible_sensitivity",
                "n_rebalances": 10,
                "total_return": -0.02,
                "sharpe": -0.2,
                "calmar": -0.4,
                "avg_turnover": 1.0,
                "max_drawdown": -0.20,
                "rolling_folds": 4,
                "rolling_positive_folds": 2,
                "rolling_avg_total_return": -0.01,
            },
        ]
        selected, annotated, _, status, passed = choose_strict_l_s_candidate(pd.DataFrame(rows))

        self.assertEqual(status, "NO_STRICT_PASS")
        self.assertFalse(passed)
        self.assertEqual(selected["variant"], "always_l_s")
        self.assertFalse(bool(annotated.loc[annotated["variant"].eq("long_only"), "is_l_s_candidate"].iloc[0]))

    def test_turnover_and_transaction_costs(self):
        dates = pd.date_range("2024-05-01", periods=80, freq="D")
        tokens = list("ABCDEFGHIJKL")
        composite = pd.DataFrame([np.arange(len(tokens), dtype=float)] * len(dates), index=dates, columns=tokens)
        returns = {h: pd.DataFrame(0.01, index=dates, columns=tokens) for h in [7, 14, 30]}
        entries = {h: pd.DataFrame(1.0, index=dates, columns=tokens) for h in [7, 14, 30]}
        exits = {h: pd.DataFrame(1.01, index=dates, columns=tokens) for h in [7, 14, 30]}
        eligible = pd.DataFrame(True, index=dates, columns=tokens)
        shortable = pd.DataFrame(True, index=dates, columns=tokens)
        metadata = pd.DataFrame({"token": tokens, "name": tokens}, index=tokens)
        regime = pd.Series("Bullish", index=dates)
        btc_returns = {h: pd.Series(0.005, index=dates) for h in [7, 14, 30]}
        ctx = EvaluationContext(dates, metadata, returns, entries, exits, eligible, eligible, shortable, regime, btc_returns)

        portfolio, constituents = run_portfolio_backtest(ctx, composite, 7, "Train", "always_l_s", cost_bps=50)
        self.assertFalse(portfolio.empty)
        self.assertEqual(int(portfolio.iloc[0]["n_positions"]), 10)
        self.assertAlmostEqual(portfolio.iloc[0]["turnover"], 1.0)
        self.assertAlmostEqual(portfolio.iloc[0]["cost"], 0.005)
        first_constituents = constituents[constituents["rebalance_date"].eq(portfolio.iloc[0]["date"])]
        self.assertEqual(len(first_constituents), 10)


if __name__ == "__main__":
    unittest.main()
