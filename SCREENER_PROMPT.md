# Prompt for New Session

Copy-paste this into a new Claude or Codex session with the `Crypto Factor Model` folder mounted.

---

Read `FACTOR_MODEL_HANDOFF.md` first. It has the full architecture, all 29 signals, data sources, methodology, and known gaps for the crypto factor model and screener.

Then read these files in order to understand the codebase:
1. `crypto_factor_model/config.py`
2. `crypto_factor_model/main.py`
3. `crypto_factor_model/signals/fundamentals.py`
4. `crypto_factor_model/signals/momentum.py`
5. `crypto_factor_model/signals/flows.py`
6. `crypto_factor_model/signals/utils.py`
7. `crypto_factor_model/composite.py`
8. `crypto_factor_model/clients/blockworks.py`
9. `crypto_factor_model/clients/binance.py`

The pipeline is functional but needs enhancements to work as a proper trading screener. Here is what I need built:

**Priority 1: Screener output with change metrics**

Build a screener view (can be Streamlit, HTML, or a structured DataFrame output) that shows for each token in the universe:

- Current price, mcap, FDV
- Revenue: current 7d avg, WoW change %, MoM change %
- Fees: current 7d avg, WoW change %, MoM change %
- Deposits (lending): current, WoW change %, MoM change %
- Open interest: current, WoW change %
- Composite factor score and rank
- Top 3 contributing signals (what's driving the score)

All change metrics should use Blockworks chain-level data that's already being fetched (rev-usd, transaction-fee-total-usd, lending-deposit-total-usd). OI from Binance Futures.

**Priority 2: Market cap bucket movers**

Add a view that buckets tokens by mcap:
- Micro: < $100M
- Small: $100M - $500M  
- Mid: $500M - $1B
- Large: $1B - $5B
- Mega: $5B+

Within each bucket, show the top movers by: (a) 7d price change, (b) 4w composite score change, (c) biggest revenue growth. This helps spot breakout tokens in their weight class.

**Priority 3: Fix known gaps**

- Wire actual Binance daily volume into universe screening (currently placeholder zeros)
- Integrate DefiLlama bridge flows into the main pipeline (client exists, not called)
- Add deposit change as a new flow signal
- Flag and handle the revenue/trading_fees correlation = 1.0 redundancy issue

**Context about me**: I'm a liquid token trader at a crypto hedge fund. I use this for trade origination. The screener needs to surface actionable signals, not just rankings. I want to see what's changing and where the momentum is, segmented by size. Write code that's production-minded. Blockworks API key is set via BLOCKWORKS_API_KEY env var.
