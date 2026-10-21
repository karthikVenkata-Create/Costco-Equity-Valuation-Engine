"""
Costco (COST) Equity Valuation Engine
======================================

One command re-values Costco against today's market:

    python valuation_engine.py

Pipeline
  1. MARKET DATA   price, market cap, multiples and 1 year of daily prices for
                   COST + peers (WMT, TGT, BJ) from Yahoo Finance via yfinance
  2. COMPS         trading-comps table and the price COST would have at peer multiples
  3. WACC          CAPM cost of equity from the live 10y Treasury yield and a beta
                   from 2 years of weekly returns vs the S&P 500 (adjusted toward 1,
                   with a sanity check)
  4. DCF           5-year FCFF projection + Gordon-growth terminal value,
                   WACC x terminal-growth sensitivity grid
  5. REVERSE DCF   what the price is assuming, four ways: 5-year growth, perpetual
                   growth, years of high growth, and margin
  6. MONTE CARLO   (a) 10,000 DCF runs with growth / margin / WACC / terminal g drawn
                       from Bear-Base-Bull ranges -> range of intrinsic value
                   (b) 10,000 one-year price paths at COST's realized volatility
                       -> odds the stock actually trades down to the target
  7. OUTPUT        rating + target, every run appended to SQLite (valuation.db),
                   optional charts (--charts)

Flags
  --offline        no network: uses FY2026 fundamentals and the fallback inputs in
                   assumptions.json; peer comps are skipped rather than faked
  --charts         save PNG charts to ./charts
  --runs N         number of simulations (default 10,000)
  --db PATH        database file (default valuation.db)

Every judgment call lives in assumptions.json, not in this file.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRADING_DAYS = 252
WIDTH = 78

# ----------------------------------------------------------------------------
# Console helpers
# ----------------------------------------------------------------------------


def banner(title: str) -> None:
    print("\n" + "=" * WIDTH + f"\n {title}\n" + "=" * WIDTH)


def line(label: str, value: str) -> None:
    print(f"  {label:<52}{value:>24}")


def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def usd(x, decimals: int = 2) -> str:
    return "n/a" if _bad(x) else f"${x:,.{decimals}f}"


def pct(x, decimals: int = 1) -> str:
    return "n/a" if _bad(x) else f"{x * 100:.{decimals}f}%"


# ----------------------------------------------------------------------------
# 1. Market data
# ----------------------------------------------------------------------------

@dataclass
class Quote:
    ticker: str
    name: str
    price: float
    market_cap: float          # $ millions
    enterprise_value: float    # $ millions
    trailing_pe: float
    ev_ebitda: float
    ev_revenue: float


@dataclass
class MarketData:
    mode: str                  # "live" or "offline"
    as_of: str
    price: float               # COST share price
    market_cap: float          # COST market cap, $ millions
    risk_free_rate: float
    beta: float
    volatility: float          # annualised
    beta_source: str
    quotes: dict[str, Quote] = field(default_factory=dict)   # empty offline
    notes: list[str] = field(default_factory=list)


def _num(info: dict, key: str) -> float:
    try:
        return float(info.get(key))
    except (TypeError, ValueError):
        return float("nan")


BETA_RANGE = (0.3, 2.0)   # outside this, a large-cap retailer's beta isn't believable


def estimate_beta(weekly: pd.DataFrame, stock: str, index: str,
                  yahoo_beta: float, fallback: float) -> tuple[float, str]:
    """Blume-adjusted beta from 2 years of weekly returns, with a sanity check.

    Raw regression betas bounce around a lot, so the standard fix is to pull
    them toward 1 (adjusted = 0.67 * raw + 0.33). If the result still looks
    unrealistic, fall back to Yahoo's published beta, then to assumptions.json.
    """
    lo, hi = BETA_RANGE
    raw = float("nan")
    try:
        r = np.log(weekly[[stock, index]]).diff().dropna()
        if len(r) >= 52:
            cov = np.cov(r[stock], r[index])
            raw = float(cov[0, 1] / cov[1, 1])
    except Exception:
        pass
    if not np.isnan(raw):
        adj = 0.67 * raw + 0.33
        if lo <= adj <= hi:
            return adj, f"2-yr weekly vs S&P 500, adjusted; raw {raw:.2f}"
    note = "n/a" if np.isnan(raw) else f"{raw:.2f}"
    if not np.isnan(yahoo_beta) and lo <= yahoo_beta <= hi:
        return yahoo_beta, f"Yahoo; regression gave {note}"
    return fallback, f"fallback; regression gave {note}"


def fetch_live(cfg: dict, lookback: int) -> MarketData:
    import yfinance as yf  # only needed online

    target, index = cfg["target_ticker"], cfg["market_index"]
    tickers = [target, *cfg["peer_tickers"]]

    quotes = {}
    for t in tickers:
        info = yf.Ticker(t).info
        price = _num(info, "currentPrice")
        if np.isnan(price):
            price = _num(info, "regularMarketPrice")
        quotes[t] = Quote(t, info.get("shortName", t), price,
                          _num(info, "marketCap") / 1e6, _num(info, "enterpriseValue") / 1e6,
                          _num(info, "trailingPE"), _num(info, "enterpriseToEbitda"),
                          _num(info, "enterpriseToRevenue"))

    # Daily prices -> realized volatility for the price simulation
    hist = yf.download([target, index], period="14mo", auto_adjust=True,
                       progress=False)["Close"].dropna()
    rets = np.log(hist).diff().dropna().tail(lookback)
    vol = float(rets[target].std(ddof=1) * np.sqrt(TRADING_DAYS))

    # Weekly prices over 2 years -> beta (daily betas are too noisy to trust)
    weekly = yf.download([target, index], period="2y", interval="1wk", auto_adjust=True,
                         progress=False)["Close"].dropna()
    yahoo_beta = _num(yf.Ticker(target).info, "beta")
    beta, beta_source = estimate_beta(weekly, target, index, yahoo_beta,
                                      cfg["wacc_inputs"]["beta_fallback"])

    tnx = yf.Ticker("^TNX").history(period="5d")["Close"].dropna()   # quoted in percent
    rf = float(tnx.iloc[-1]) / 100 if len(tnx) else cfg["wacc_inputs"]["risk_free_rate_fallback"]

    q = quotes[target]
    return MarketData("live", str(hist.index[-1].date()), q.price, q.market_cap, rf, beta, vol,
                      beta_source, quotes)


def build_offline(cfg: dict) -> MarketData:
    f, w = cfg["fundamentals_fy2026"], cfg["wacc_inputs"]
    price = f["reference_price"]
    return MarketData(
        "offline", f["reference_price_date"], price, price * f["shares_outstanding_mm"],
        w["risk_free_rate_fallback"], w["beta_fallback"], w["volatility_fallback"],
        "fallback from assumptions.json",
        notes=["OFFLINE MODE: price, risk-free rate, beta and volatility come from "
               "assumptions.json; live peer comps are skipped."],
    )


# ----------------------------------------------------------------------------
# 2. Comparable companies (live only)
# ----------------------------------------------------------------------------

def comps_analysis(md: MarketData, cfg: dict) -> tuple[pd.DataFrame | None, dict]:
    if not md.quotes:
        return None, {}
    T, f = cfg["target_ticker"], cfg["fundamentals_fy2026"]
    df = pd.DataFrame([{
        "Ticker": q.ticker, "Price": q.price, "Mkt Cap ($bn)": q.market_cap / 1e3,
        "P/E": q.trailing_pe, "EV/EBITDA": q.ev_ebitda, "EV/Rev": q.ev_revenue,
    } for q in md.quotes.values()]).set_index("Ticker")
    med = df.drop(index=T).median()
    df.loc["Peer median"] = med

    q = md.quotes[T]
    net_debt = f["total_debt"] - f["cash"]
    ebitda = q.enterprise_value / q.ev_ebitda if q.ev_ebitda else float("nan")
    implied = {
        "Peer P/E": (q.price / q.trailing_pe) * med["P/E"],
        "Peer EV/EBITDA": (ebitda * med["EV/EBITDA"] - net_debt) / f["shares_outstanding_mm"],
    }
    return df, implied


# ----------------------------------------------------------------------------
# 3. WACC
# ----------------------------------------------------------------------------

@dataclass
class WACC:
    cost_of_equity: float
    after_tax_cost_of_debt: float
    weight_equity: float
    wacc: float


def compute_wacc(cfg: dict, md: MarketData) -> WACC:
    w, f = cfg["wacc_inputs"], cfg["fundamentals_fy2026"]
    ke = md.risk_free_rate + md.beta * w["equity_risk_premium"]
    kd = w["pre_tax_cost_of_debt"] * (1 - w["tax_rate"])
    we = md.market_cap / (md.market_cap + f["total_debt"])
    return WACC(ke, kd, we, we * ke + (1 - we) * kd)


# ----------------------------------------------------------------------------
# 4. DCF (vectorised: the same function powers the Monte Carlo)
# ----------------------------------------------------------------------------

def dcf_per_share(cfg: dict, wacc, growth_shift=0.0, margin_shift=0.0,
                  terminal_g=None, detail: bool = False):
    """Intrinsic value per share. Arguments may be scalars or NumPy arrays."""
    f, b = cfg["fundamentals_fy2026"], cfg["dcf_base_case"]
    tax = cfg["wacc_inputs"]["tax_rate"]
    wacc = np.asarray(wacc, dtype=float)
    g = np.asarray(b["terminal_growth"] if terminal_g is None else terminal_g, dtype=float)

    rev_prev, pv_fcff, rows = f["revenue"], 0.0, []
    for t in range(b["projection_years"]):
        rev = rev_prev * (1 + b["revenue_growth"][t] + growth_shift)
        ebit = rev * (b["ebit_margin"][t] + margin_shift)
        nopat = ebit * (1 - tax)
        reinvest = (rev - rev_prev) / b["sales_to_capital"]
        fcff = nopat - reinvest
        pv_fcff = pv_fcff + fcff / (1 + wacc) ** (t + 1)
        rows.append({"Year": f"FY{2027 + t}", "Revenue": rev, "EBIT": ebit,
                     "NOPAT": nopat, "Reinvestment": reinvest, "FCFF": fcff})
        rev_prev = rev

    # Terminal year reinvests only what it needs to grow at g
    rev_tv = rev_prev * (1 + g)
    fcff_tv = (rev_tv * (b["ebit_margin"][-1] + margin_shift) * (1 - tax)
               - (rev_tv - rev_prev) / b["sales_to_capital"])
    tv = fcff_tv / np.maximum(wacc - g, 0.005)        # guard against wacc <= g
    pv_tv = tv / (1 + wacc) ** b["projection_years"]

    ev = pv_fcff + pv_tv
    equity = ev + f["cash"] - f["total_debt"]
    value = equity / f["shares_outstanding_mm"]
    if not detail:
        return value
    return float(value), {
        "table": pd.DataFrame(rows).set_index("Year"),
        "pv_fcff": float(pv_fcff), "pv_tv": float(pv_tv),
        "equity_value": float(equity), "tv_share": float(pv_tv / ev),
    }


def sensitivity_grid(cfg: dict, wacc_mid: float) -> pd.DataFrame:
    waccs = wacc_mid + np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
    gs = [0.020, 0.025, 0.030, 0.035, 0.040]
    return pd.DataFrame(
        {f"g {g:.1%}": [float(dcf_per_share(cfg, w, terminal_g=g)) for w in waccs] for g in gs},
        index=[f"WACC {w:.2%}" for w in waccs])


def _solve(f, lo: float, hi: float) -> float:
    """Bisection for an increasing function f: returns x where f(x) crosses 0."""
    if f(hi) < 0:
        return float("nan")            # not reachable inside the range
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def implied_growth(cfg: dict, wacc: float, price: float) -> float:
    """Reverse DCF, version 1: extra growth needed if ONLY the next 5 years change.

    This overstates the requirement for a long-runway business, because
    terminal growth stays at its base value. Versions 2-4 relax that.
    """
    return _solve(lambda x: float(dcf_per_share(cfg, wacc, growth_shift=x)) - price, -0.10, 0.60)


def implied_terminal_growth(cfg: dict, wacc: float, price: float) -> float:
    """Reverse DCF, version 2: perpetual growth after 2031 that justifies the price."""
    return _solve(lambda g: float(dcf_per_share(cfg, wacc, terminal_g=g)) - price,
                  0.0, wacc - 0.0051)


def runway_cfg(cfg: dict, years: int, growth: float) -> dict:
    """Copy of cfg with `years` of flat `growth`, margins held at the final base-case level."""
    c = json.loads(json.dumps(cfg))
    b = c["dcf_base_case"]
    m = b["ebit_margin"]
    b["projection_years"] = years
    b["revenue_growth"] = [growth] * years
    b["ebit_margin"] = (m + [m[-1]] * years)[:years]
    return c


def implied_runway_years(cfg: dict, wacc: float, price: float, growth: float,
                         max_years: int = 60) -> int | None:
    """Reverse DCF, version 3: years of `growth` before fading to terminal growth."""
    for n in range(1, max_years + 1):
        if float(dcf_per_share(runway_cfg(cfg, n, growth), wacc)) >= price:
            return n
    return None


def implied_margin(cfg: dict, wacc: float, price: float) -> float:
    """Reverse DCF, version 4: 2031 operating margin needed with base-case growth."""
    shift = _solve(lambda x: float(dcf_per_share(cfg, wacc, margin_shift=x)) - price, -0.03, 0.15)
    return cfg["dcf_base_case"]["ebit_margin"][-1] + shift


# ----------------------------------------------------------------------------
# 5. Monte Carlo
# ----------------------------------------------------------------------------

def _tri(rng, spec: dict, n: int, center: float = 0.0) -> np.ndarray:
    lo, mode, hi = sorted([spec["bear"], spec["base"], spec["bull"]])
    return rng.triangular(center + lo, center + mode, center + hi, n)


def monte_carlo_dcf(cfg: dict, wacc: float, n: int, rng) -> np.ndarray:
    s = cfg["scenarios"]
    return dcf_per_share(cfg, _tri(rng, s["wacc_shift"], n, wacc),
                         _tri(rng, s["growth_shift"], n), _tri(rng, s["ebit_margin_shift"], n),
                         _tri(rng, s["terminal_growth"], n))


def monte_carlo_price(spot, sigma, mu, n, days, rng) -> np.ndarray:
    """Share price after `days` under geometric Brownian motion (drift = cost of equity)."""
    dt = 1 / TRADING_DAYS
    steps = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n, days))
    return spot * np.exp(steps.sum(axis=1))


def bands(x: np.ndarray) -> dict[str, float]:
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    return {"bear_p10": float(p10), "base_p50": float(p50), "bull_p90": float(p90),
            "mean": float(x.mean()), "std": float(x.std())}


# ----------------------------------------------------------------------------
# 6. Charts
# ----------------------------------------------------------------------------

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
ACCENT, ACCENT_LIGHT, WARN = "#0969da", "#9cc7f5", "#cf222e"


def save_charts(out: Path, price: float, target: float, mc_dcf: np.ndarray,
                ranges: list[tuple[str, float, float]], as_of: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
    paths = []

    # Football field: valuation ranges vs the current price
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(ranges) + 1.6), dpi=160)
    for i, (label, lo, hi) in enumerate(reversed(ranges)):
        ax.barh(i, hi - lo, left=lo, height=0.5, color=ACCENT_LIGHT, edgecolor=ACCENT)
        if hi - lo < 20:   # too narrow for two labels
            ax.text(lo - 8, i, f"${lo:,.0f}-{hi:,.0f}", ha="right", va="center", color=INK, fontsize=9)
        else:
            ax.text(lo - 8, i, f"${lo:,.0f}", ha="right", va="center", color=INK, fontsize=9)
            ax.text(hi + 8, i, f"${hi:,.0f}", ha="left", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(ranges)), [r[0] for r in reversed(ranges)])
    ax.axvline(price, color=WARN, lw=2)
    ax.axvline(target, color=INK, lw=1.5, ls="--")
    ax.text(price, len(ranges) - 0.35, f" Price ${price:,.0f}", color=WARN, fontsize=9, va="bottom")
    ax.text(target, len(ranges) - 0.35, f"Target ${target:,.0f} ", color=INK, fontsize=9,
            va="bottom", ha="right")
    hi_all = max(price, max(r[2] for r in ranges))
    ax.set_xlim(0, hi_all * 1.15)
    ax.set_ylim(-0.6, len(ranges) - 0.1)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"COST valuation ranges vs share price  ({as_of})", loc="left", fontsize=11)
    fig.tight_layout()
    paths.append(out / "football_field.png")
    fig.savefig(paths[-1])
    plt.close(fig)

    # Monte Carlo distribution of intrinsic value
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    ax.hist(mc_dcf, bins=60, color=ACCENT, edgecolor="white", linewidth=0.4)
    p10, p50, p90 = np.percentile(mc_dcf, [10, 50, 90])
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.28)                         # headroom so labels clear the bars
    for v, lab in ((p10, "Bear P10"), (p50, "Base P50"), (p90, "Bull P90")):
        ax.axvline(v, color=INK, lw=1, ls=":")
        ax.text(v, top * 1.25, f"{lab}\n${v:,.0f}", fontsize=8, color=INK, va="top", ha="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.set_xlabel("Intrinsic value per share")
    ax.set_ylabel("Simulations")
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    share_below = (mc_dcf < price).mean()
    ax.set_title(f"{len(mc_dcf):,} DCF simulations: {share_below:.0%} value COST below its "
                 f"${price:,.0f} price", loc="left", fontsize=11)
    fig.tight_layout()
    paths.append(out / "monte_carlo_dcf.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------
# 7. SQLite
# ----------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp_utc TEXT, data_as_of TEXT, mode TEXT, price REAL,
    risk_free_rate REAL, beta REAL, wacc REAL, terminal_growth REAL,
    dcf_value REAL, target_price REAL, rating TEXT, upside REAL,
    implied_5y_growth REAL, implied_terminal_growth REAL, implied_runway_years INTEGER,
    implied_2031_margin REAL, comps_pe_value REAL, comps_ev_ebitda_value REAL,
    mc_dcf_p10 REAL, mc_dcf_p50 REAL, mc_dcf_p90 REAL, mc_share_below_price REAL,
    volatility REAL, prob_price_below_target_1y REAL
);
CREATE TABLE IF NOT EXISTS comps (
    run_id INTEGER, ticker TEXT, price REAL, market_cap_bn REAL,
    pe REAL, ev_ebitda REAL, ev_revenue REAL
);
CREATE TABLE IF NOT EXISTS dcf_projection (
    run_id INTEGER, year TEXT, revenue REAL, ebit REAL, nopat REAL,
    reinvestment REAL, fcff REAL
);
CREATE TABLE IF NOT EXISTS monte_carlo_summary (
    run_id INTEGER, simulation TEXT, n_runs INTEGER,
    p10 REAL, p50 REAL, p90 REAL, mean REAL, std REAL
);
"""


def save_run(db: Path, run: dict, comps: pd.DataFrame | None, proj: pd.DataFrame,
             mc: dict[str, dict], n: int) -> int:
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    for col in run:                    # older databases: add any new columns
        if col not in have:
            con.execute(f"ALTER TABLE runs ADD COLUMN {col}")
    cur = con.execute(f"INSERT INTO runs ({', '.join(run)}) VALUES ({', '.join('?' * len(run))})",
                      list(run.values()))
    rid = cur.lastrowid
    if comps is not None:
        for t, r in comps.drop(index="Peer median").iterrows():
            con.execute("INSERT INTO comps VALUES (?,?,?,?,?,?,?)",
                        (rid, t, r["Price"], r["Mkt Cap ($bn)"], r["P/E"], r["EV/EBITDA"], r["EV/Rev"]))
    for y, r in proj.iterrows():
        con.execute("INSERT INTO dcf_projection VALUES (?,?,?,?,?,?,?)",
                    (rid, y, *r[["Revenue", "EBIT", "NOPAT", "Reinvestment", "FCFF"]]))
    for name, s in mc.items():
        con.execute("INSERT INTO monte_carlo_summary VALUES (?,?,?,?,?,?,?,?)",
                    (rid, name, n, s["bear_p10"], s["base_p50"], s["bull_p90"], s["mean"], s["std"]))
    con.commit()
    con.close()
    return rid


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def rate(upside: float, rule: dict) -> str:
    return "BUY" if upside > rule["buy_above"] else "SELL" if upside < rule["sell_below"] else "HOLD"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Costco equity valuation engine")
    ap.add_argument("--offline", action="store_true", help="run without internet")
    ap.add_argument("--charts", action="store_true", help="save PNG charts to ./charts")
    ap.add_argument("--runs", type=int, help="number of Monte Carlo simulations")
    ap.add_argument("--assumptions", default=str(HERE / "assumptions.json"))
    ap.add_argument("--db", default=str(HERE / "valuation.db"))
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.assumptions).read_text())
    sim, f = cfg["simulation"], cfg["fundamentals_fy2026"]
    n = args.runs or sim["n_runs"]
    rng = np.random.default_rng(sim["random_seed"])

    # 1. Data ------------------------------------------------------------------
    if args.offline:
        md = build_offline(cfg)
    else:
        try:
            md = fetch_live(cfg, sim["lookback_trading_days"])
        except Exception as exc:  # no network, Yahoo throttling, etc.
            print(f"\n[!] Live data unavailable ({type(exc).__name__}). Falling back to offline mode.")
            md = build_offline(cfg)
    price = md.price

    banner(f"COSTCO (COST) VALUATION ENGINE  |  data as of {md.as_of}  |  {md.mode}")
    for note in md.notes:
        print(f"  ** {note}")
    line("Share price", usd(price))
    line("Trailing P/E (price / FY2026 EPS)", f"{price / f['eps_diluted']:.1f}x")
    line("Price / book", f"{md.market_cap / f['total_equity']:.1f}x")

    # 2. Comps -----------------------------------------------------------------
    comps, implied = comps_analysis(md, cfg)
    banner("1. PUBLIC COMPARABLES")
    if comps is None:
        print("  Skipped offline. Run online to pull live multiples for WMT, TGT and BJ.")
    else:
        with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", WIDTH):
            print(comps.to_string())
        print()
        for k, v in implied.items():
            line(f"COST value at {k.lower()} median", usd(v))

    # 3. WACC ------------------------------------------------------------------
    w = compute_wacc(cfg, md)
    wi = cfg["wacc_inputs"]
    banner("2. COST OF CAPITAL")
    line("Risk-free rate (10y Treasury)", pct(md.risk_free_rate, 2))
    line(f"Beta ({md.beta_source})", f"{md.beta:.2f}")
    line("Equity risk premium", pct(wi["equity_risk_premium"], 2))
    line("Cost of equity = rf + beta x ERP", pct(w.cost_of_equity, 2))
    line("After-tax cost of debt", pct(w.after_tax_cost_of_debt, 2))
    line("Equity / debt weights", f"{w.weight_equity:.1%} / {1 - w.weight_equity:.1%}")
    line("WACC", pct(w.wacc, 2))

    # 4. DCF -------------------------------------------------------------------
    dcf_val, d = dcf_per_share(cfg, w.wacc, detail=True)
    banner("3. DISCOUNTED CASH FLOW  ($ millions)")
    with pd.option_context("display.float_format", "{:,.0f}".format, "display.width", WIDTH):
        print(d["table"].to_string())
    print()
    line("PV of 5-year free cash flow", usd(d["pv_fcff"], 0))
    line("PV of terminal value", usd(d["pv_tv"], 0))
    line("Terminal value share of enterprise value", pct(d["tv_share"], 0))
    line("Equity value (EV + cash - debt)", usd(d["equity_value"], 0))
    line("DCF value per share", usd(dcf_val))
    grid = sensitivity_grid(cfg, w.wacc)
    print("\n  Sensitivity: value per share by WACC (rows) and terminal growth (columns)")
    with pd.option_context("display.float_format", "${:,.0f}".format, "display.width", WIDTH):
        print(grid.to_string())

    # 5. Reverse DCF -----------------------------------------------------------
    b = cfg["dcf_base_case"]
    base_avg = float(np.mean(b["revenue_growth"]))
    g5 = base_avg + implied_growth(cfg, w.wacc, price)
    g_term = implied_terminal_growth(cfg, w.wacc, price)
    rw_g = cfg.get("reverse_dcf", {}).get("runway_growth", 0.10)
    rw_years = implied_runway_years(cfg, w.wacc, price, rw_g)
    m_need = implied_margin(cfg, w.wacc, price)
    banner("4. WHAT IS THE PRICE ASSUMING?  (reverse DCF, four ways)")
    print(f"  Base case: {base_avg:.0%} average growth for 5 years, {b['terminal_growth']:.0%} after,"
          f" {b['ebit_margin'][-1]:.1%} margin")
    line("1. Growth for 5 yrs, nothing else changed", pct(g5))
    line("2. Growth forever after 2031", "above WACC" if _bad(g_term) else pct(g_term))
    line(f"3. Years of {rw_g:.0%} growth before slowing",
         f"over 60" if rw_years is None else f"{rw_years}")
    line("4. 2031 operating margin, base-case growth", pct(m_need))
    print("  For reference: FY2026 net sales grew 10.1%; revenue has compounded ~8.7%")
    print("  a year since FY2004; FY2026 operating margin was 3.85%.")

    # 6. Monte Carlo ------------------------------------------------------------
    mc_dcf = monte_carlo_dcf(cfg, w.wacc, n, rng)
    s_dcf = bands(mc_dcf)
    share_below = float((mc_dcf < price).mean())

    target = round(dcf_val)
    upside = target / price - 1
    rating = rate(upside, cfg["rating_rule"])

    mc_px = monte_carlo_price(price, md.volatility, w.cost_of_equity, n,
                              sim["horizon_trading_days"], rng)
    s_px = bands(mc_px)
    prob_hit = float((mc_px <= target).mean())

    banner(f"5. MONTE CARLO STRESS TEST  ({n:,} runs each)")
    print("  (a) What COST is worth: DCF re-run with growth, margin, WACC and terminal g")
    print("      drawn from the Bear / Base / Bull ranges in assumptions.json")
    line("    Bear (10th percentile)", usd(s_dcf["bear_p10"]))
    line("    Base (median)", usd(s_dcf["base_p50"]))
    line("    Bull (90th percentile)", usd(s_dcf["bull_p90"]))
    line("    Simulations valuing COST below today's price", pct(share_below))
    print(f"\n  (b) Where the stock trades in 1 year: {n:,} price paths")
    line("    Annualised volatility", pct(md.volatility))
    line("    Bear / Base / Bull price",
         f"{usd(s_px['bear_p10'], 0)} / {usd(s_px['base_p50'], 0)} / {usd(s_px['bull_p90'], 0)}")
    line(f"    Probability price <= {usd(target, 0)} target within 1 year", pct(prob_hit))

    # 7. Call -------------------------------------------------------------------
    banner("BOTTOM LINE")
    line("Rating", rating)
    line("Target price (base-case DCF)", usd(target, 0))
    line("Upside / (downside)", pct(upside))
    runway_txt = "over 60" if rw_years is None else str(rw_years)
    print(f"\n  To justify the price, Costco needs about {runway_txt} years of {rw_g:.0%} growth, or"
          f"\n  {pct(g_term)} growth forever after 2031. {share_below:.0%} of {n:,} simulated DCFs"
          f"\n  land below today's price.")
    odds = "under 1%" if prob_hit < 0.01 else f"about {prob_hit:.0%}"
    print(f"  But a valuation gap is not a timing call: at its normal volatility the stock has\n"
          f"  {odds} odds of trading at the target within a year.")

    # Save ----------------------------------------------------------------------
    run = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_as_of": md.as_of, "mode": md.mode, "price": price,
        "risk_free_rate": md.risk_free_rate, "beta": md.beta, "wacc": w.wacc,
        "terminal_growth": cfg["dcf_base_case"]["terminal_growth"], "dcf_value": dcf_val,
        "target_price": target, "rating": rating, "upside": upside,
        "implied_5y_growth": g5, "implied_terminal_growth": g_term,
        "implied_runway_years": rw_years, "implied_2031_margin": m_need,
        "comps_pe_value": implied.get("Peer P/E"), "comps_ev_ebitda_value": implied.get("Peer EV/EBITDA"),
        "mc_dcf_p10": s_dcf["bear_p10"], "mc_dcf_p50": s_dcf["base_p50"], "mc_dcf_p90": s_dcf["bull_p90"],
        "mc_share_below_price": share_below, "volatility": md.volatility,
        "prob_price_below_target_1y": prob_hit,
    }
    rid = save_run(Path(args.db), run, comps, d["table"],
                   {"dcf_intrinsic_value": s_dcf, "price_1y": s_px}, n)
    print(f"\n  Saved as run #{rid} in {Path(args.db).name}")

    if args.charts:
        ranges = [("DCF Monte Carlo (P10-P90)", s_dcf["bear_p10"], s_dcf["bull_p90"]),
                  ("DCF sensitivity (WACC x g)", float(grid.values.min()), float(grid.values.max())),
                  ("1-yr trading range (P10-P90)", s_px["bear_p10"], s_px["bull_p90"])]
        if implied:
            vals = [v for v in implied.values() if not _bad(v)]
            if vals:
                ranges.insert(2, ("Peer multiples (P/E, EV/EBITDA)", min(vals), max(vals)))
        for p in save_charts(HERE / "charts", price, target, mc_dcf, ranges, md.as_of):
            print(f"  Chart saved: {p.relative_to(HERE)}")
    print()
    return run


if __name__ == "__main__":
    main()
    sys.exit(0)
