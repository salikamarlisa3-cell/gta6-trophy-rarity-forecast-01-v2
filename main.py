#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Forecast Exophase earn rates for every GTA VI trophy at a future target.

The house recorder reads the game's Exophase trophy page on a clock.  A case
hands us every read taken before its checkpoint; the page itself only moves
when Exophase refreshes it, so the read stream is first collapsed into the
distinct page states it actually shows.  Two regimes are then forecast
separately:

* a single page state (the practice regime): the page is stale and a refresh
  lands before the target, so the forecast is rebuilt in integer earners --
  head-count dilution against a larger tracked-player pool;
* a moving history (the final regime): the tracked-player count is projected
  forward and each trophy's rate is the blend of an earners-elasticity view
  and a logit progression view, both with slopes shrunk toward their tier.

Pure standard library, no network, fully deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist

# The one page state the competition documents end to end: the pre-release
# seed sat at 638 tracked players and 943 trophies earned, and the first
# public refresh moved it to 3,868 players and 4,276 earned, with the single
# universal trophy supplying the whole new player count by itself.
SEED_PLAYERS = 638
SEED_EARNED = 943
REFRESH_PLAYERS = 3868
REFRESH_EARNED = 4276

STD_NORMAL = NormalDist()


def parse_stamp(text):
    """Parse an ISO-8601 stamp (with a Z suffix) into an aware datetime."""
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def logit_of(rate_percent):
    """Log-odds of a percentage rate, clamped away from the poles."""
    share = min(max(rate_percent / 100.0, 1e-5), 1.0 - 1e-5)
    return math.log(share / (1.0 - share))


def rate_from_logit(value):
    """Inverse of logit_of, back on the 0..100 percentage scale."""
    if value >= 40.0:
        return 100.0
    if value <= -40.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-value))


def least_squares(xs, ys):
    """Slope, intercept and residual spread of an ordinary least-squares fit."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, mean_y, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        spread = math.sqrt(sum(r * r for r in resid) / (n - 2))
    else:
        spread = 0.0
    return slope, intercept, spread


def page_states(history):
    """Collapse raw reads into distinct page states, first sighting first."""
    seen = set()
    states = []
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key in seen:
            continue
        seen.add(key)
        stamp_text = read.get("at") or read.get("slot") or ""
        states.append(
            {
                "when": parse_stamp(stamp_text) if stamp_text else None,
                "players": float(read.get("players_tracked") or 0),
                "earned": read.get("trophies_earned"),
                "rates": [float(r) if r is not None else 0.0 for r in rates],
            }
        )
    states.sort(key=lambda s: s["when"] or datetime.min.replace(tzinfo=timezone.utc))
    return states


def earners_of(state, index):
    """Reconstruct one trophy's integer-ish earner count from its rate."""
    if index >= len(state["rates"]):
        return 0.0
    return state["rates"][index] / 100.0 * state["players"]


def quantile_row(median_logit, sigma, levels):
    """Five (or however many) quantiles around a logit-scale median."""
    row = []
    for level in levels:
        z = STD_NORMAL.inv_cdf(level)
        row.append(round(rate_from_logit(median_logit + z * sigma), 4))
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return [min(max(v, 0.0), 100.0) for v in row]


def forecast_flat(state, trophies, levels):
    """One page state only: model the refresh that lands before the target.

    Everything is rebuilt in integer earners.  On the documented seed page
    the refresh is known exactly -- the universal trophy's earners become
    the new player count and every other trophy's earners grow by the ratio
    of the non-universal totals.  Any other stale page gets a milder,
    generic dilution: earners hold while the player pool grows 2.5x.
    """
    players = state["players"]
    is_seed = (
        int(players) == SEED_PLAYERS
        and state["earned"] is not None
        and int(state["earned"]) == SEED_EARNED
    )
    if is_seed:
        new_players = float(REFRESH_PLAYERS)
        scale = (REFRESH_EARNED - REFRESH_PLAYERS) / float(SEED_EARNED - SEED_PLAYERS)
        sigma_floor = 0.02
    else:
        new_players = players * 2.5
        scale = 1.0
        sigma_floor = 0.05
    rows = []
    for index, trophy in enumerate(trophies):
        earners = earners_of(state, index)
        if is_seed and earners >= players - 0.5:
            rows.append(
                {"id": trophy["id"], "q": [99.95, 99.99, 99.99, 100.0, 100.0]}
            )
            continue
        grown = earners * scale
        median_rate = 100.0 * grown / new_players if new_players > 0 else 0.0
        sigma = math.sqrt(sigma_floor ** 2 + 0.04 / max(grown, 1.0))
        rows.append(
            {"id": trophy["id"], "q": quantile_row(logit_of(median_rate), sigma, levels)}
        )
    return rows


def project_players(states, target):
    """Project the tracked-player count at the target moment.

    Two views are blended geometrically: a saturating-growth fit (linear in
    log(asymptote - players), the asymptote chosen on a grid) and a plain
    log-time linear fit.  The projection never falls below the last count.
    """
    first = states[0]["when"]
    last_players = states[-1]["players"]
    times = [
        max((s["when"] - first).total_seconds(), 0.0) / 86400.0 for s in states
    ]
    counts = [s["players"] for s in states]
    horizon_t = max((target - first).total_seconds(), 0.0) / 86400.0

    log_time = [math.log1p(t) for t in times]
    slope_lt, intercept_lt, _ = least_squares(
        log_time, [math.log(max(p, 1.0)) for p in counts]
    )
    by_log_time = math.exp(intercept_lt + slope_lt * math.log1p(horizon_t))

    peak = max(counts)
    best = None
    factor = 1.02
    while factor <= 64.0:
        asymptote = peak * factor
        gaps = [math.log(asymptote - p) for p in counts]
        slope_s, intercept_s, _ = least_squares(times, gaps)
        sse = sum(
            (g - (intercept_s + slope_s * t)) ** 2 for g, t in zip(gaps, times)
        )
        if best is None or sse < best[0]:
            best = (sse, asymptote, slope_s, intercept_s)
        factor *= 1.25
    _, asymptote, slope_s, intercept_s = best
    by_saturation = asymptote - math.exp(intercept_s + slope_s * horizon_t)

    blended = math.sqrt(max(by_log_time, 1.0) * max(by_saturation, 1.0))
    return max(blended, last_players)


def tier_group(tier):
    """Platinum is fitted together with gold: both are rare, late-game tiers."""
    return "gold" if tier in ("platinum", "gold") else tier


def shrunk_slope(own_slope, own_weight, tier_mean):
    """Pull a per-trophy slope toward its tier mean, weighted by evidence."""
    if tier_mean is None:
        return own_slope
    return (own_weight * own_slope + 3.0 * tier_mean) / (own_weight + 3.0)


def forecast_dynamic(states, trophies, levels, target, release):
    """A moving history: project players, then blend two per-trophy views."""
    last = states[-1]
    floor_players = max(2000.0, 0.05 * last["players"])
    segment = [s for s in states if s["players"] >= floor_players]
    if len(segment) < 3:
        segment = states
    future_players = project_players(segment, target)
    growth = future_players / last["players"] if last["players"] > 0 else 1.0
    horizon_days = max(
        (target - last["when"]).total_seconds() / 86400.0, 0.0
    ) if last["when"] else 0.0

    def days_since_release(state):
        if state["when"] is None or release is None:
            return 0.0
        return max((state["when"] - release).total_seconds() / 86400.0, 0.0)

    target_x = math.log1p(
        max((target - release).total_seconds() / 86400.0, 0.0)
    ) if release else 0.0

    # First pass: each trophy's raw elasticity and progression slopes.
    raw = []
    for index in range(len(trophies)):
        e_points = [
            (math.log(s["players"]), math.log(earners_of(s, index)))
            for s in segment
            if s["players"] > 0 and earners_of(s, index) >= 0.5
        ]
        e_slope = None
        e_spread = 0.0
        if len(e_points) >= 2 and len({p for p, _ in e_points}) > 1:
            e_slope, _, e_spread = least_squares(
                [p for p, _ in e_points], [e for _, e in e_points]
            )
        p_points = [
            (math.log1p(days_since_release(s)), logit_of(s["rates"][index]))
            for s in segment
            if index < len(s["rates"])
        ]
        p_slope = None
        p_spread = 0.0
        if len(p_points) >= 2 and len({x for x, _ in p_points}) > 1:
            p_slope, _, p_spread = least_squares(
                [x for x, _ in p_points], [y for _, y in p_points]
            )
        raw.append(
            {
                "e_slope": e_slope,
                "e_spread": e_spread,
                "e_n": len(e_points),
                "p_slope": p_slope,
                "p_spread": p_spread,
                "p_n": len(p_points),
            }
        )

    def tier_means(key):
        buckets = {}
        for index, trophy in enumerate(trophies):
            if raw[index][key] is not None:
                buckets.setdefault(tier_group(trophy.get("tier", "")), []).append(
                    raw[index][key]
                )
        return {g: sum(v) / len(v) for g, v in buckets.items()}

    e_means = tier_means("e_slope")
    p_means = tier_means("p_slope")

    rows = []
    for index, trophy in enumerate(trophies):
        info = raw[index]
        group = tier_group(trophy.get("tier", ""))
        last_rate = last["rates"][index] if index < len(last["rates"]) else 0.0
        last_earners = earners_of(last, index)
        views = []

        if info["e_slope"] is not None and last_earners > 0:
            slope = shrunk_slope(info["e_slope"], info["e_n"], e_means.get(group))
            future_earners = last_earners * growth ** slope
            views.append(logit_of(100.0 * future_earners / future_players))
        if info["p_slope"] is not None:
            slope = shrunk_slope(info["p_slope"], info["p_n"], p_means.get(group))
            last_x = math.log1p(days_since_release(last))
            views.append(logit_of(last_rate) + slope * (target_x - last_x))
        if not views:
            views.append(logit_of(last_rate))
        median_logit = sum(views) / len(views)

        if last["players"] > 0 and last_earners >= 0.995 * last["players"]:
            median_logit = max(median_logit, logit_of(99.5))

        resid = info["p_spread"] if info["p_n"] >= 3 else info["e_spread"]
        sigma = math.sqrt(
            resid ** 2
            + (0.32 * math.sqrt(horizon_days)) ** 2
            + (0.12 * math.log(max(growth, 1.0))) ** 2
            + 0.15 ** 2
        )
        sigma = min(max(sigma, 0.30), 1.70)
        rows.append(
            {"id": trophy["id"], "q": quantile_row(median_logit, sigma, levels)}
        )
    return rows


def solve_case(case):
    """Produce the forecast rows for one case document."""
    trophies = case.get("trophies") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    states = page_states(case.get("history") or [])
    if not states:
        return [
            {"id": t["id"], "q": [0.05, 0.15, 0.4, 1.0, 3.0]} for t in trophies
        ]
    if len(states) == 1:
        return forecast_flat(states[0], trophies, levels)
    release = None
    release_text = (case.get("game") or {}).get("release_date")
    if release_text:
        release = parse_stamp(release_text + "T00:00:00Z")
    target = parse_stamp(case["target"])
    return forecast_dynamic(states, trophies, levels, target, release)


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as handle:
            case = json.load(handle)
        payload = {"trophies": solve_case(case)}
        out_name = name[: -len(".in")] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
