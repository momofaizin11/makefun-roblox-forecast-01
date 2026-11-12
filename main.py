#!/usr/bin/env python3
"""Forecast Roblox's "Up and Coming" chart (top 10) for the 72 hours after the deadline.

Reads MAKEFUN_INPUT_DIR (default /input) for task.json + history.ndjson and writes
MAKEFUN_OUTPUT_DIR/forecast.json (default /output/forecast.json).

Deterministic: no network, no wall clock, no randomness.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone

# --- model constants -------------------------------------------------------
HORIZON_H = 72.0          # hours covered by the window (f saturates at 1)
CANDIDATES = 30           # games considered from the last read's chart
MAX_ROWS_PER_PLACE = 100
GROWTH_LOOKBACK = 10      # reads searched for the growth pair
MOM_LOOKBACK = 48         # reads searched per momentum window
MOM_WINDOWS = ((6.0, 1.0), (18.0, 0.5))   # (hours back, weight) pairs

# Two-table kernel, indexed by |candidate rank - place|, index 10 = distance >= 10.
KN = [1.55, 1.55, 0.6777, 0.5305, 0.3189, 0.3189, 0.2551, 0.0440, 0.0024, 0.0024, 0.0]
KF = [2.9750, 0.8752, 0.8752, 0.6703, 0.5866, 0.5866, 0.5617, 0.3932, 0.2753, 0.2753, 0.2203]

# factor coefficients, ordered (growth, logplay, firstseen, other, momentum, playrank)
W = [0.2, 0.2, -0.7, -0.3, 0.3, -0.8]
CENTRE = [0.0, 0.0, 5.0, 3.0, 0.0, 0.0]
SCALE = [1.0, 2.0, 5.0, 2.0, 1.0, 3.0]

PM = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 1.0, 0.85, 0.85]   # per-place mass cap
DECAY0 = 0.86
DECAY1 = -0.05

# --- p4 model constants, ported verbatim from work/params_p4.json -----------
# Blend of an empirical rank->place transition table (alpha), a game-identity
# occupancy prior (beta) and the v4 parametric kernel (rest), times the p2
# 16-feature factor. See work/p4.py and work/core_p3.py.
P4_KN = [1.55, 1.55, 0.6777, 0.5305, 0.3189, 0.3189, 0.2551, 0.044, 0.0024, 0.0024, 0.0]
P4_KF = [2.975, 0.8752, 0.8752, 0.6703, 0.5866, 0.5866, 0.5617, 0.3932, 0.2753, 0.2753, 0.2203]
P4_W = [0.1, -0.05, -0.6, -0.3, 0.2, -0.65,
        0.0, 0.0, 0.0, 0.3, 0.0, -0.15, 0.0, 0.05, 0.0, -0.05]
P4_CENTRE = [0.0, 0.0, 5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             2.0, 0.3, 0.3, 2.0]
P4_SCALE = [1.0, 2.0, 5.0, 2.0, 1.0, 3.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5,
            2.0, 0.3, 0.3, 1.5]
P4_D0 = 0.86
P4_D1 = -0.08
P4_CAND = 30
P4_PM = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 1.0, 0.85, 0.85]
P4_MK = 18
P4_ALPHA = 0.3
P4_BETA = 0.2
P4_LAM = 3.0
P4_OCC_TAU_H = 12.0
P4_KC = 8.0
P4_RMAX = 45                       # cap on the rank index used by the table
P4_SLOTS = 8                       # sub-slot approximation (shipped)
P4_BUCKET_H = ((0.0, 6.0), (6.0, 18.0), (18.0, 36.0), (36.0, 54.0), (54.0, 80.0))
P4_MAX_LAG_H = 80.0
P4_CHARTS = ("explore:top-trending", "explore:top-playing-now",
             "explore:top-revisited", "explore:fun-with-friends")
P2_MOM_WINDOWS = ((6.0, 1.0), (18.0, 0.5))


def parse_ts(value):
    """ISO-8601 (with Z or offset) -> aware UTC datetime, or None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chart_rows(rows):
    """Normalise a chart list to [{'uid', 'rank', 'playing'}...] keeping source order."""
    out = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        uid = _int_or_none(row.get("universe_id"))
        if uid is None:
            continue
        out.append({
            "uid": uid,
            "rank": _int_or_none(row.get("rank")),
            "playing": _int_or_none(row.get("playing")),
        })
    return out


def normalize_read(rec):
    """One raw history record -> internal read dict, or None when unusable."""
    if not isinstance(rec, dict):
        return None
    at = parse_ts(rec.get("at"))
    rows = _chart_rows(rec.get("up_and_coming"))
    if at is None or not rows:
        return None
    details = rec.get("details") if isinstance(rec.get("details"), dict) else {}
    by_id = {}
    for pos, row in enumerate(rows):
        if row["uid"] not in by_id:
            by_id[row["uid"]] = {
                "rank": row["rank"],
                "pos": pos,
                "playing": row["playing"],
            }
    return {
        "at": at,
        "rows": rows,
        "by_id": by_id,
        "trending": _chart_rows(rec.get("top_trending")),
        "playing_now": _chart_rows(rec.get("top_playing_now")),
        "details": details,
    }


def load_history(path):
    """Read history.ndjson into sorted, usable reads."""
    reads = []
    if not os.path.exists(path):
        return reads
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            read = normalize_read(rec)
            if read is not None:
                reads.append(read)
    reads.sort(key=lambda r: r["at"])
    return reads


def load_task(path):
    task = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                task = loaded
        except Exception:
            task = {}

    places = _int_or_none(task.get("places"))
    if places is None or places < 1:
        places = 10
    slots = None
    window = task.get("window")
    if isinstance(window, dict):
        slots = _int_or_none(window.get("slots"))
    if slots is None or slots < 1:
        slots = 72
    return {"places": places, "slots": slots}


def ordered_chart(read):
    """The read's up_and_coming order, sorted by rank when ranks are present."""
    rows = read["rows"]
    if any(row["rank"] is not None for row in rows):
        return sorted(
            rows,
            key=lambda r: (r["rank"] if r["rank"] is not None else 10 ** 9),
        )
    return list(rows)


# --- model -----------------------------------------------------------------

def _rank_in(read, uid):
    """The uid's rank in a read, or None when it is absent / rankless."""
    entry = read["by_id"].get(uid)
    return entry["rank"] if entry is not None else None


def _play_in(read, uid):
    """The uid's playing count in a read, or None when it is absent."""
    entry = read["by_id"].get(uid)
    return entry["playing"] if entry is not None else None


def growth_of(uid, reads, index, p_new):
    """Log player growth per 24h, from the newest read at least 6h older, clamped."""
    if p_new is None or p_new <= 0:
        return 0.0
    last_at = reads[index]["at"]
    older = []
    for idx in range(max(0, index - GROWTH_LOOKBACK), index):
        playing = _play_in(reads[idx], uid)
        if playing is not None and playing > 0:
            older.append((reads[idx]["at"], playing))
    if not older:
        return 0.0
    pick = None
    for at, playing in reversed(older):
        if (last_at - at).total_seconds() >= 6 * 3600:
            pick = (at, playing)
            break
    if pick is None:
        pick = older[0]
    hours = max(1.0, (last_at - pick[0]).total_seconds() / 3600.0)
    g = math.log(p_new / pick[1]) / hours * 24.0
    return max(-1.5, min(1.5, g))


def logplay_of(uid, last, p_new):
    """Log playing minus the log median playing of the chart, clamped."""
    plays = [
        entry["playing"]
        for entry in last["by_id"].values()
        if entry["playing"] is not None and entry["playing"] > 0
    ]
    median = statistics.median(plays) if plays else 0
    base = math.log1p(median) if median else 0.0
    cur = math.log1p(p_new) if (p_new is not None and p_new > 0) else 0.0
    return max(-3.0, min(3.0, cur - base))


def firstseen_of(uid, reads, index):
    """Days since the uid first appeared in the chart, clamped to [0, 10]."""
    last_at = reads[index]["at"]
    for read in reads[: index + 1]:
        if uid in read["by_id"]:
            return min(10.0, (last_at - read["at"]).total_seconds() / 86400.0)
    return 10.0


def other_of(uid, last):
    """ln(1 + best rank on top_trending / top_playing_now), 0 if absent."""
    ranks = []
    for row in last["trending"] + last["playing_now"]:
        if row["uid"] == uid and row["rank"] is not None:
            ranks.append(row["rank"])
    return math.log1p(min(ranks)) if ranks else 0.0


def momentum_of(uid, reads, index, rank_now):
    """Rank improvement over the 6h and 18h windows, weighted and clamped."""
    last_at = reads[index]["at"]
    mom = 0.0
    for hours, weight in MOM_WINDOWS:
        for j in range(index - 1, max(-1, index - MOM_LOOKBACK), -1):
            if (last_at - reads[j]["at"]).total_seconds() >= hours * 3600.0:
                old_rank = _rank_in(reads[j], uid)
                if old_rank is not None:
                    mom += weight * (old_rank - rank_now) / 5.0
                break
    return max(-2.0, min(2.0, mom))


def playrank_of(uid, last):
    """Position when the last chart is ordered by playing, mapped and clamped."""
    by_play = sorted(
        last["rows"],
        key=lambda r: (-(r["playing"] if r["playing"] is not None else 0),
                       r["rank"] if r["rank"] is not None else 10 ** 9),
    )
    for pos, row in enumerate(by_play):
        if row["uid"] == uid:
            return max(-3.0, min(3.0, (pos + 1 - 25) / 25.0))
    return 10.0 / 25.0


def factor_of(x):
    """Multiplicative per-game factor from the six centred features."""
    e = 0.0
    for i in range(6):
        e += W[i] / SCALE[i] * (x[i] - CENTRE[i])
    return math.exp(max(-4.0, min(4.0, e)))


def build_candidates(reads):
    """The last read's top games (rank order), each with its factor."""
    index = len(reads) - 1
    last = reads[index]
    order = ordered_chart(last)
    cands = []
    seen = set()
    for pos, row in enumerate(order):
        uid = row["uid"]
        if uid in seen:
            continue
        seen.add(uid)
        rank = row["rank"] if row["rank"] is not None else pos + 1
        p_new = row["playing"]
        x = (
            growth_of(uid, reads, index, p_new),
            logplay_of(uid, last, p_new),
            firstseen_of(uid, reads, index),
            other_of(uid, last),
            momentum_of(uid, reads, index, rank),
            playrank_of(uid, last),
        )
        cands.append({"uid": uid, "rank": rank, "factor": factor_of(x)})
        if len(cands) >= CANDIDATES:
            break
    return cands


def place_weights(cands, place, f):
    """One place's unnormalised candidate weights at horizon fraction f in [0, 1]."""
    weights = []
    for cand in cands:
        d = min(abs(cand["rank"] - place), 10)
        kernel = KN[d] + (KF[d] - KN[d]) * f
        weights.append(kernel * cand["factor"])
    return weights


def top10_of(last, reads):
    """First 10 distinct ids of the last read's order, padded from that order /
    from older reads only if it holds fewer than 10 distinct ids."""
    ids = []
    for read in [last] + [r for r in reversed(reads) if r is not last]:
        for row in ordered_chart(read):
            if row["uid"] not in ids:
                ids.append(row["uid"])
            if len(ids) >= 10:
                return ids[:10]
    return ids[:10]


def forecast(task, reads):
    """Compute the forecast dict from a usable, sorted read list."""
    places = task["places"]
    slots = task["slots"]

    if not reads:
        return {"probabilities": [], "top10": []}

    cands = build_candidates(reads)
    n = len(cands)
    totals = [[0.0] * n for _ in range(places + 1)]

    for slot in range(slots):
        h = (slot + 0.5) / slots * HORIZON_H
        f = math.sqrt(max(0.0, min(1.0, h / HORIZON_H)))
        decay = DECAY0 + DECAY1 * f
        for place in range(1, places + 1):
            weights = place_weights(cands, place, f)
            total = 0.0
            for w in weights:
                total += w
            if total <= 0.0:
                continue
            pm = PM[place - 1] if place - 1 < len(PM) else PM[-1]
            mult = decay * pm / total / slots
            acc = totals[place]
            for idx in range(n):
                acc[idx] += mult * weights[idx]

    probabilities = []
    for place in range(1, places + 1):
        acc = totals[place]
        rows = []
        for idx, cand in enumerate(cands):
            p = math.floor(acc[idx] * 1e9) / 1e9
            if p <= 0.0:
                continue
            rows.append({"universe_id": int(cand["uid"]), "place": int(place), "p": p})
        rows.sort(key=lambda r: (-r["p"], r["universe_id"]))
        probabilities.extend(rows[:MAX_ROWS_PER_PLACE])

    return {"probabilities": probabilities, "top10": [int(u) for u in top10_of(reads[-1], reads)]}


# --- p4 model (ported from work/p4.py + work/core_p3.py + work/p2.py) -------
# One forecaster per (place, sub-slot): a blend of an empirical rank->place
# transition table, a game-identity occupancy prior and the v4 kernel, times the
# p2 16-feature factor.  Everything is causal (built only from reads before the
# anchor) and null-tolerant: a missing feature degrades to its neutral default.

def p4_bucket_of_h(hours):
    """Lag bucket index for a lag in hours (core_p3.bucket_of_h)."""
    for b, (lo, hi) in enumerate(P4_BUCKET_H):
        if lo <= hours < hi:
            return b
    return len(P4_BUCKET_H) - 1


def p4_read_map(read):
    """A normalised read -> {t, uc, rank, playing, other} with positions as ranks."""
    order = ordered_chart(read)
    rank = {}
    playing = {}
    uc = []
    for pos, row in enumerate(order):
        uid = row["uid"]
        if uid in rank:
            continue
        r = row["rank"]
        if not isinstance(r, int) or r < 1:
            r = pos + 1
        rank[uid] = r
        p = row["playing"]
        playing[uid] = p if isinstance(p, int) and p > 0 else 0
        uc.append(uid)
    other = {}
    for chart, rows in (("explore:top-trending", read["trending"]),
                        ("explore:top-playing-now", read["playing_now"])):
        for pos, row in enumerate(rows):
            uid = row["uid"]
            r = row["rank"]
            if not isinstance(r, int) or r < 1:
                r = pos + 1
            other.setdefault(uid, {})[chart] = r
    return {"t": read["at"].timestamp(), "uc": uc, "rank": rank,
            "playing": playing, "other": other}


def p4_created(maps):
    """uid -> timestamp of its earliest appearance in the history."""
    created = {}
    for m in maps:
        for u in m["uc"]:
            created.setdefault(u, m["t"])
    return created


def p4_rank_before(maps, i, uid, hours):
    """Rank of uid in the newest read at least `hours` older than read i."""
    t = maps[i]["t"]
    for j in range(i - 1, max(-1, i - 48), -1):
        if t - maps[j]["t"] >= hours * 3600.0:
            return maps[j]["rank"].get(uid)
    return None


def p4_growth_win(maps, i, uid, lo_h, maxk=18):
    """Daily playing log-growth using the newest reference at least lo_h old."""
    ser = []
    for k in range(max(0, i - maxk), i + 1):
        if uid in maps[k]["playing"]:
            ser.append((maps[k]["t"], maps[k]["playing"][uid]))
    if len(ser) < 2:
        return 0.0
    t1, p1 = ser[-1]
    if not p1:
        return 0.0
    for t0, p0 in reversed(ser[:-1]):
        if p0 and (t1 - t0) >= lo_h * 3600.0:
            return math.log(p1 / p0) / ((t1 - t0) / 3600.0) * 24.0
    return 0.0


def p4_rank_series(maps, i, uid, maxk=18):
    out = []
    for k in range(max(0, i - maxk), i + 1):
        r = maps[k]["rank"].get(uid)
        if r:
            out.append((maps[k]["t"], r))
    return out


def p4_feats(maps, created, i, uid):
    """The 16 p2 features for one game at read i (mirrors core10.Data/p2.Data)."""
    s = maps[i]
    r_now = s["rank"].get(uid)
    # growth over a >=6h window (core10.Data.feats)
    ser = []
    for k in range(max(0, i - 10), i + 1):
        if uid in maps[k]["playing"]:
            ser.append((maps[k]["t"], maps[k]["playing"][uid]))
    g21 = 0.0
    if len(ser) >= 2:
        t1, p1 = ser[-1]
        ref = None
        for t0, p0 in reversed(ser[:-1]):
            if p0 and (t1 - t0) >= 6 * 3600.0:
                ref = (t0, p0)
                break
        if ref is None:
            ref = ser[0]
        t0, p0 = ref
        if p0 and p1:
            dt = max(1.0, (t1 - t0) / 3600.0)
            g21 = math.log(p1 / p0) / dt * 24.0
    growth = max(-1.5, min(1.5, g21))
    # days since first seen (capped at 10)
    c = created.get(uid)
    age = min(10.0, (s["t"] - c) / 86400.0) if c is not None else 10.0
    # best rank over the other public charts
    others = [v for v in s["other"].get(uid, {}).values() if v]
    other = math.log1p(min(others)) if others else 0.0
    # rank momentum over the 6h / 18h windows
    mom = 0.0
    if r_now is not None:
        for hours, scale in P2_MOM_WINDOWS:
            pr = p4_rank_before(maps, i, uid, hours)
            if pr:
                mom += scale * (pr - r_now) / 5.0
    momentum = max(-2.0, min(2.0, mom))
    # log playing relative to the chart's median
    plays = [p for p in s["playing"].values() if p]
    med = statistics.median(plays) if plays else 0
    logplay = math.log1p(s["playing"].get(uid) or 0) - (math.log1p(med) if med else 0.0)
    # rank if the chart were ordered by playing
    order = sorted(s["rank"], key=lambda x: -(s["playing"].get(x) or 0))
    pr = order.index(uid) + 1 if uid in order else 50
    playrank = max(-3.0, min(3.0, (pr - 25) / 25.0))
    # rank lags (frozen at w=0, computed for fidelity)
    def lag(hours):
        pr = p4_rank_before(maps, i, uid, hours)
        if pr is None or r_now is None:
            return 0.0
        return max(-6.0, min(6.0, pr - r_now))
    lag12, lag24, lag48 = lag(12.0), lag(24.0), lag(48.0)
    # longer-window playing growth + acceleration
    g24 = max(-1.5, min(1.5, p4_growth_win(maps, i, uid, 18.0)))
    g48 = max(-1.5, min(1.5, p4_growth_win(maps, i, uid, 36.0)))
    accel = max(-1.5, min(1.5, g24 - g48))
    # rank volatility over the last <=48h
    rs = [(t, r) for t, r in p4_rank_series(maps, i, uid) if t >= s["t"] - 48 * 3600.0]
    vol = 0.0
    if len(rs) >= 3:
        vol = min(6.0, statistics.pstdev([r for _, r in rs]))
    other_map = s["other"].get(uid, {})

    def rankok(chart):
        r = other_map.get(chart)
        return (51 - min(r, 51)) / 50.0 if r else 0.0

    trendok = rankok("explore:top-trending")
    playok = rankok("explore:top-playing-now")
    ncharts = float(1 + sum(1 for ch in P4_CHARTS if other_map.get(ch)))
    return (growth, logplay, age, other, momentum, playrank,
            lag12, lag24, lag48, g24, g48, accel, vol, trendok, playok, ncharts)


def p4_factor(f):
    """p2 factor: exp(clamp(sum w[i]/scale[i]*(f[i]-centre[i]), -4, 4))."""
    x = 0.0
    for i in range(16):
        x += (P4_W[i] / P4_SCALE[i]) * (f[i] - P4_CENTRE[i])
    return math.exp(max(-4.0, min(4.0, x)))


def p4_table(maps, upto):
    """Empirical P(place | rank, lag bucket) from read pairs strictly before `upto`."""
    nb = len(P4_BUCKET_H)
    cnt = [[[0.0] * 11 for _ in range(P4_RMAX + 1)] for _ in range(nb)]
    for j in range(upto):
        sj = maps[j]
        for jp in range(j + 1, upto):
            dh = (maps[jp]["t"] - sj["t"]) / 3600.0
            if dh <= 0.0 or dh > P4_MAX_LAG_H:
                continue
            b = p4_bucket_of_h(dh)
            ranks_p = maps[jp]["rank"]
            for u, r in sj["rank"].items():
                rr = r if r < P4_RMAX else P4_RMAX
                rp = ranks_p.get(u)
                col = rp if (rp is not None and rp <= 10) else 0
                cnt[b][rr][col] += 1.0
    marg = []
    for b in range(nb):
        m = [0.0] * 11
        for r in range(P4_RMAX + 1):
            row = cnt[b][r]
            for k in range(11):
                m[k] += row[k]
        tot = sum(m) or 1.0
        marg.append([x / tot for x in m])
    return {"cnt": cnt, "marg": marg}


def p4_prob(table, b, r, k, lam):
    """P(place k | rank r, bucket b) with Dirichlet shrinkage toward the marginal."""
    r = min(max(1, r), P4_RMAX)
    row = table["cnt"][b][r]
    tot = sum(row)
    prior = table["marg"][b][k]
    return (row[k] + lam * prior) / (tot + lam)


def p4_occ(maps, upto, tau_h):
    """Time-decayed count of past top-10 occupancy per game per place (before `upto`)."""
    occ = {}
    ti = maps[upto]["t"]
    for j in range(upto):
        dt = (ti - maps[j]["t"]) / 3600.0
        w = math.exp(-dt / tau_h)
        if w < 1e-4:
            continue
        for u, r in maps[j]["rank"].items():
            if r <= 10:
                occ.setdefault(u, [0.0] * 11)[r] += w
    return occ


def forecast_p4(task, reads):
    """The p4 forecast dict; raises on anything the caller should fall back from."""
    places = task["places"]
    if places != 10 or len(reads) < 2:
        raise ValueError("p4 needs 10 places and at least 2 reads")
    i = len(reads) - 1
    maps = [p4_read_map(r) for r in reads]
    created = p4_created(maps)
    s = maps[i]
    cand = s["uc"][:P4_CAND]
    if not cand:
        raise ValueError("no candidates")
    table = p4_table(maps, i)
    occ = p4_occ(maps, i, P4_OCC_TAU_H)
    n = i
    a_eff = P4_ALPHA * (n / (n + P4_KC)) if n > 0 else 0.0
    F = {u: p4_factor(p4_feats(maps, created, i, u)) for u in cand}
    slots = P4_SLOTS
    acc = {k: {} for k in range(1, places + 1)}
    for slot in range(slots):
        h = (slot + 0.5) / slots * HORIZON_H
        f = math.sqrt(min(1.0, max(0.0, h / HORIZON_H)))
        b = p4_bucket_of_h(h)
        decay = P4_D0 + P4_D1 * f
        for k in range(1, places + 1):
            tocc = 0.0
            for u in cand:
                e = occ.get(u)
                if e is not None:
                    tocc += e[k]
            w = {}
            for u in cand:
                r = s["rank"][u]
                dd = min(abs(r - k), 10)
                kv4 = P4_KN[dd] + (P4_KF[dd] - P4_KN[dd]) * f
                pemp = p4_prob(table, b, r, k, P4_LAM)
                e = occ.get(u)
                pocc = (e[k] / tocc) if (e is not None and tocc > 0.0) else 0.0
                kern = a_eff * pemp + P4_BETA * pocc + (1.0 - a_eff - P4_BETA) * kv4
                w[u] = kern * F[u]
            top = dict(sorted(w.items(), key=lambda kv: -kv[1])[:P4_MK]) if P4_MK < len(w) else w
            tot = sum(top.values()) or 1.0
            scale = min(1.0, decay * P4_PM[k - 1]) / slots
            for u, v in top.items():
                acc[k][u] = acc[k].get(u, 0.0) + scale * v / tot
    probabilities = []
    for k in range(1, places + 1):
        rows = []
        for u, p in acc[k].items():
            if p <= 0.0:
                continue
            rows.append({"universe_id": int(u), "place": int(k),
                         "p": math.floor(p * 1e9) / 1e9})
        rows.sort(key=lambda r: (-r["p"], r["universe_id"]))
        probabilities.extend(rows[:MAX_ROWS_PER_PLACE])
    return {"probabilities": probabilities,
            "top10": [int(u) for u in top10_of(reads[-1], reads)]}


def payload_sane(payload):
    """The p4 output must satisfy the schema and the format sanity checks."""
    if not isinstance(payload, dict):
        return False
    probs = payload.get("probabilities")
    top = payload.get("top10")
    if not isinstance(probs, list) or not isinstance(top, list):
        return False
    sums = {}
    for row in probs:
        if not isinstance(row, dict):
            return False
        uid = row.get("universe_id")
        place = row.get("place")
        p = row.get("p")
        if isinstance(uid, bool) or not isinstance(uid, int):
            return False
        if isinstance(place, bool) or not isinstance(place, int):
            return False
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            return False
        if p != p or p < 0.0:            # NaN or negative
            return False
        if not 1 <= place <= 10:
            return False
        sums[place] = sums.get(place, 0.0) + float(p)
    if set(sums) != set(range(1, 11)):
        return False
    for tot in sums.values():
        if tot > 1.0 + 1e-9:
            return False
    if len(top) < 10:
        return False
    for u in top:
        if isinstance(u, bool) or not isinstance(u, int):
            return False
    return True


def emergency_forecast(task, reads):
    """Last-resort valid payload: the last read's top ids, 0.1 each at every place."""
    ids = []
    if reads:
        try:
            rows = ordered_chart(reads[-1])
        except Exception:
            rows = []
        for row in rows:
            if row["uid"] not in ids:
                ids.append(row["uid"])
            if len(ids) >= 10:
                break
    while len(ids) < 10:
        ids.append(1000000 + len(ids))
    probs = [{"universe_id": int(ids[place - 1]), "place": place, "p": 0.1}
             for place in range(1, 11)]
    return {"probabilities": probs, "top10": [int(u) for u in ids[:10]]}


def build_forecast(task, reads):
    """p4 when it validates, else the unchanged v4 path, else a trivial valid payload."""
    try:
        payload = forecast_p4(task, reads)
        if payload_sane(payload):
            return payload
    except Exception:
        pass
    try:
        payload = forecast(task, reads)
        if payload_sane(payload):
            return payload
    except Exception:
        pass
    return emergency_forecast(task, reads)


def write_output(out_dir, payload):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "forecast.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return path


def selftest():
    """Forecast a small synthetic history built in memory and assert the invariants."""
    base = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    reads = []
    for step in range(4):
        at = base.replace(hour=step * 2)
        rows = []
        for i in range(30):
            uid = 9000000 + i
            playing = 500 + 900 * i + 400 * step * (i % 4)
            rows.append({"universe_id": uid, "rank": i + 1, "playing": playing})
        trending = [{"universe_id": 9000000 + i, "rank": 40 - i} for i in range(0, 12)]
        reads.append(normalize_read({
            "at": at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "up_and_coming": rows,
            "top_trending": trending,
            "top_playing_now": [{"universe_id": 9000029, "rank": 3}],
        }))

    task = load_task(os.path.join(os.sep, "nonexistent", "task.json"))
    payload = forecast(task, reads)

    places = sorted({row["place"] for row in payload["probabilities"]})
    assert places == list(range(1, 11)), places

    per_place = {p: 0.0 for p in range(1, 11)}
    for row in payload["probabilities"]:
        assert 0.0 <= row["p"] <= 1.0, row
        per_place[row["place"]] += row["p"]
    for place, total in per_place.items():
        assert total <= 1.0 + 1e-12, (place, total)

    top = payload["top10"]
    assert len(top) == 10 and len(set(top)) == 10, top

    # --- the p4 path must satisfy the same invariants ---
    p4_payload = forecast_p4(task, reads)
    assert payload_sane(p4_payload), "p4 payload failed its sanity check"
    assert sorted({row["place"] for row in p4_payload["probabilities"]}) == list(range(1, 11))
    per_place = {p: 0.0 for p in range(1, 11)}
    for row in p4_payload["probabilities"]:
        assert 0.0 <= row["p"] <= 1.0, row
        per_place[row["place"]] += row["p"]
    for place, total in per_place.items():
        assert total <= 1.0 + 1e-9, (place, total)
    assert len(p4_payload["top10"]) == 10
    # the dispatcher picks p4 here and still returns a sane payload
    chosen = build_forecast(task, reads)
    assert payload_sane(chosen), "build_forecast produced an insane payload"
    assert chosen == p4_payload, "build_forecast should prefer the p4 path"

    print("selftest ok")


def main(argv):
    if "--selftest" in argv:
        selftest()
        return 0

    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

    task = load_task(os.path.join(in_dir, "task.json"))
    reads = load_history(os.path.join(in_dir, "history.ndjson"))
    payload = build_forecast(task, reads)
    write_output(out_dir, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
