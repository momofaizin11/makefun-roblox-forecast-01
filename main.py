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

    print("selftest ok")


def main(argv):
    if "--selftest" in argv:
        selftest()
        return 0

    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

    task = load_task(os.path.join(in_dir, "task.json"))
    reads = load_history(os.path.join(in_dir, "history.ndjson"))
    payload = forecast(task, reads)
    write_output(out_dir, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
