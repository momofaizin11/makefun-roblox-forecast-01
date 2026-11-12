#!/usr/bin/env python3
"""Forecast Roblox's "Up and Coming" chart (top 10) for the 72 hours after the deadline.

Reads MAKEFUN_INPUT_DIR (default /input) for task.json + history.ndjson and writes
MAKEFUN_OUTPUT_DIR/forecast.json (default /output/forecast.json).

Deterministic: no network, no wall clock, no randomness.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

HORIZON = 259200.0        # 3 days in seconds (u saturates here)
CANDIDATES = 25           # games considered from the last read's chart
AGE_DEFAULT = 30.0        # days, when created is unknown
AGE_CAP = 60.0
GROWTH_LOOKBACK = 6       # reads searched for the growth pair
MAX_ROWS_PER_PLACE = 100


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


def find_created(uid, reads, last_at):
    """Days between last_at and the most recent known created timestamp."""
    for read in reversed(reads):
        entry = read["details"].get(str(uid))
        if not isinstance(entry, dict):
            continue
        created = parse_ts(entry.get("created"))
        if created is None:
            continue
        return min(AGE_CAP, (last_at - created).total_seconds() / 86400.0)
    return AGE_DEFAULT


def growth_of(uid, reads):
    """Log player growth per 24h from the last two usable appearances."""
    found = []
    for read in reversed(reads[-GROWTH_LOOKBACK:]):
        row = read["by_id"].get(uid)
        if row is None:
            continue
        playing = row["playing"]
        if playing is None or playing <= 0:
            continue
        found.append((read["at"], playing))
        if len(found) == 2:
            break
    if len(found) < 2:
        return 0.0
    (at1, p1), (at0, p0) = found[0], found[1]
    gap_hours = (at1 - at0).total_seconds() / 3600.0
    g = math.log(p1 / p0) / max(0.25, gap_hours) * 24.0
    return max(-1.5, min(1.5, g))


def other_of(uid, last):
    """ln(1 + best rank on top_trending / top_playing_now), 0 if absent."""
    ranks = []
    for rows in (last["trending"], last["playing_now"]):
        for row in rows:
            if row["uid"] == uid and row["rank"] is not None:
                ranks.append(row["rank"])
    if not ranks:
        return 0.0
    return math.log(1.0 + min(ranks))


def build_candidates(reads):
    last = reads[-1]
    order = ordered_chart(last)[:CANDIDATES]
    cands = []
    for pos, row in enumerate(order):
        uid = row["uid"]
        rank = row["rank"] if row["rank"] is not None else pos + 1
        playing = row["playing"] if row["playing"] is not None else 0
        logplay = math.log(1.0 + max(0, playing))
        age = find_created(uid, reads, last["at"])
        g = growth_of(uid, reads)
        other = other_of(uid, last)
        factor = math.exp(
            0.2 * g
            + 0.1 * (logplay - 8.5)
            - 0.02 * (age - 20.0)
            - 0.15 * (other - 3.0)
        )
        cands.append({"uid": uid, "rank": rank, "factor": factor})
    return cands


def place_weights(cands, place, horizon_seconds):
    u = min(1.0, max(0.0, horizon_seconds / HORIZON))
    f = math.sqrt(u)
    spread = 3.5 + 4.0 * f
    stay = 1.0
    decay = 0.8 - 0.05 * f

    weights = []
    for cand in cands:
        rank = cand["rank"]
        base = math.exp(-abs(rank - place) / spread)
        if rank == place:
            base += stay
        weights.append(base * cand["factor"])

    total = 0.0
    for w in weights:
        total += w
    if total <= 0.0:
        return [0.0] * len(cands)
    return [decay * w / total for w in weights]


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
    totals = [[0.0] * len(cands) for _ in range(places + 1)]

    for slot in range(slots):
        horizon = (slot + 1) * 3600.0
        for place in range(1, places + 1):
            probs = place_weights(cands, place, horizon)
            acc = totals[place]
            for i, p in enumerate(probs):
                acc[i] += p

    probabilities = []
    for place in range(1, places + 1):
        acc = totals[place]
        rows = []
        for i, cand in enumerate(cands):
            p = acc[i] / slots
            if p <= 0.0:
                continue
            p = math.floor(p * 1e9) / 1e9
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
    for step in range(3):
        at = base.replace(hour=step)
        rows = []
        for i in range(30):
            uid = 9000000 + i
            playing = 500 + 900 * i + 400 * step * (i % 4)
            rows.append({"universe_id": uid, "rank": i + 1, "playing": playing})
        details = {
            str(9000000 + i): {
                "created": "2026-09-{:02d}T00:00:00.000Z".format((i % 27) + 1),
            }
            for i in range(0, 30, 2)
        }
        trending = [{"universe_id": 9000000 + i, "rank": 40 - i} for i in range(0, 12)]
        reads.append(normalize_read({
            "at": at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "up_and_coming": rows,
            "top_trending": trending,
            "top_playing_now": [{"universe_id": 9000029, "rank": 3}],
            "details": details,
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