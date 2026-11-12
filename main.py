#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 for the 72 hours after the deadline.

Reads /input/task.json and /input/history.ndjson (or $MAKEFUN_INPUT_DIR), writes
forecast.json to /output (or $MAKEFUN_OUTPUT_DIR). Standard library only, no network,
deterministic: the same history always produces the same bytes.

The model: a rank-diffusion kernel around each game's place when the history ends,
widened by the forecast horizon, times a few per-game factors (player growth, size,
age, presence on the other charts), renormalised at every place and multiplied by a
decay that leaves room for the games that will join the chart later.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

PLACES = 10
SLOTS = 72
HORIZON = 72 * 3600.0

# fitted on the public archive: kernel + per-game factors, decay per horizon
PARAMS = {
    "cand": 25,
    "s0": 3.5, "s1": 4.0,      # spread = s0 + s1 * sqrt(h / 72h)
    "st0": 1.0, "st1": 0.0,    # extra weight on a game that holds the place now
    "d0": 0.8, "d1": -0.05,    # decay = share of a place named, rest left unnamed
    "w_growth": 0.2, "w_logplay": 0.2, "w_age": -0.4, "w_other": -0.3,
}


def parse_ts(text):
    if not text:
        return None
    t = text.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(t).timestamp()
    except ValueError:
        return None


def read_history(path):
    reads = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            at = parse_ts(rec.get("at"))
            chart = rec.get("up_and_coming")
            if at is None or not isinstance(chart, list):
                continue
            order, playing, created = [], {}, {}
            for row in chart:
                if not isinstance(row, dict):
                    continue
                uid = row.get("universe_id")
                if not isinstance(uid, int):
                    continue
                rank = row.get("rank")
                if not isinstance(rank, int) or rank < 1:
                    rank = len(order) + 1
                order.append((rank, uid))
                play = row.get("playing")
                if isinstance(play, (int, float)):
                    playing[uid] = float(play)
            order.sort()
            details = rec.get("details")
            if isinstance(details, dict):
                for key, det in details.items():
                    if not isinstance(det, dict):
                        continue
                    try:
                        uid = int(key)
                    except (TypeError, ValueError):
                        continue
                    when = parse_ts(det.get("created"))
                    if when:
                        created[uid] = when
            others = {}
            for name in ("top_trending", "top_playing_now"):
                rows = rec.get(name)
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    uid, rank = row.get("universe_id"), row.get("rank")
                    if isinstance(uid, int) and isinstance(rank, int) and rank > 0:
                        others.setdefault(uid, []).append(rank)
            ids = [uid for _, uid in order]
            reads.append({
                "t": at,
                "ids": ids,
                "rank": {uid: i + 1 for i, uid in enumerate(ids)},
                "playing": playing,
                "created": created,
                "other": {uid: min(v) for uid, v in others.items() if v},
            })
    reads.sort(key=lambda r: r["t"])
    return reads


def growth_of(reads, u):
    """log growth of a game's players per 24h over the last two reads that list it."""
    seen = []
    for rec in reads[-6:]:
        play = rec["playing"].get(u)
        if play is not None:
            seen.append((rec["t"], play))
    if len(seen) < 2:
        return 0.0
    (t1, p1), (t0, p0) = seen[-1], seen[-2]
    if not p0 or not p1:
        return 0.0
    hours = max(0.25, (t1 - t0) / 3600.0)
    return math.log(p1 / p0) / hours * 24.0


def factors(reads, u, last):
    created = None
    for rec in reversed(reads):
        if u in rec["created"]:
            created = rec["created"][u]
            break
    age = (last["t"] - created) / 86400.0 if created else 30.0
    other = last["other"].get(u)
    return {
        "growth": max(-1.5, min(1.5, growth_of(reads, u))),
        "logplay": math.log1p(last["playing"].get(u) or 0.0),
        "age": min(60.0, age),
        "other": math.log1p(other) if other else 0.0,
    }


def spread_stay_decay(hour):
    f = max(0.0, min(1.0, hour / HORIZON)) ** 0.5
    return (PARAMS["s0"] + PARAMS["s1"] * f,
            PARAMS["st0"] + PARAMS["st1"] * f,
            PARAMS["d0"] + PARAMS["d1"] * f)


def forecast(reads, slots=SLOTS):
    last = reads[-1]
    cand = last["ids"][:PARAMS["cand"]]
    feats = {u: factors(reads, u, last) for u in cand}
    acc = {k: {} for k in range(1, PLACES + 1)}
    for slot in range(slots):
        spread, stay, decay = spread_stay_decay((slot + 1) * 3600.0)
        for k in range(1, PLACES + 1):
            w = {}
            for u in cand:
                f = feats[u]
                dist = abs(last["rank"][u] - k)
                v = math.exp(-dist / spread) + (stay if dist == 0 else 0.0)
                v *= math.exp(PARAMS["w_growth"] * f["growth"]
                              + PARAMS["w_logplay"] * (f["logplay"] - 8.5) / 2.0
                              + PARAMS["w_age"] * (f["age"] - 20.0) / 20.0
                              + PARAMS["w_other"] * (f["other"] - 3.0) / 2.0)
                w[u] = v
            total = sum(w.values()) or 1.0
            for u, v in w.items():
                acc[k][u] = acc[k].get(u, 0.0) + decay * v / total / slots
    return acc, cand


def main():
    inp = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    task = {}
    try:
        with open(os.path.join(inp, "task.json"), "r", encoding="utf-8") as fh:
            task = json.load(fh)
    except (OSError, ValueError):
        task = {}
    places = int(task.get("places") or PLACES)
    slots = int((task.get("window") or {}).get("slots") or SLOTS)

    reads = read_history(os.path.join(inp, "history.ndjson"))
    if not reads:
        raise SystemExit("no usable reads in the history")

    acc, cand = forecast(reads, slots)
    probabilities = []
    for k in range(1, places + 1):
        rows = acc.get(k, {})
        total = sum(rows.values())
        for u in sorted(rows):
            p = rows[u] if total <= 1.0 else rows[u] / total
            # round downward: no place may sum above 1 once the rows are rounded
            p = math.floor(p * 1e9) / 1e9
            probabilities.append({"universe_id": u, "place": k, "p": p})
    top10 = cand[:places]
    while len(top10) < places:
        for u in reads[-1]["ids"]:
            if u not in top10:
                top10.append(u)
                break

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "forecast.json"), "w", encoding="utf-8") as fh:
        json.dump({"probabilities": probabilities, "top10": top10}, fh,
                  separators=(",", ":"), sort_keys=False)
        fh.write("\n")


def selftest():
    """Check the output contract on a synthetic history: no input files, no network."""
    import tempfile
    reads = []
    for step in range(3):
        chart = [{"universe_id": 900000 + i, "rank": i + 1, "playing": 5000 + i * 40 + step * 10}
                 for i in range(30)]
        reads.append({"t": 1000.0 + step * 3600, "ids": [r["universe_id"] for r in chart],
                      "rank": {r["universe_id"]: r["rank"] for r in chart},
                      "playing": {r["universe_id"]: r["playing"] for r in chart},
                      "created": {}, "other": {}})
    acc, cand = forecast(reads, SLOTS)
    assert len(cand) == min(PARAMS["cand"], 30)
    for k, rows in acc.items():
        assert 1 <= k <= PLACES
        assert sum(rows.values()) <= 1.0 + 1e-12, (k, sum(rows.values()))
    assert sorted(acc) == list(range(1, PLACES + 1))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "forecast.json")
        payload = {"probabilities": [{"universe_id": u, "place": k, "p": math.floor(p * 1e9) / 1e9}
                                     for k, rows in sorted(acc.items()) for u, p in sorted(rows.items())],
                   "top10": cand[:PLACES]}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        back = json.load(open(path, encoding="utf-8"))
        by_place = {}
        for row in back["probabilities"]:
            by_place.setdefault(row["place"], []).append(row["p"])
        for k, ps in by_place.items():
            assert len(ps) <= 100 and 0.0 <= min(ps) and sum(ps) <= 1.0 + 1e-12
        assert len(back["top10"]) == PLACES == len(set(back["top10"]))
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        selftest()
    else:
        main()
