"""STEP 5 - Evaluation of tracking.

MOTA  = 1 - (FN + FP + ID switches) / GT   -> detection-heavy
IDF1  = F1 of identity matches over time   -> association-heavy
HOTA  = sqrt(DetA * AssA)                  -> balances both (use the TrackEval repo)
These need ground truth in MOT format. Our pipeline writes tracks_mot.txt for that.
"""


def _load_mot(path):
    data = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            data.setdefault(int(float(p[0])), []).append((int(float(p[1])), *map(float, p[2:6])))
    return data


def evaluate_mot(gt_path, pred_path, iou_threshold=0.5):
    """Compare our tracks_mot.txt with a ground-truth MOT file. Needs: pip install motmetrics."""
    import motmetrics as mm

    gt, pred = _load_mot(gt_path), _load_mot(pred_path)
    acc = mm.MOTAccumulator(auto_id=False)
    for fr in sorted(set(gt) | set(pred)):
        g, p = gt.get(fr, []), pred.get(fr, [])
        dist = mm.distances.iou_matrix([b[1:] for b in g], [b[1:] for b in p],
                                       max_iou=1 - iou_threshold) if g and p else []
        acc.update([b[0] for b in g], [b[0] for b in p], dist, frameid=fr)
    s = mm.metrics.create().compute(acc, metrics=[
        "mota", "motp", "idf1", "idp", "idr", "num_switches", "num_fragmentations",
        "mostly_tracked", "mostly_lost", "num_false_positives", "num_misses"], name="seq")
    return {k: float(s[k].iloc[0]) for k in s.columns}
