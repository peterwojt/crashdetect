import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanBoxTracker

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

class Sort:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.1):
        self.trackers = []
        self.frame_count = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t][:4] = pos
            trks[t][4] = 0
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        results = []
        for t in reversed(range(len(self.trackers))):
            trk = self.trackers[t]
            if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()
                results.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            if trk.time_since_update > self.max_age:
                self.trackers.pop(t)

        if len(results) > 0:
            return np.concatenate(results)
        return np.empty((0, 5))

def associate_detections_to_trackers(dets, trks, iou_threshold=0.3):
    if len(trks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), []

    iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)

    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(*matched_indices)))

    unmatched_dets = []
    for d in range(len(dets)):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    unmatched_trks = []
    for t in range(len(trks)):
        if t not in matched_indices[:, 1]:
            unmatched_trks.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)
