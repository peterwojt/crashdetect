import numpy as np

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        # State: [x, y, s, r, dx, dy, ds]
        x, y, s, r = self._convert_bbox_to_z(bbox)
        self.kf = self._init_kf()
        self.kf['x'][:4, 0] = np.array([x, y, s, r])
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def _init_kf(self):
        kf = {
            'x': np.zeros((7, 1)),
            'P': np.eye(7) * 10,
            'F': np.eye(7),
            'Q': np.eye(7),
            'H': np.eye(4, 7),
            'R': np.eye(4)
        }
        for i in range(4):
            kf['F'][i, i+3] = 1
        return kf

    def _convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r])

    def _convert_x_to_bbox(self, x):
        product = x[2] * x[3]
        w = np.sqrt(product) if product > 0 else 1e-6
        h = x[2] / (w + 1e-6)
        return np.array([
            x[0] - w / 2,
            x[1] - h / 2,
            x[0] + w / 2,
            x[1] + h / 2
        ])

    def update(self, bbox):
        z = self._convert_bbox_to_z(bbox)
        z = z.reshape((4, 1))

        # Kalman gain
        y = z - self.kf['H'] @ self.kf['x']
        S = self.kf['H'] @ self.kf['P'] @ self.kf['H'].T + self.kf['R']
        K = self.kf['P'] @ self.kf['H'].T @ np.linalg.inv(S)

        self.kf['x'] += K @ y
        self.kf['P'] = (np.eye(7) - K @ self.kf['H']) @ self.kf['P']

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def predict(self):
        self.kf['x'] = self.kf['F'] @ self.kf['x']
        self.kf['P'] = self.kf['F'] @ self.kf['P'] @ self.kf['F'].T + self.kf['Q']
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._convert_x_to_bbox(self.kf['x'].flatten())

    def get_state(self):
        return self._convert_x_to_bbox(self.kf['x'].flatten())
