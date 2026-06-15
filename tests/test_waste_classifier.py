from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from waste_classifier.features import extract_features
from waste_classifier.image_io import save_ppm
from waste_classifier.model import train_classifier


class WasteClassifierTests(unittest.TestCase):
    def test_feature_extraction_returns_expected_vector_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.ppm"
            save_ppm(path, 2, 2, [(255, 0, 0), (250, 5, 5), (240, 10, 10), (245, 8, 8)])

            features = extract_features(path)

        self.assertEqual(len(features), 12)
        self.assertGreater(features[0], features[1])
        self.assertGreater(features[9], 0.9)

    def test_knn_predicts_nearest_label(self) -> None:
        model = train_classifier(
            features=[[0.9, 0.1], [0.85, 0.15], [0.1, 0.8]],
            labels=["plastic", "plastic", "organic"],
            k=3,
        )

        prediction = model.predict([0.88, 0.12])

        self.assertEqual(prediction, "plastic")


if __name__ == "__main__":
    unittest.main()
