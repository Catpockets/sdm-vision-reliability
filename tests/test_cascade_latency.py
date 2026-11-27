import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_subset_cascade import cached_sdm_signals, metric_functions, streamed_signals


class CachedSDMSignalsTests(unittest.TestCase):
    def test_cached_signals_match_the_existing_exact_path(self):
        rng = np.random.default_rng(61)
        raw_support = rng.normal(size=(30, 16)).astype(np.float32)
        labels = rng.integers(0, 3, size=len(raw_support))
        raw_query = rng.normal(size=(13, 16)).astype(np.float32)
        predictions = rng.integers(0, 3, size=len(raw_query))
        support = torch.from_numpy(raw_support)
        functions = metric_functions()
        functions.update(
            DEVICE=torch.device("cpu"),
            reference={"targets": labels, "predictions": rng.integers(0, 3, size=len(raw_support))},
            embedding_mean=support.mean(), embedding_std=support.std(), neighbors=8,
        )
        functions["reference_embeddings"] = torch.from_numpy(functions["normalize_embeddings"](raw_support))
        cached = cached_sdm_signals(functions)
        for queries, classes in ((raw_query[:1], predictions[:1]), (raw_query, predictions)):
            with self.subTest(images=len(queries)):
                expected_distance, expected_q = streamed_signals(
                    functions, functions["normalize_embeddings"](queries), classes
                )
                observed_distance, observed_q = cached(torch.from_numpy(queries), classes)
                np.testing.assert_array_equal(observed_distance, expected_distance)
                np.testing.assert_array_equal(observed_q, expected_q)


if __name__ == "__main__":
    unittest.main()
