"""Graph tests: each topic's graph functions run on small synthetic data and save into their topic folder."""
import numpy as np
import pytest

import config
from src import features as F
from src import figures as G


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    names = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"]

    def draw(n):
        y = np.repeat(np.arange(3), n)
        X = np.abs(rng.normal(size=(len(y), F.N_FEATURES)))
        for k in range(3):
            X[y == k, 3 * k:3 * k + 3] += 2.0
        return X, y

    X, y = draw(40)
    Xv, yv = draw(10)
    return {"X_train": X, "y_train": y, "X_valid": Xv, "y_valid": yv, "class_names": names}


@pytest.mark.parametrize("topic,name,make", [
    ("features", "1_segmentation_examples", lambda d, r: G.segmentation_examples(r)),
    ("features", "2_key_features_by_class", lambda d, r: G.feature_boxplots(d, r)),
    ("tree", "1_depth_vs_overfitting", lambda d, r: G.tree_depth_curve(d, r, 1)),
    ("tree", "2_tree_depth3", lambda d, r: G.tree_diagram(d, r)),
    ("rf", "1_accuracy_vs_number_of_trees", lambda d, r: G.rf_trees_curve(d, r, 1, {})),
    ("rf", "2_feature_importance", lambda d, r: G.rf_importance(d, r, 1, {})),
    ("boosting", "1_accuracy_per_round", lambda d, r: G.boosting_rounds(d, r)),
    ("knn", "1_accuracy_vs_k", lambda d, r: G.knn_k_curve(d, r, 1)),
    ("cv", "2_svm_grid_search", lambda d, r: G.svm_grid_heatmap(d, r, 1)),
    ("ablation", "2_leave_one_group_out", lambda d, r: G.leave_one_group_out(d, r, 1, {})),
])
def test_graph_is_saved_in_its_topic_folder(data, tmp_path, topic, name, make):
    path = make(data, tmp_path)
    assert path == config.figure_dir(topic, tmp_path) / f"{name}.png"
    assert path.exists() and path.stat().st_size > 10_000


def test_graphs_needing_results_are_skipped_without_them(tmp_path):
    assert G.model_comparison(tmp_path) is None
    assert G.cumulative_ablation(tmp_path) is None


def test_every_topic_has_a_numbered_folder():
    folders = list(config.FIG.values())
    assert len(set(folders)) == len(folders)
    assert all(f[:2].isdigit() for f in folders)
