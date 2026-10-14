"""Booster-tree tests on a tiny random BERT (no downloads), so they run in seconds."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from boostree.model.boost import (BoostConfig, BoostTrainer, Split, assign_leaves, describe_condition,  # noqa: E402
                                  grow_tree, text_features, tree_leaves)


def tiny_setup(n=120, seed=0):
    torch.manual_seed(seed)
    cfg = transformers.BertConfig(vocab_size=60, hidden_size=16, num_hidden_layers=2,
                                  num_attention_heads=2, intermediate_size=32,
                                  max_position_embeddings=32)
    model = transformers.BertModel(cfg).eval()
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    ids = torch.tensor(rng.integers(5, 60, size=(n, 12)))
    ids[:, 1] = torch.tensor(y + 1)  # make the label recoverable from token 1
    mask = torch.ones_like(ids)
    feats = np.column_stack([y.astype(float), rng.random(n)])  # feature 0 = true group
    split = Split(ids, mask, torch.tensor(y), feats, np.ones(n))
    head = torch.nn.Linear(16, 2)
    return model, head, split


def test_trainer_runs_and_never_increases_sample_loss():
    model, head, split = tiny_setup()
    cfg = BoostConfig(max_rounds=3, sample_size=80, batch_size=40, min_leaf=10, t_min=0.0)
    trainer = BoostTrainer(model, head, split, split, cfg, log=lambda *_: None)
    assert len(trainer.mods) == 4  # 2 layers x (attention, feed-forward)
    trainer.fit()
    kept = [r for r in trainer.rounds if "module" in r]
    assert kept, "expected at least one booster to be kept"
    for r in kept:
        assert r["sample_loss_after"] < r["sample_loss_before"]
        assert set(r["signal"]) == set(trainer.mods)      # every matrix is screened
        assert set(r["gains"]) <= set(trainer.mods)       # only the shortlist gets a gain


def test_shortlist_limits_gain_tests():
    model, head, split = tiny_setup(seed=3)
    cfg = BoostConfig(max_rounds=1, sample_size=60, batch_size=30, min_leaf=10, t_min=0.0, top_k=2)
    trainer = BoostTrainer(model, head, split, split, cfg, log=lambda *_: None)
    trainer.fit()
    assert len(trainer.rounds[0]["gains"]) == 2 and len(trainer.rounds[0]["signal"]) == 4
    assert trainer.predict("test").shape == (len(split.y),)


def test_tree_splits_opposite_pulls_and_gives_opposite_weights():
    rng = np.random.default_rng(1)
    n = 200
    group = rng.integers(0, 2, size=n)
    g = np.where(group == 1, 1.0, -1.0) + rng.normal(0, 0.1, n)  # two groups pull opposite ways
    h = np.ones(n)
    feats = np.column_stack([group.astype(float), rng.random(n)])
    cfg = BoostConfig(max_depth=1, min_leaf=20)
    tree = grow_tree(feats, g, h, np.ones(n), np.arange(n), 0, cfg, [[0.5], [0.5]])
    assert tree["feature"] == 0
    w = assign_leaves(tree, feats)
    assert (w[group == 1] < 0).all() and (w[group == 0] > 0).all()
    assert len(tree_leaves(tree)) == 2


def test_no_split_when_pulls_agree():
    rng = np.random.default_rng(2)
    n = 200
    g = 1.0 + rng.normal(0, 0.1, n)
    feats = np.column_stack([rng.integers(0, 2, n).astype(float)])
    tree = grow_tree(feats, g, np.ones(n), np.ones(n), np.arange(n), 0, BoostConfig(), [[0.5]])
    assert "w" in tree and tree["w"] < 0


def test_text_features_and_rule_text():
    f = text_features(["[BUG] Crash\nTraceback: boom\n```x```", "Add dark mode\nplease"])
    assert f[0, :4].tolist() == [1, 1, 1, 0] and f[1, :4].tolist() == [0, 0, 0, 0]
    assert describe_condition(1, 0.5, ">") == "text mentions an error, exception or traceback: yes"
