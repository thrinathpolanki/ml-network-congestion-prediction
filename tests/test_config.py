from src.config import horizon_steps, load_config


def test_horizon_steps_default(cfg):
    assert horizon_steps(cfg) == 6  # 30 min / 5 min


def test_env_override(monkeypatch):
    monkeypatch.setenv("NCP_PREDICTION__HORIZON_MINUTES", "60")
    monkeypatch.setenv("NCP_GENERATOR__N_SITES", "7")
    c = load_config()
    assert c["prediction"]["horizon_minutes"] == 60 and horizon_steps(c) == 12
    assert c["generator"]["n_sites"] == 7
