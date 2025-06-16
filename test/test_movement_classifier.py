from movement_classifier import MovementClassifier

def test_default_classify_neutral():
    mc = MovementClassifier(config_path="nonexistent.json")
    out = mc.classify({})
    assert out["movement_type"] == "NEUTRAL"
    assert out["expected_move_pct"] == 0.0

def test_rsi_overbought_and_oversold():
    mc = MovementClassifier(config_path="nonexistent.json")
    # RSI > overbought threshold
    out1 = mc.classify({"rsi": 100})
    assert out1["movement_type"] == "OVERBOUGHT"
    # RSI < oversold threshold
    out2 = mc.classify({"rsi": 0})
    assert out2["movement_type"] == "OVERSOLD"
