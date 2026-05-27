from corvin_jarvis.signals import universe_loader


def test_load_returns_all_tickers():
    tickers = universe_loader.load()
    assert len(tickers) >= 20
    symbols = {t.symbol for t in tickers}
    assert "005930" in symbols
    assert "NVDA" in symbols


def test_ticker_has_market_and_sector():
    tickers = universe_loader.load()
    samsung = next(t for t in tickers if t.symbol == "005930")
    assert samsung.market == "KR"
    assert samsung.sector == "semiconductor"


def test_by_sector_groups_symbols():
    tickers = universe_loader.load()
    groups = universe_loader.by_sector(tickers)
    assert "005930" in {t.symbol for t in groups["semiconductor"]}
    assert "000660" in {t.symbol for t in groups["semiconductor"]}
