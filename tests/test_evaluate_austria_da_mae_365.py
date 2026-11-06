from datetime import datetime

import pandas as pd
from openpyxl import Workbook

from scripts.decision_aware.evaluate_austria_da_mae_365 import (
    APCS_IMBALANCE,
    APCS_SHEET,
    APCS_UTC_END,
    load_apcs_imbalance,
)
from scripts.decision_aware.evaluate_austria_settlement_365 import (
    load_energy_charts_price,
)


def test_load_apcs_imbalance_converts_quarter_end_to_start(tmp_path):
    path = tmp_path / "apcs.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = APCS_SHEET
    sheet.cell(1, 3, APCS_UTC_END)
    sheet.cell(1, 26, APCS_IMBALANCE)
    sheet.cell(5, 3, datetime(2026, 9, 1, 0, 15))
    sheet.cell(5, 26, 100.125)
    sheet.cell(6, 3, datetime(2026, 9, 1, 0, 30))
    sheet.cell(6, 26, -5.5)
    workbook.save(path)

    result = load_apcs_imbalance(path)

    assert result.index.equals(
        pd.DatetimeIndex(
            ["2026-09-01 00:00:00+00:00", "2026-09-01 00:15:00+00:00"]
        )
    )
    assert result.tolist() == [100.125, -5.5]


def test_load_energy_charts_price(tmp_path):
    path = tmp_path / "price.json"
    path.write_text(
        '{"license_info":"test","unit":"EUR / MWh",'
        '"unix_seconds":[1790812800,1790813700],"price":[10.0,20.0]}',
        encoding="utf-8",
    )

    result, metadata = load_energy_charts_price(path)

    assert len(result) == 2
    assert result.iloc[0] == 10.0
    assert result.index[1] - result.index[0] == pd.Timedelta(minutes=15)
    assert metadata["unit"] == "EUR / MWh"
