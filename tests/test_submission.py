import numpy as np
import pandas as pd

from src.data import save_submission


def test_submission_schema(tmp_path) -> None:
    path = tmp_path / "submission.csv"
    save_submission(np.log1p([100_000, 150_000]), pd.DataFrame({"Id": [1, 2]}), path)
    submission = pd.read_csv(path)
    assert submission.columns.tolist() == ["Id", "SalePrice"]
    assert submission.notna().all().all()
    assert np.isfinite(submission["SalePrice"]).all()
