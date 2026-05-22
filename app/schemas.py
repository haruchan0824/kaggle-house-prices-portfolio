from pydantic import BaseModel


class HouseInput(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: float
    GarageArea: float
    TotalBsmtSF: float
    FirstFlrSF: float
    SecondFlrSF: float
    YearBuilt: int
    YearRemodAdd: int
    YrSold: int
    FullBath: int
    HalfBath: int
    BsmtFullBath: float = 0
    BsmtHalfBath: float = 0
    Fireplaces: int = 0