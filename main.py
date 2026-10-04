"""
California Housing Price Prediction using Linear Regression
Modernized for PEP standards with type hints and structured execution.
"""

from typing import Any
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_linear_model(
    test_size: float = 0.2, random_state: int = 42
) -> tuple[float, float, LinearRegression]:
    """Load California housing dataset, train Linear Regression, and return evaluation metrics."""
    housing: Any = fetch_california_housing(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=test_size, random_state=random_state
    )

    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    y_pred = lin_reg.predict(x_test)
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return mse, r2, lin_reg


def main() -> None:
    mse, r2, _ = train_linear_model()
    print("=" * 50)
    print("🏡 California Housing Regression Results")
    print("=" * 50)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score:                 {r2:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
