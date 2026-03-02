"""Stub da solução do case: cálculo mensal e rolling 3 meses por escola."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.utils import to_decimal_zero, validate_payments_schema

PAST_DUE_STATUSES = ("overdue", "late")
ROLLING_MONTHS = 3

SCHOOL_ID_COL = "school_id"
DUE_DATE_COL = "due_date"
STATUS_COL = "status"
AMOUNT_COL = "amount"
MONTH_COL = "month"
YEAR_MONTH_COL = "year_month"

TOTAL_DUE_AMOUNT = "total_due_amount_month"
TOTAL_PAST_DUE_AMOUNT = "total_overdue_amount_month"
PAST_DUE_RATIO_MONTH = "overdue_ratio_month"
PAST_DUE_RATIO_ROLLING = "overdue_ratio_rolling_3m"


def safe_ratio(numerator: F.Column, denominator: F.Column) -> F.Column:
    ratio = numerator / denominator
    return F.when(denominator > 0, ratio).otherwise(to_decimal_zero())


def rolling_last_n_months(n: int) -> Window:
    return Window.partitionBy(SCHOOL_ID_COL).orderBy(MONTH_COL).rowsBetween(-(n - 1), 0)


def compute_overdue_rolling_3m(payments_df: DataFrame) -> DataFrame:
    """Computa inadimplência mensal e rolling 3M por escola.

    Args:
        payments_df: DataFrame de entrada com schema `payments`.

    Returns:
        DataFrame por `school_id` e `month` com colunas:
            - school_id
            - month
            - year_month
            - total_due_amount_month
            - total_overdue_amount_month
            - overdue_ratio_month
            - overdue_ratio_rolling_3m
    """
    validate_payments_schema(payments_df)

    amount_col = F.col(AMOUNT_COL)
    is_past_due_payment = F.col(STATUS_COL).isin(*PAST_DUE_STATUSES)
    rolling_window = rolling_last_n_months(ROLLING_MONTHS)

    total_due_amount = F.sum(amount_col).alias(TOTAL_DUE_AMOUNT)
    total_past_due_amount = F.sum(F.when(is_past_due_payment, amount_col)).alias(TOTAL_PAST_DUE_AMOUNT)

    return (
        payments_df.withColumn(MONTH_COL, F.trunc(DUE_DATE_COL, "month"))
        .withColumn(YEAR_MONTH_COL, F.date_format(MONTH_COL, "yyyy-MM"))
        .groupBy(SCHOOL_ID_COL, MONTH_COL, YEAR_MONTH_COL)
        .agg(
            total_due_amount,
            total_past_due_amount,
        )
        .withColumn(
            PAST_DUE_RATIO_MONTH,
            safe_ratio(
                F.col(TOTAL_PAST_DUE_AMOUNT),
                F.col(TOTAL_DUE_AMOUNT),
            ),
        )
        .withColumn(
            PAST_DUE_RATIO_ROLLING,
            safe_ratio(
                F.sum(TOTAL_PAST_DUE_AMOUNT).over(rolling_window),
                F.sum(TOTAL_DUE_AMOUNT).over(rolling_window),
            ),
        )
        .orderBy(SCHOOL_ID_COL, MONTH_COL)
    )
