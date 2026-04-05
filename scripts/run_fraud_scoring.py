import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score orders with trained fraud model and write probabilities to Postgres."
    )
    parser.add_argument(
        "--model-path",
        default=str(Path("crispdm-pipeline-model") / "fraud_model.sav"),
        help="Path to the trained sklearn pipeline artifact.",
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("SUPABASE_DB_URL"),
        help="Postgres connection URL. Defaults to SUPABASE_DB_URL.",
    )
    parser.add_argument(
        "--model-version",
        default=os.getenv("FRAUD_MODEL_VERSION", "fraud_pipeline_v1"),
        help="Version tag written to delivery_scores.",
    )
    return parser.parse_args()


def load_supabase_db_url(arg_value: str | None) -> str:
    if arg_value:
        return normalize_sqlalchemy_postgres_url(arg_value)

    env_value = os.getenv("SUPABASE_DB_URL")
    if env_value:
        return normalize_sqlalchemy_postgres_url(env_value)

    raise RuntimeError("SUPABASE_DB_URL is required. Provide --db-url or set the environment variable.")


def normalize_sqlalchemy_postgres_url(raw_url: str) -> str:
    url = raw_url.strip()
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    return url


def load_orders_frame(engine) -> pd.DataFrame:
    orders = pd.read_sql("SELECT * FROM orders", engine)

    items_agg = pd.read_sql(
        """
        SELECT order_id, SUM(quantity) AS num_items
        FROM order_items
        GROUP BY order_id
        """,
        engine,
    )
    orders = orders.merge(items_agg, on="order_id", how="left")
    orders["num_items"] = orders["num_items"].fillna(0)

    customers = pd.read_sql("SELECT customer_id, birthdate FROM customers", engine)
    orders = orders.merge(customers, on="customer_id", how="left")

    orders["order_datetime"] = pd.to_datetime(orders["order_datetime"], errors="coerce", utc=True)
    orders["birthdate"] = pd.to_datetime(orders["birthdate"], errors="coerce", utc=True)

    # Keep feature naming aligned with the notebook training pipeline.
    orders["total_value"] = orders["order_total"]
    orders["customer_age"] = ((orders["order_datetime"] - orders["birthdate"]).dt.days // 365).astype("Int64")

    orders = orders.sort_values(["customer_id", "order_datetime", "order_id"])
    orders["customer_order_count"] = orders.groupby("customer_id").cumcount()

    orders["order_value_per_item"] = orders["total_value"] / (orders["num_items"] + 1)
    orders["order_datetime_year"] = orders["order_datetime"].dt.year
    orders["order_datetime_month"] = orders["order_datetime"].dt.month
    orders["order_datetime_hour"] = orders["order_datetime"].dt.hour
    orders["days_between"] = (orders["order_datetime"] - orders["birthdate"]).dt.days

    return orders


def extract_required_features(model) -> list[str]:
    if hasattr(model, "named_steps") and "pre" in model.named_steps:
        pre = model.named_steps["pre"]
        if hasattr(pre, "transformers_"):
            for transformer in pre.transformers_:
                if len(transformer) >= 3:
                    cols = transformer[2]
                    if isinstance(cols, (list, tuple)):
                        return [str(c) for c in cols]
    if hasattr(model, "feature_names_in_"):
        return [str(c) for c in model.feature_names_in_]
    raise RuntimeError("Unable to determine required features from model artifact.")


def score_orders(df: pd.DataFrame, model) -> pd.DataFrame:
    required_features = extract_required_features(model)

    for feature in required_features:
        if feature not in df.columns:
            df[feature] = np.nan

    x = df[required_features]
    probabilities = model.predict_proba(x)[:, 1]
    df["fraud_probability"] = np.clip(probabilities, 0.0, 1.0)
    return df


def persist_scores(engine, scored_df: pd.DataFrame, model_version: str) -> int:
    now_utc = datetime.now(timezone.utc)

    records = [
        {
            "order_id": int(row.order_id),
            "risk_score": round(float(row.fraud_probability), 4),
            "late_delivery_probability": round(float(row.fraud_probability), 4),
            "scored_at": now_utc,
            "score_source": "pipeline_predict_proba",
            "model_version": model_version,
        }
        for row in scored_df.itertuples(index=False)
    ]

    if not records:
        return 0

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE orders
                SET risk_score = :risk_score
                WHERE order_id = :order_id
                """
            ),
            records,
        )

        conn.execute(
            text(
                """
                INSERT INTO delivery_scores (
                    order_id,
                    late_delivery_probability,
                    scored_at,
                    score_source,
                    model_version
                ) VALUES (
                    :order_id,
                    :late_delivery_probability,
                    :scored_at,
                    :score_source,
                    :model_version
                )
                ON CONFLICT (order_id) DO UPDATE SET
                    late_delivery_probability = EXCLUDED.late_delivery_probability,
                    scored_at = EXCLUDED.scored_at,
                    score_source = EXCLUDED.score_source,
                    model_version = EXCLUDED.model_version
                """
            ),
            records,
        )

    return len(records)


def main() -> None:
    args = parse_args()
    db_url = load_supabase_db_url(args.db_url)
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    engine = create_engine(db_url)

    orders_df = load_orders_frame(engine)
    scored_df = score_orders(orders_df, model)
    scored_count = persist_scores(engine, scored_df, args.model_version)

    print(json.dumps({"scored": scored_count, "model_path": str(model_path)}))


if __name__ == "__main__":
    main()
