# pip install mlflow pyspark datasets

import mlflow
import mlflow.spark
from datasets import load_dataset
from mlflow import MlflowClient
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LinearSVC,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IDF, HashingTF, Tokenizer
from pyspark.sql import SparkSession

# ── SETUP ─────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://mlflow:5001")
mlflow.set_experiment("Timothy")

spark = SparkSession.builder.appName("HW8-IMDB").getOrCreate()

# ── DATASET ───────────────────────────────────────────────────────────────────
# Load IMDB from Hugging Face.
dataset = load_dataset("stanfordnlp/imdb", split="train")

# Select exactly 4500 reviews.
dataset = dataset.shuffle(seed=42).select(range(4500))

rows = [(row["text"], float(row["label"])) for row in dataset]

df = spark.createDataFrame(rows, ["text", "label"])

# 80% training / 20% testing
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"Total reviews: {df.count()}")
print(f"Training reviews: {train_df.count()}")
print(f"Testing reviews: {test_df.count()}")

# ── CANDIDATE MODELS ──────────────────────────────────────────────────────────
# Each pipeline = TF-IDF processing + classifier.
# TF-IDF converts raw text into numerical feature vectors.
# MLflow records every combination so we can compare them in the UI.

tokenizer = Tokenizer(
    inputCol="text",
    outputCol="words",
)

hashing_tf = HashingTF(
    inputCol="words",
    outputCol="rawFeatures",
    numFeatures=5000,
)

idf = IDF(
    inputCol="rawFeatures",
    outputCol="features",
)

candidates = [
    {
        "run_name": "tfidf-logistic",
        "pipeline": Pipeline(
            stages=[
                tokenizer,
                hashing_tf,
                idf,
                LogisticRegression(
                    featuresCol="features",
                    labelCol="label",
                ),
            ]
        ),
        "params": {
            "vectorizer": "tfidf",
            "model": "LogisticRegression",
            "num_features": 5000,
        },
    },
    {
        "run_name": "tfidf-linearsvc",
        "pipeline": Pipeline(
            stages=[
                tokenizer,
                hashing_tf,
                idf,
                LinearSVC(
                    featuresCol="features",
                    labelCol="label",
                ),
            ]
        ),
        "params": {
            "vectorizer": "tfidf",
            "model": "LinearSVC",
            "num_features": 5000,
        },
    },
    {
        "run_name": "tfidf-naivebayes",
        "pipeline": Pipeline(
            stages=[
                tokenizer,
                hashing_tf,
                idf,
                NaiveBayes(
                    featuresCol="features",
                    labelCol="label",
                ),
            ]
        ),
        "params": {
            "vectorizer": "tfidf",
            "model": "NaiveBayes",
            "num_features": 5000,
        },
    },
    {
        "run_name": "tfidf-randomforest",
        "pipeline": Pipeline(
            stages=[
                tokenizer,
                hashing_tf,
                idf,
                RandomForestClassifier(
                    featuresCol="features",
                    labelCol="label",
                    numTrees=100,
                    seed=42,
                ),
            ]
        ),
        "params": {
            "vectorizer": "tfidf",
            "model": "RandomForest",
            "num_features": 5000,
            "num_trees": 100,
        },
    },
]

# ── RUN ALL CANDIDATES ────────────────────────────────────────────────────────
run_results = []

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy",
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1",
)

for candidate in candidates:
    with mlflow.start_run(run_name=candidate["run_name"]) as run:
        model = candidate["pipeline"].fit(train_df)

        predictions = model.transform(test_df)

        metrics = {
            "accuracy": accuracy_evaluator.evaluate(predictions),
            "f1_score": f1_evaluator.evaluate(predictions),
        }

        mlflow.log_params(candidate["params"])
        mlflow.log_metrics(metrics)

        # Log the full Spark PipelineModel.
        # This keeps preprocessing + classifier together.
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="model",
            registered_model_name="imdb-sentiment",
        )

        run_results.append(
            {
                "run_id": run.info.run_id,
                "run_name": candidate["run_name"],
                **metrics,
            }
        )

        print(
            f"{candidate['run_name']:25s} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"f1={metrics['f1_score']:.4f}"
        )

# ── FIND BEST & ASSIGN ALIAS ──────────────────────────────────────────────────
best = max(run_results, key=lambda x: x["f1_score"])

print(f"\nBest model: {best['run_name']} (F1={best['f1_score']:.4f})")

client = MlflowClient()

versions = client.search_model_versions("name='imdb-sentiment'")

best_version = next(
    version.version for version in versions if version.run_id == best["run_id"]
)

client.set_registered_model_alias(
    "imdb-sentiment",
    "champion",
    version=best_version,
)

print(f"Alias 'champion' -> version {best_version} ({best['run_name']})")

spark.stop()
