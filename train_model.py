"""
train_model.py
Trains a Decision Tree Classifier and evaluates the Hybrid Recommender System.

Updated to include:
  - Decision Tree training (original)
  - Hybrid Recommender evaluation metrics (Precision@K, Hit Rate@K)
  - Graph-based link prediction statistics
  - K-Means cluster analysis
  - Comparison report between baseline DT and hybrid system
"""

import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Try importing sklearn.tree.DecisionTreeClassifier;
# if blocked by Windows Smart App Control / WDAC, use decision_tree.DecisionTreeClassifier
try:
    from sklearn.tree import DecisionTreeClassifier
    _test = DecisionTreeClassifier(max_depth=1)
except (ImportError, OSError):
    from decision_tree import DecisionTreeClassifier


def train_airline_model(csv_path='airline_dataset.csv'):
    print("=" * 60)
    print("SMART FLIGHT RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("=" * 60)

    print(f"\nLoading dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {len(df)} records, {df.shape[1]} columns")

    # 1. City Encoder for Source & Destination
    all_cities = sorted(list(set(df['Source'].unique()).union(set(df['Destination'].unique()))))
    city_encoder = LabelEncoder()
    city_encoder.fit(all_cities)

    df['Source_Enc'] = city_encoder.transform(df['Source'])
    df['Destination_Enc'] = city_encoder.transform(df['Destination'])

    # 2. Encoders for other categorical features and target
    label_encoders = {}

    categorical_cols = ['Class', 'Seat', 'Departure_Time']
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_Enc'] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Target encoder for Airline
    airline_encoder = LabelEncoder()
    df['Airline_Enc'] = airline_encoder.fit_transform(df['Airline'])
    label_encoders['Airline'] = airline_encoder

    # Feature columns used for training (9 features)
    feature_cols = [
        'Source_Enc', 'Destination_Enc', 'Price', 'Duration',
        'Stops', 'Rating', 'Class_Enc', 'Seat_Enc', 'Departure_Time_Enc'
    ]

    X = df[feature_cols]
    y = df['Airline_Enc']

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {feature_cols}")
    print(f"Airlines: {list(airline_encoder.classes_)}")
    print(f"Cities: {all_cities}")

    # 80% Train, 20% Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTraining samples: {len(X_train)} (80%)")
    print(f"Testing samples:  {len(X_test)} (20%)")

    # ==========================================
    # PART 1: Decision Tree Classifier Training
    # ==========================================
    print("\n" + "=" * 60)
    print("PART 1: DECISION TREE CLASSIFIER")
    print("=" * 60)

    dt_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    start_time = time.time()
    print("\nTraining DecisionTreeClassifier...")
    dt_model.fit(X_train, y_train)
    dt_train_time = time.time() - start_time

    # Predictions & Evaluation
    train_pred = dt_model.predict(X_train)
    test_pred = dt_model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\nTraining time: {dt_train_time:.2f}s")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Testing Accuracy:  {test_acc * 100:.2f}%")
    print("\nClassification Report:")
    target_names = list(airline_encoder.classes_)
    print(classification_report(y_test, test_pred, target_names=target_names))

    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(f"{'':>15}", end='')
    for name in target_names:
        print(f"{name:>12}", end='')
    print()
    for i, name in enumerate(target_names):
        print(f"{name:>15}", end='')
        for j in range(len(target_names)):
            print(f"{cm[i][j]:>12}", end='')
        print()

    # Save artifacts
    print("\nSaving trained model and encoders...")
    with open('airline_model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    print("Saved: airline_model.pkl")

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("Saved: label_encoder.pkl")

    with open('city_encoder.pkl', 'wb') as f:
        pickle.dump(city_encoder, f)
    print("Saved: city_encoder.pkl")

    # ==========================================
    # PART 2: Hybrid Recommender Evaluation
    # ==========================================
    print("\n" + "=" * 60)
    print("PART 2: HYBRID RECOMMENDER EVALUATION")
    print("(Graph Link Prediction + K-Means Clustering + Decision Tree)")
    print("=" * 60)

    try:
        from graph_recommender import HybridRecommender

        # Load raw dataset for recommender
        raw_df = pd.read_csv(csv_path)

        start_time = time.time()
        recommender = HybridRecommender(
            df=raw_df,
            model=dt_model,
            label_encoders=label_encoders,
            city_encoder=city_encoder
        )
        hybrid_init_time = time.time() - start_time
        print(f"\nHybrid Recommender initialized in {hybrid_init_time:.2f}s")

        # Graph statistics
        graph = recommender.graph
        print(f"\nAviation Graph:")
        print(f"  Nodes (cities): {len(graph.nodes)}")
        print(f"  Edges (routes): {int((graph.adj_matrix > 0).sum() / 2)}")
        print(f"  Routes tracked: {len(graph.route_stats)}")

        # Similarity matrix
        sim_matrix = recommender.similarity_matrix
        print(f"\nLink Prediction Similarity Matrix ({sim_matrix.shape[0]}x{sim_matrix.shape[1]}):")
        print(f"  Max similarity: {sim_matrix.max():.4f}")
        print(f"  Mean similarity: {sim_matrix.mean():.4f}")
        print(f"  Non-zero entries: {(sim_matrix > 0).sum()}")

        # Print similarity between all city pairs
        print(f"\n  Pairwise Route Connectivity Scores:")
        cities = graph.nodes
        for i in range(len(cities)):
            for j in range(i+1, len(cities)):
                score = sim_matrix[i][j]
                print(f"    {cities[i]:>12} <-> {cities[j]:<12}: {score:.4f}")

        # Cluster statistics
        print(f"\nK-Means Clusters (K={recommender.clusterer.n_clusters}):")
        cluster_stats = recommender.clusterer.get_cluster_stats(raw_df)
        for k, stats in sorted(cluster_stats.items()):
            print(f"  Cluster {k}: {stats['count']} flights, "
                  f"avg ₹{stats['avg_price']:,.0f}, "
                  f"{stats['avg_rating']}★, "
                  f"{stats['avg_stops']} stops")

        # Evaluation
        print(f"\nEvaluating hybrid recommender on test samples...")
        start_time = time.time()
        metrics = recommender.get_evaluation_metrics()
        eval_time = time.time() - start_time

        print(f"\nHybrid Recommender Evaluation Results:")
        print(f"  Test samples evaluated: {metrics.get('total_test_samples', 0)}")
        print(f"  Precision@1: {metrics.get('precision_at_1', 0):.2f}%")
        print(f"  Recall@1:    {metrics.get('recall_at_1', 0):.2f}%")
        print(f"  F1@1:        {metrics.get('f1_at_1', 0):.2f}%")
        print(f"  Hit Rate@1:  {metrics.get('hit_rate_at_1', 0):.2f}%")
        print(f"  Hit Rate@3:  {metrics.get('hit_rate_at_3', 0):.2f}%")
        print(f"  Evaluation time: {eval_time:.2f}s")

        # Comparison summary
        print("\n" + "=" * 60)
        print("COMPARISON: BASELINE vs HYBRID")
        print("=" * 60)
        print(f"  Decision Tree Accuracy:    {test_acc * 100:.2f}%")
        print(f"  Hybrid Hit Rate@1:         {metrics.get('hit_rate_at_1', 0):.2f}%")
        print(f"  Hybrid Hit Rate@3:         {metrics.get('hit_rate_at_3', 0):.2f}%")
        print(f"  Added capability:          Multi-airline ranking, explainability,")
        print(f"                             route graph analysis, cluster-based filtering")

        # Save evaluation report
        report = {
            'dt_train_accuracy': round(train_acc * 100, 2),
            'dt_test_accuracy': round(test_acc * 100, 2),
            'dt_train_time': round(dt_train_time, 2),
            'hybrid_init_time': round(hybrid_init_time, 2),
            'hybrid_metrics': metrics,
            'graph_nodes': len(graph.nodes),
            'graph_edges': int((graph.adj_matrix > 0).sum() / 2),
            'n_clusters': recommender.clusterer.n_clusters,
        }
        with open('evaluation_report.pkl', 'wb') as f:
            pickle.dump(report, f)
        print("\nSaved: evaluation_report.pkl")

    except Exception as e:
        print(f"\nNote: Hybrid evaluation skipped ({e})")
        print("The Decision Tree model was still trained and saved successfully.")

    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    train_airline_model()
