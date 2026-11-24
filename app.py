"""
app.py
Flask web server for the Smart Personalized Flight Recommendation System.

Upgraded Features (informed by IEEE Access paper on Link Prediction in Aviation Data):
  - /predict API: returns MULTIPLE ranked airline recommendations with hybrid scoring
  - Graph-based route connectivity scores (link prediction inspired)
  - K-Means cluster-based flight filtering
  - Decision Tree classification (retained from original project)
  - Explainable recommendations with score breakdowns
  - Comparison categories (best overall, cheapest, shortest, highest rated)
  - /feedback API: stores user feedback in MySQL (with SQLite fallback)
  - /api/feedback: fetches recent feedback
  - /api/routes: returns available routes and cities
  - /api/evaluation: returns model evaluation metrics
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from decision_tree import DecisionTreeClassifier
from graph_recommender import HybridRecommender

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Paths to artifacts
MODEL_PATH = 'airline_model.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
CITY_ENCODER_PATH = 'city_encoder.pkl'
DATASET_PATH = 'airline_dataset.csv'

# Global state
model = None
label_encoders = None
city_encoder = None
recommender = None
dataset_df = None

# MySQL Database configuration
DB_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', 'localhost'),
    'user': os.environ.get('MYSQL_USER', 'root'),
    'password': os.environ.get('MYSQL_PASSWORD', ''),
    'port': int(os.environ.get('MYSQL_PORT', 3306)),
    'database': 'airline_db'
}

# Valid cities and options (from dataset)
VALID_CITIES = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
VALID_CLASSES = ['Economy', 'Premium Economy', 'Business']
VALID_SEATS = ['Window', 'Aisle', 'Middle']
VALID_DEPARTURE_TIMES = ['6 AM', '8 AM', '10 AM', '12 PM', '2 PM', '4 PM', '6 PM', '8 PM', '10 PM']


def load_artifacts():
    """Load trained model, encoders, dataset, and initialize the hybrid recommender."""
    global model, label_encoders, city_encoder, recommender, dataset_df

    # Load dataset
    if os.path.exists(DATASET_PATH):
        dataset_df = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset loaded: {len(dataset_df)} records")
    else:
        logger.warning(f"Dataset '{DATASET_PATH}' not found. Run generate_data.py first.")
        return

    # Load model artifacts
    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(CITY_ENCODER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoders = pickle.load(f)
        with open(CITY_ENCODER_PATH, 'rb') as f:
            city_encoder = pickle.load(f)
        logger.info("Model and Encoders loaded successfully.")
    else:
        logger.warning("Model artifacts not found. Run train_model.py first.")
        logger.info("Proceeding with graph-based recommender only (no Decision Tree).")

    # Initialize hybrid recommender
    try:
        recommender = HybridRecommender(
            df=dataset_df,
            model=model,
            label_encoders=label_encoders,
            city_encoder=city_encoder
        )
        logger.info("Hybrid Recommender initialized (Graph + Clustering + Decision Tree)")
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")


# Load on startup
load_artifacts()


def save_feedback_to_db(source, dest, budget, travel_class, airline, rating, comment):
    """Saves feedback to MySQL if accessible, else falls back to SQLite."""
    # 1. Try MySQL
    try:
        import mysql.connector
        cnx = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database']
        )
        cursor = cnx.cursor()
        query = """
        INSERT INTO feedback (source, destination, budget, travel_class, recommended_airline, rating, comment)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(query, (source, dest, budget, travel_class, airline, rating, comment))
        cnx.commit()
        cursor.close()
        cnx.close()
        return True, "Feedback successfully saved to MySQL database (airline_db.feedback)!"
    except Exception as mysql_err:
        # 2. Fallback to SQLite
        try:
            import sqlite3
            conn = sqlite3.connect('feedback.db')
            c = conn.cursor()
            c.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                destination TEXT NOT NULL,
                budget INTEGER NOT NULL,
                travel_class TEXT NOT NULL,
                recommended_airline TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            c.execute("""
            INSERT INTO feedback (source, destination, budget, travel_class, recommended_airline, rating, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (source, dest, budget, travel_class, airline, rating, comment))
            conn.commit()
            conn.close()
            return True, f"Feedback saved locally to SQLite (MySQL not currently accessible)."
        except Exception as sqlite_err:
            return False, f"Failed to save feedback: {str(sqlite_err)}"


def validate_input(data):
    """Validate user input and return cleaned data or error message."""
    errors = []

    source = data.get('source', '').strip()
    destination = data.get('destination', '').strip()

    if not source or source not in VALID_CITIES:
        errors.append(f"Invalid source city. Must be one of: {', '.join(VALID_CITIES)}")
    if not destination or destination not in VALID_CITIES:
        errors.append(f"Invalid destination city. Must be one of: {', '.join(VALID_CITIES)}")
    if source and destination and source == destination:
        errors.append("Source and Destination cannot be the same city.")

    try:
        budget = float(data.get('budget', 0))
        if budget < 2000 or budget > 50000:
            errors.append("Budget must be between ₹2,000 and ₹50,000.")
    except (ValueError, TypeError):
        errors.append("Budget must be a valid number.")
        budget = 5000

    try:
        duration = float(data.get('duration', 0))
        if duration < 30 or duration > 800:
            errors.append("Duration must be between 30 and 800 minutes.")
    except (ValueError, TypeError):
        errors.append("Duration must be a valid number.")
        duration = 150

    try:
        stops = int(data.get('stops', 0))
        if stops < 0 or stops > 3:
            errors.append("Stops must be 0, 1, or 2.")
    except (ValueError, TypeError):
        errors.append("Stops must be a valid integer.")
        stops = 0

    try:
        rating = float(data.get('rating', 3.5))
        if rating < 1.0 or rating > 5.0:
            errors.append("Rating must be between 1.0 and 5.0.")
    except (ValueError, TypeError):
        errors.append("Rating must be a valid number.")
        rating = 4.0

    travel_class = data.get('travel_class', 'Economy').strip()
    if travel_class not in VALID_CLASSES:
        errors.append(f"Invalid travel class. Must be one of: {', '.join(VALID_CLASSES)}")

    seat = data.get('seat', 'Window').strip()
    if seat not in VALID_SEATS:
        errors.append(f"Invalid seat. Must be one of: {', '.join(VALID_SEATS)}")

    departure_time = data.get('departure_time', '10 AM').strip()
    if departure_time not in VALID_DEPARTURE_TIMES:
        errors.append(f"Invalid departure time. Must be one of: {', '.join(VALID_DEPARTURE_TIMES)}")

    return {
        'source': source,
        'destination': destination,
        'budget': budget,
        'duration': duration,
        'stops': stops,
        'rating': rating,
        'travel_class': travel_class,
        'seat': seat,
        'departure_time': departure_time,
    }, errors


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main recommendation endpoint.
    Returns multiple ranked airline recommendations with explanations.
    """
    global recommender

    if recommender is None:
        load_artifacts()
        if recommender is None:
            return jsonify({
                'success': False,
                'error': 'Recommendation system not initialized. Run generate_data.py and train_model.py first.'
            }), 500

    try:
        data = request.get_json(force=True)
        cleaned, errors = validate_input(data)

        if errors:
            return jsonify({
                'success': False,
                'error': '; '.join(errors)
            }), 400

        # Get recommendations from hybrid system
        recommendations, meta = recommender.recommend(
            source=cleaned['source'],
            destination=cleaned['destination'],
            budget=cleaned['budget'],
            duration=cleaned['duration'],
            stops=cleaned['stops'],
            rating=cleaned['rating'],
            travel_class=cleaned['travel_class'],
            seat=cleaned['seat'],
            departure_time=cleaned['departure_time'],
            top_k=5
        )

        if not recommendations:
            return jsonify({
                'success': True,
                'recommendations': [],
                'meta': meta,
                'message': f"No flights found for {cleaned['source']} → {cleaned['destination']}. Try a different route."
            })

        # Format response
        formatted_recs = []
        for i, rec in enumerate(recommendations):
            formatted_recs.append({
                'rank': i + 1,
                'airline': rec['airline'],
                'score': round(rec['total_score'] * 100, 1),
                'score_components': rec['score_components'],
                'airline_stats': rec['airline_stats'],
                'best_flight': rec.get('best_flight'),
                'explanation': rec.get('explanation', {}).get('text', ''),
                'highlights': rec.get('explanation', {}).get('highlights', []),
                'trade_offs': rec.get('explanation', {}).get('trade_offs', []),
                'dt_boost': rec.get('dt_boost', 0),
            })

        # Also include legacy single-recommendation format for backward compatibility
        top_rec = formatted_recs[0]

        return jsonify({
            'success': True,
            # New multi-recommendation format
            'recommendations': formatted_recs,
            'meta': {
                'route_connectivity': meta.get('route_connectivity', 0),
                'cluster_id': meta.get('cluster_id', 0),
                'total_route_flights': meta.get('total_route_flights', 0),
                'dt_prediction': meta.get('dt_prediction'),
                'dt_confidence': meta.get('dt_confidence'),
                'categories': meta.get('categories', {}),
            },
            # Legacy single-recommendation format (backward compatible)
            'recommended_airline': top_rec['airline'],
            'confidence': top_rec['score'],
            'explanation': top_rec['explanation'],
            'highlights': top_rec['highlights'],
            'inputs': cleaned,
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/routes', methods=['GET'])
def get_routes():
    """Return available cities and route information."""
    if dataset_df is None:
        return jsonify({'cities': VALID_CITIES, 'routes': []})

    routes = []
    for (src, dst), group in dataset_df.groupby(['Source', 'Destination']):
        routes.append({
            'source': src,
            'destination': dst,
            'flight_count': len(group),
            'avg_price': int(group['Price'].mean()),
            'avg_duration': int(group['Duration'].mean()),
            'airlines': list(group['Airline'].unique()),
        })

    return jsonify({
        'cities': VALID_CITIES,
        'classes': VALID_CLASSES,
        'seats': VALID_SEATS,
        'departure_times': VALID_DEPARTURE_TIMES,
        'routes': routes,
    })


@app.route('/api/evaluation', methods=['GET'])
def get_evaluation():
    """Return model evaluation metrics."""
    if recommender is None:
        return jsonify({'error': 'Recommender not initialized'}), 500

    try:
        metrics = recommender.get_evaluation_metrics()
        return jsonify({
            'success': True,
            'metrics': metrics,
            'model_info': {
                'type': 'Hybrid (Graph + Clustering + Decision Tree)',
                'graph_nodes': len(recommender.graph.nodes),
                'graph_edges': int((recommender.graph.adj_matrix > 0).sum() / 2),
                'n_clusters': recommender.clusterer.n_clusters,
                'dataset_size': len(recommender.df),
                'dt_available': recommender.model is not None,
            }
        })
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json(force=True)
        source = data.get('source', 'Unknown')
        dest = data.get('destination', 'Unknown')
        budget = int(data.get('budget', 0))
        travel_class = data.get('travel_class', 'Economy')
        airline = data.get('recommended_airline', 'Unknown')
        rating = int(data.get('rating', 5))
        comment = data.get('comment', '').strip()

        success, msg = save_feedback_to_db(source, dest, budget, travel_class, airline, rating, comment)
        return jsonify({
            'success': success,
            'message': msg
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/feedback', methods=['GET'])
def get_feedback():
    """Retrieve feedback records."""
    records = []
    # Try MySQL first
    try:
        import mysql.connector
        cnx = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database']
        )
        cursor = cnx.cursor(dictionary=True)
        cursor.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 5;")
        records = cursor.fetchall()
        cursor.close()
        cnx.close()
        return jsonify({'source': 'MySQL', 'data': records})
    except Exception:
        # Fallback to SQLite
        try:
            import sqlite3
            if os.path.exists('feedback.db'):
                conn = sqlite3.connect('feedback.db')
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 5;")
                rows = [dict(r) for r in c.fetchall()]
                conn.close()
                return jsonify({'source': 'SQLite (Local)', 'data': rows})
        except Exception:
            pass
    return jsonify({'source': 'None', 'data': []})


if __name__ == '__main__':
    print("=" * 60)
    print("Smart Personalized Flight Recommendation System")
    print("Powered by: Graph-Based Link Prediction + Decision Tree")
    print("=" * 60)
    print("Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=True)
