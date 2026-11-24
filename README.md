# Smart Personalized Flight Recommendation System

An intelligent machine learning application that recommends the most suitable domestic airlines using a **hybrid recommendation engine** combining **Graph-Based Link Prediction**, **K-Means Clustering**, and **Decision Tree Classification** — inspired by the IEEE research paper *"A Personalized Flight Recommender System Based on Link Prediction in Aviation Data"* (IEEE Access, Vol. 12, 2024).

The system takes 9 user travel preferences and returns **ranked, explainable** airline recommendations with score breakdowns, comparison categories, and trade-off analysis.

---

## 📁 Project Architecture & Structure

```
project/
│
├── app.py                 # Flask web backend with /predict, /feedback, /api/routes, /api/evaluation
├── graph_recommender.py   # NEW: Aviation Graph + Link Prediction + K-Means + Hybrid Recommender
├── decision_tree.py       # Scikit-learn compatible Decision Tree engine (retained)
├── train_model.py         # Trains DT model + evaluates Hybrid Recommender
├── generate_data.py       # Generates 10,000 synthetic flight records
├── upload_to_db.py        # MySQL/SQLite database setup for feedback
├── test_app.py            # NEW: Automated test suite (11 tests)
│
├── airline_dataset.csv    # 10,000 flight records (6 cities, 5 airlines)
├── airline_model.pkl      # Trained Decision Tree model
├── label_encoder.pkl      # Categorical encoders
├── city_encoder.pkl       # Unified city encoder
├── evaluation_report.pkl  # NEW: Evaluation metrics report
├── feedback.db            # SQLite feedback storage
├── requirements.txt       # Dependencies
├── README.md              # This file
│
└── templates/
    └── index.html         # Responsive frontend with multi-recommendation UI
```

---

## 🧠 Hybrid Recommendation Engine

The system implements a 4-component hybrid approach inspired by the IEEE paper:

### 1. Aviation Route Graph + Link Prediction (from IEEE paper)
- **Paper approach**: Users as nodes, shared flight routes as edges, multi-hop path similarity
- **Our adaptation**: Airports as nodes, routes as weighted edges, path-based connectivity scoring
- **Formula**: Implements IEEE paper Equation (4) for similarity:
  ```
  sim(vx, vy) = Σ (1/(i-1)) × |path_i(vx,vy)| / Π(n-j)
  ```
- Path length set to 3 (paper's optimal value from Figure 8)

### 2. K-Means Clustering (from IEEE paper)
- **Paper approach**: Cluster users by demographics (age, gender, occupation, education)
- **Our adaptation**: Cluster flights by features (price, duration, stops, rating)
- 9 clusters (paper's optimal from Figure 8)
- Partitions search space for efficient recommendation

### 3. Decision Tree Classification (original project)
- Retained from original project as a component
- 73.05% test accuracy on 5-airline classification
- Predictions boost the hybrid score by 15% when aligned

### 4. Multi-Criteria Preference Matching (new)
- Scores: price match, rating match, duration match, stops match, class availability, seat/time match
- Weighted combination adapted from IEEE paper Equations (5) and (6)
- Transparent score decomposition shown to users

---

## ✈️ 9 Input Features & Target

| # | Feature | Values / Range | Description |
|---|---|---|---|
| 1 | **Source** | Chennai, Delhi, Mumbai, Kolkata, Bangalore, Hyderabad | Departure city |
| 2 | **Destination** | Same 6 cities (Source ≠ Destination) | Arrival city |
| 3 | **Budget** | ₹2,000 – ₹50,000 | Maximum price |
| 4 | **Duration** | 30 – 800 minutes | Expected travel duration |
| 5 | **Stops** | 0 (Direct), 1 Stop, 2 Stops | Preferred stop count |
| 6 | **Rating** | 3.5★ – 5.0★ | Minimum airline rating |
| 7 | **Travel Class** | Economy, Premium Economy, Business | Cabin class |
| 8 | **Seat Type** | Window, Aisle, Middle | Seating preference |
| 9 | **Departure Time** | 6 AM – 10 PM (9 slots) | Preferred flight slot |
| **Output** | **Ranked Airlines** | **5 airlines with scores, explanations, comparisons** | Sorted recommendations |

---

## 🚀 How to Run (Step-by-Step)

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Generate Dataset (10,000 Synthetic Records)
```bash
python generate_data.py
```

### 3. Train Model & Evaluate Hybrid System
```bash
python train_model.py
```
Outputs: Decision Tree accuracy, Hybrid Hit Rate, Graph analysis, Cluster stats.

### 4. (Optional) Initialize Database
```bash
python upload_to_db.py
```

### 5. Launch the Application
```bash
python app.py
```
Open browser: **http://127.0.0.1:5000**

### 6. Run Automated Tests
```bash
python test_app.py
```

---

## 📊 Evaluation Results

### Decision Tree Classifier (Baseline)
| Metric | Value |
|---|---|
| Training Accuracy | 77.96% |
| Testing Accuracy | 73.05% |
| Macro F1-Score | 0.72 |

### Hybrid Recommender (Upgraded)
| Metric | Value |
|---|---|
| Precision@1 | 66.00% |
| Hit Rate@1 | 66.00% |
| **Hit Rate@3** | **92.50%** |
| F1@1 | 79.52% |
| Test Samples | 200 |

### Aviation Graph
| Property | Value |
|---|---|
| Nodes (cities) | 6 |
| Edges (routes) | 15 bidirectional |
| Route connectivity score | 0.90 (all pairs) |

### K-Means Clusters
| Cluster | Flights | Avg Price | Rating | Stops |
|---|---|---|---|---|
| 0 | 1,746 | ₹5,806 | 4.58★ | 0.0 |
| 3 | 2,470 | ₹6,187 | 4.05★ | 1.0 |
| 4 | 371 | ₹23,381 | 4.67★ | 0.3 |
| 7 | 2,903 | ₹5,562 | 4.08★ | 0.0 |

---

## 🔀 Comparison: Original vs Upgraded System

| Aspect | Original | Upgraded |
|---|---|---|
| Algorithm | Decision Tree only | Hybrid (Graph + Clustering + DT) |
| Output | Single airline recommendation | Ranked list of 5 airlines |
| Explainability | Hardcoded airline descriptions | Dynamic score breakdown + trade-offs |
| Graph Analysis | None | Aviation route connectivity scoring |
| Clustering | None | K-Means flight clustering (K=9) |
| Comparison View | None | Side-by-side comparison table |
| Categories | None | Best Overall, Cheapest, Fastest, Highest Rated |
| Score Transparency | Confidence % only | 8-component score decomposition |
| API Endpoints | 3 (/predict, /feedback, /api/feedback) | 5 (+/api/routes, /api/evaluation) |
| Validation | Basic | Comprehensive input validation |
| Testing | None | 11 automated tests |

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Frontend application |
| `/predict` | POST | Get ranked recommendations |
| `/feedback` | POST | Submit user feedback |
| `/api/feedback` | GET | Retrieve feedback log |
| `/api/routes` | GET | Get available cities and routes |
| `/api/evaluation` | GET | Get model evaluation metrics |

### Example `/predict` Request
```json
{
  "source": "Chennai",
  "destination": "Delhi",
  "budget": 5000,
  "duration": 150,
  "stops": 0,
  "rating": 4.0,
  "travel_class": "Economy",
  "seat": "Window",
  "departure_time": "10 AM"
}
```

### Example Response (abbreviated)
```json
{
  "success": true,
  "recommendations": [
    {
      "rank": 1, "airline": "IndiGo", "score": 90.5,
      "explanation": "IndiGo excels on both affordability and quality...",
      "highlights": ["Budget Fit: ...", "Rating: ...", "ML Confirmed: ..."],
      "score_components": {"price_match": 0.85, "rating_match": 0.84, ...}
    },
    {"rank": 2, "airline": "Akasa Air", "score": 77.3, ...},
    ...
  ],
  "meta": {
    "route_connectivity": 0.9,
    "categories": {"best_overall": "IndiGo", "lowest_price": "SpiceJet", ...}
  }
}
```

---

## 📚 IEEE Paper Reference

> Ho Yin Kan, Dennis Wong, Keith Chau. "A Personalized Flight Recommender System Based on Link Prediction in Aviation Data." *IEEE Access*, Volume 12, pp. 36961–36973, 2024. DOI: 10.1109/ACCESS.2024.3369487

### Key adaptations from the paper:
- **Link prediction similarity** (Eq. 4): Implemented for route connectivity scoring
- **K-Means clustering** (Section III-B-2): Applied to flight feature vectors
- **Hybrid scoring** (Eq. 5, 6): Combined graph scores with preference matching
- **Path length = 3, Clusters = 9**: Paper's optimal parameters (Figure 8)

### Limitations vs. paper:
- Paper uses real user travel histories from 131 passengers; our dataset is synthetic
- Paper uses CNN for delay/cancellation prediction; we use preference matching (no delay data)
- Paper uses user demographics (age, gender, occupation); our users are anonymous per-session
- Paper evaluates with 1,273 real flight tickets; we evaluate on synthetic data

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `Model artifacts not found` | Run `python generate_data.py` then `python train_model.py` |
| `Module not found` | Run `pip install -r requirements.txt` |
| Port 5000 in use | Change port in `app.py` or kill existing process |
| MySQL not connecting | App auto-falls back to SQLite — this is normal |
| `OSError` on sklearn | The custom `decision_tree.py` handles this automatically |
| Blank results page | Check browser console; ensure Flask is running |

---

## 🎯 Known Limitations

1. Dataset is synthetic (generated by `generate_data.py`), not real airline data
2. No real-time flight availability, pricing, or booking capability
3. Graph connectivity is uniform across all city pairs (fully connected domestic network)
4. No user travel history tracking across sessions
5. Flight delay/cancellation prediction not implemented (requires weather + operations data)
6. Evaluation metrics are computed on synthetic data
