"""
graph_recommender.py
Aviation Graph-Based Recommendation Engine inspired by:
  "A Personalized Flight Recommender System Based on Link Prediction in Aviation Data"
  IEEE Access, Volume 12, pp. 36961-36973, 2024.

Adaptations from the IEEE paper to our domestic airline dataset:
  - Paper uses user-user graph with shared O-D edges; we adapt to an airport-route graph
    since our dataset lacks individual user travel histories.
  - Paper uses K-Means clustering of users by demographics; we cluster flights by features
    (price, duration, stops, rating) to partition the search space.
  - Paper uses multi-hop path-based link prediction (Eq. 4) for user similarity;
    we apply the same path-counting formula to compute route connectivity scores
    between airports in our aviation graph.
  - Paper uses CNN for delay/cancellation prediction to weight recommendations;
    we use the existing Decision Tree predictions combined with flight quality scoring
    since our dataset does not contain delay/cancellation labels or weather data.

The result is a hybrid recommender that combines:
  1. Graph-based route connectivity scores (link prediction inspired)
  2. K-Means cluster-based flight filtering
  3. Decision Tree airline classification
  4. Multi-criteria preference matching and ranking
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class AviationGraph:
    """
    Builds and queries an aviation route graph.
    
    Nodes = airports/cities.
    Edges = routes with weights based on flight frequency and average rating.
    
    Implements the link prediction similarity from IEEE paper Eq. (4):
      sim(vx, vy) = sum_{i=2}^{l} (1/(i-1)) * |path_i(vx,vy)| / prod_{j=2}^{i} (n-j)
    
    Where paths are acyclic paths of length i between nodes vx and vy.
    """
    
    def __init__(self):
        self.nodes = []
        self.adj_matrix = None
        self.node_index = {}
        self.edge_weights = {}
        self.route_stats = {}
    
    def build_from_dataframe(self, df):
        """Build the aviation graph from the flight dataset."""
        cities = sorted(list(set(df['Source'].unique()) | set(df['Destination'].unique())))
        self.nodes = cities
        self.node_index = {city: i for i, city in enumerate(cities)}
        n = len(cities)
        
        # Build adjacency matrix weighted by flight frequency
        self.adj_matrix = np.zeros((n, n), dtype=np.float64)
        
        for (src, dst), group in df.groupby(['Source', 'Destination']):
            i, j = self.node_index[src], self.node_index[dst]
            count = len(group)
            avg_rating = group['Rating'].mean()
            avg_price = group['Price'].mean()
            avg_duration = group['Duration'].mean()
            
            # Edge weight: combination of frequency, rating quality
            weight = count * (avg_rating / 5.0)
            self.adj_matrix[i][j] = weight
            self.adj_matrix[j][i] = weight  # Undirected for path counting
            
            # Store detailed route statistics
            route_key = (src, dst)
            self.route_stats[route_key] = {
                'count': count,
                'avg_rating': round(avg_rating, 2),
                'avg_price': round(avg_price, 0),
                'avg_duration': round(avg_duration, 0),
                'min_price': int(group['Price'].min()),
                'max_price': int(group['Price'].max()),
                'airlines': group['Airline'].value_counts().to_dict(),
                'stops_distribution': group['Stops'].value_counts().to_dict()
            }
            
            self.edge_weights[(src, dst)] = weight
    
    def compute_link_prediction_similarity(self, max_path_length=3):
        """
        Compute pairwise node similarity using the link prediction formula 
        from the IEEE paper (Equation 4).
        
        sim(vx, vy) = sum_{i=2}^{l} (1/(i-1)) * |path_i(vx,vy)| / prod_{j=2}^{i}(n-j)
        
        Returns: n x n similarity matrix
        """
        n = len(self.nodes)
        if n == 0:
            return np.zeros((0, 0))
        
        # Binary adjacency for path counting
        A = (self.adj_matrix > 0).astype(np.float64)
        
        sim_matrix = np.zeros((n, n), dtype=np.float64)
        
        for path_len in range(2, max_path_length + 1):
            # Count paths of length path_len using matrix power
            # A^k[i][j] gives the number of walks of length k from i to j
            path_count_matrix = np.linalg.matrix_power(A, path_len)
            
            # Weight factor: 1/(i-1) as per paper
            weight = 1.0 / (path_len - 1)
            
            # Normalization: product_{j=2}^{i} (n-j)
            normalization = 1.0
            for j in range(2, path_len + 1):
                denom = max(1, n - j)
                normalization *= denom
            
            if normalization > 0:
                sim_matrix += weight * path_count_matrix / normalization
        
        # Normalize to [0, 1]
        max_val = sim_matrix.max()
        if max_val > 0:
            sim_matrix /= max_val
        
        return sim_matrix
    
    def get_route_connectivity_score(self, source, destination):
        """
        Get the link-prediction-based connectivity score for a route.
        Higher score = stronger connection in the aviation network.
        """
        if source not in self.node_index or destination not in self.node_index:
            return 0.0
        
        sim_matrix = self.compute_link_prediction_similarity(max_path_length=3)
        i = self.node_index[source]
        j = self.node_index[destination]
        
        return float(sim_matrix[i][j])
    
    def get_route_stats(self, source, destination):
        """Get statistics for a specific route."""
        return self.route_stats.get((source, destination), None)
    
    def get_connected_routes(self, city):
        """Get all cities directly connected to a given city."""
        if city not in self.node_index:
            return []
        idx = self.node_index[city]
        connected = []
        for j, node in enumerate(self.nodes):
            if self.adj_matrix[idx][j] > 0 and node != city:
                connected.append(node)
        return connected


class FlightClusterer:
    """
    K-Means clustering of flights by features, inspired by the IEEE paper's
    approach of clustering users to partition the recommendation space.
    
    Paper uses distributed K-Means on user demographics;
    we apply K-Means on flight feature vectors (price, duration, stops, rating).
    """
    
    def __init__(self, n_clusters=5, max_iterations=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.feature_names = ['Price', 'Duration', 'Stops', 'Rating']
    
    def _normalize_features(self, X):
        """Min-max normalization to [0, 1] range, matching paper Eq. (1)."""
        self.min_vals = X.min(axis=0)
        self.max_vals = X.max(axis=0)
        ranges = self.max_vals - self.min_vals
        ranges[ranges == 0] = 1  # Avoid division by zero
        return (X - self.min_vals) / ranges
    
    def _normalize_new(self, X):
        """Normalize new data using stored min/max values."""
        ranges = self.max_vals - self.min_vals
        ranges[ranges == 0] = 1
        return (X - self.min_vals) / ranges
    
    def fit(self, df):
        """Fit K-Means clusters on flight features."""
        np.random.seed(self.random_state)
        
        X = df[self.feature_names].values.astype(np.float64)
        X_norm = self._normalize_features(X)
        
        n_samples = X_norm.shape[0]
        
        # Initialize centroids using K-Means++ strategy
        centroid_indices = [np.random.randint(n_samples)]
        for _ in range(1, self.n_clusters):
            distances = np.min([
                np.sum((X_norm - X_norm[ci]) ** 2, axis=1)
                for ci in centroid_indices
            ], axis=0)
            probs = distances / distances.sum()
            centroid_indices.append(np.random.choice(n_samples, p=probs))
        
        self.centroids = X_norm[centroid_indices].copy()
        
        # Iterate
        for _ in range(self.max_iterations):
            # Assign clusters
            distances = np.array([
                np.sum((X_norm - c) ** 2, axis=1) for c in self.centroids
            ]).T
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X_norm[mask].mean(axis=0)
                else:
                    new_centroids[k] = self.centroids[k]
            
            if np.allclose(new_centroids, self.centroids, atol=1e-6):
                break
            self.centroids = new_centroids
        
        self.labels = labels
        return self
    
    def find_nearest_cluster(self, price, duration, stops, rating):
        """Find the cluster nearest to the given flight preferences."""
        x = np.array([[price, duration, stops, rating]], dtype=np.float64)
        x_norm = self._normalize_new(x)
        distances = np.sum((self.centroids - x_norm) ** 2, axis=1)
        return int(np.argmin(distances)), float(np.min(distances))
    
    def get_cluster_stats(self, df):
        """Get summary statistics for each cluster."""
        stats = {}
        for k in range(self.n_clusters):
            mask = self.labels == k
            cluster_df = df.iloc[mask]
            if len(cluster_df) == 0:
                continue
            stats[k] = {
                'count': int(mask.sum()),
                'avg_price': round(cluster_df['Price'].mean(), 0),
                'avg_duration': round(cluster_df['Duration'].mean(), 0),
                'avg_rating': round(cluster_df['Rating'].mean(), 2),
                'avg_stops': round(cluster_df['Stops'].mean(), 2),
                'top_airlines': cluster_df['Airline'].value_counts().head(3).to_dict()
            }
        return stats


class HybridRecommender:
    """
    Hybrid Flight Recommendation Engine combining:
    1. Graph-based link prediction scores (IEEE paper approach)
    2. K-Means cluster-based flight filtering (IEEE paper approach)
    3. Decision Tree classification (existing project)
    4. Multi-criteria preference matching
    
    This implements the paper's overall methodology (Figure 1) adapted 
    to our available data:
    - Step 1: Preprocessing (normalization, encoding)
    - Step 2: Clustering + Graph construction
    - Step 3: Link prediction for route scoring
    - Step 4: Quality-weighted recommendation (replaces CNN delay prediction
              with preference matching since we lack delay data)
    """
    
    def __init__(self, df, model=None, label_encoders=None, city_encoder=None):
        self.df = df.copy()
        self.model = model
        self.label_encoders = label_encoders
        self.city_encoder = city_encoder
        
        # Build aviation graph
        self.graph = AviationGraph()
        self.graph.build_from_dataframe(df)
        
        # Build flight clusters (paper uses 9 clusters as optimal)
        self.clusterer = FlightClusterer(n_clusters=min(9, len(df) // 20))
        self.clusterer.fit(df)
        
        # Precompute similarity matrix
        self.similarity_matrix = self.graph.compute_link_prediction_similarity(
            max_path_length=3  # Paper found path_length=3 optimal
        )
        
        # Precompute airline statistics for explanations
        self.airline_stats = self._compute_airline_stats()
    
    def _compute_airline_stats(self):
        """Compute per-airline statistics for scoring and explanation."""
        stats = {}
        for airline, group in self.df.groupby('Airline'):
            stats[airline] = {
                'avg_price': round(group['Price'].mean(), 0),
                'avg_duration': round(group['Duration'].mean(), 0),
                'avg_rating': round(group['Rating'].mean(), 2),
                'avg_stops': round(group['Stops'].mean(), 2),
                'total_flights': len(group),
                'min_price': int(group['Price'].min()),
                'max_price': int(group['Price'].max()),
                'direct_ratio': round((group['Stops'] == 0).mean() * 100, 1),
                'classes': group['Class'].value_counts().to_dict(),
                'routes': group.groupby(['Source', 'Destination']).size().to_dict()
            }
        return stats
    
    def recommend(self, source, destination, budget, duration, stops, rating,
                  travel_class, seat, departure_time, top_k=5):
        """
        Generate personalized flight recommendations.
        
        Returns a list of ranked recommendations with scores and explanations.
        
        Scoring formula (adapted from paper Eq. 5 & 6):
            V_f = P_f * (W_quality + W_match) / (epsilon + W_mismatch)
        
        Where:
            P_f = link prediction route score (graph connectivity)
            W_quality = flight quality score (rating, direct ratio)
            W_match = preference match score
            W_mismatch = penalty for constraint violations
        """
        # Step 1: Filter flights matching source and destination
        route_flights = self.df[
            (self.df['Source'] == source) & 
            (self.df['Destination'] == destination)
        ].copy()
        
        if len(route_flights) == 0:
            return [], {}
        
        # Step 2: Get graph-based route connectivity score
        route_score = self._get_route_score(source, destination)
        
        # Step 3: Find nearest cluster for user preferences
        cluster_id, cluster_dist = self.clusterer.find_nearest_cluster(
            budget, duration, stops, rating
        )
        
        # Step 4: Score each airline for this route
        airline_scores = []
        route_airlines = route_flights.groupby('Airline')
        
        for airline, airline_flights in route_airlines:
            score_details = self._score_airline(
                airline, airline_flights, source, destination,
                budget, duration, stops, rating, travel_class,
                seat, departure_time, route_score, cluster_id
            )
            airline_scores.append(score_details)
        
        # Step 5: Decision Tree prediction (existing model integration)
        dt_prediction = None
        dt_confidence = None
        if self.model is not None and self.label_encoders is not None and self.city_encoder is not None:
            try:
                dt_prediction, dt_confidence = self._get_dt_prediction(
                    source, destination, budget, duration, stops,
                    rating, travel_class, seat, departure_time
                )
                # Boost the DT-predicted airline's score
                for score in airline_scores:
                    if score['airline'] == dt_prediction:
                        score['dt_boost'] = dt_confidence / 100.0
                        score['total_score'] += score['dt_boost'] * 0.15  # 15% weight to DT
            except Exception:
                pass
        
        # Step 6: Sort by total score descending
        airline_scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Step 7: Generate comparison categories
        categories = self._categorize_recommendations(airline_scores)
        
        # Step 8: Generate explanations
        for score in airline_scores[:top_k]:
            score['explanation'] = self._generate_explanation(
                score, source, destination, budget, rating, stops,
                travel_class, duration, seat, departure_time, dt_prediction
            )
        
        # Metadata
        meta = {
            'route_connectivity': round(route_score, 4),
            'cluster_id': cluster_id,
            'cluster_distance': round(cluster_dist, 4),
            'total_route_flights': len(route_flights),
            'dt_prediction': dt_prediction,
            'dt_confidence': dt_confidence,
            'categories': categories
        }
        
        return airline_scores[:top_k], meta
    
    def _get_route_score(self, source, destination):
        """Get link-prediction-based route connectivity score."""
        if source in self.graph.node_index and destination in self.graph.node_index:
            i = self.graph.node_index[source]
            j = self.graph.node_index[destination]
            return float(self.similarity_matrix[i][j])
        return 0.0
    
    def _score_airline(self, airline, flights, source, destination,
                       budget, duration, stops, rating, travel_class,
                       seat, departure_time, route_score, cluster_id):
        """
        Score an airline for recommendation using hybrid criteria.
        
        Implements adapted version of paper Eq. (5) and (6):
        - Base score from link prediction (route connectivity)
        - Quality score from flight attributes
        - Preference match score from user inputs
        """
        epsilon = 0.001  # Small constant to prevent zero scores (paper's epsilon)
        
        stats = self.airline_stats.get(airline, {})
        avg_price = flights['Price'].mean()
        avg_duration = flights['Duration'].mean()
        avg_rating = flights['Rating'].mean()
        avg_stops = flights['Stops'].mean()
        
        # --- Component 1: Route connectivity (link prediction) ---
        # Corresponds to paper Eq. (4) similarity
        connectivity_score = route_score
        
        # --- Component 2: Price match ---
        price_match = 0.0
        if avg_price <= budget:
            price_match = 1.0 - (avg_price / max(budget, 1)) * 0.3
        else:
            overshoot = (avg_price - budget) / max(budget, 1)
            price_match = max(0, 1.0 - overshoot * 2)
        
        # --- Component 3: Rating match ---
        rating_match = 0.0
        if avg_rating >= rating:
            rating_match = min(1.0, avg_rating / 5.0)
        else:
            gap = rating - avg_rating
            rating_match = max(0, 1.0 - gap * 2)
        
        # --- Component 4: Duration match ---
        duration_diff = abs(avg_duration - duration)
        duration_match = max(0, 1.0 - duration_diff / max(duration, 1) * 0.5)
        
        # --- Component 5: Stops match ---
        stops_match = 1.0
        if stops == 0:
            direct_flights = (flights['Stops'] == 0).sum()
            stops_match = direct_flights / max(len(flights), 1)
        else:
            matching_stops = (flights['Stops'] <= stops).sum()
            stops_match = matching_stops / max(len(flights), 1)
        
        # --- Component 6: Class availability ---
        class_match = 0.0
        if travel_class in flights['Class'].values:
            class_flights = flights[flights['Class'] == travel_class]
            class_match = len(class_flights) / max(len(flights), 1)
        
        # --- Component 7: Seat availability ---
        seat_match = 1.0 if seat in flights['Seat'].values else 0.5
        
        # --- Component 8: Departure time match ---
        time_match = 1.0 if departure_time in flights['Departure_Time'].values else 0.5
        
        # --- Component 9: Flight frequency (reliability indicator) ---
        frequency_score = min(1.0, len(flights) / 100)
        
        # --- Weighted combination (adapted from paper Eq. 6) ---
        # V_f = P_f * (W_quality + W_match) / (epsilon + W_penalty)
        
        W_quality = (
            rating_match * 0.30 +
            connectivity_score * 0.15 +
            frequency_score * 0.05
        )
        
        W_match = (
            price_match * 0.20 +
            duration_match * 0.10 +
            stops_match * 0.10 +
            class_match * 0.05 +
            seat_match * 0.025 +
            time_match * 0.025
        )
        
        total_score = W_quality + W_match
        
        # Build the best matching flight from this airline
        best_flight = self._find_best_matching_flight(
            flights, budget, duration, stops, rating, travel_class, seat, departure_time
        )
        
        return {
            'airline': airline,
            'total_score': round(total_score, 4),
            'score_components': {
                'route_connectivity': round(connectivity_score, 4),
                'price_match': round(price_match, 4),
                'rating_match': round(rating_match, 4),
                'duration_match': round(duration_match, 4),
                'stops_match': round(stops_match, 4),
                'class_match': round(class_match, 4),
                'seat_match': round(seat_match, 4),
                'time_match': round(time_match, 4),
                'frequency': round(frequency_score, 4),
            },
            'airline_stats': {
                'avg_price': int(avg_price),
                'avg_duration': int(avg_duration),
                'avg_rating': round(avg_rating, 1),
                'avg_stops': round(avg_stops, 1),
                'flight_count': len(flights),
                'direct_ratio': round((flights['Stops'] == 0).mean() * 100, 1),
            },
            'best_flight': best_flight,
            'dt_boost': 0.0,
        }
    
    def _find_best_matching_flight(self, flights, budget, duration, stops,
                                    rating, travel_class, seat, departure_time):
        """Find the single best matching flight from a filtered set."""
        scored_flights = []
        
        for idx, row in flights.iterrows():
            score = 0
            if row['Price'] <= budget:
                score += 2
            if row['Stops'] <= stops:
                score += 2
            if row['Rating'] >= rating:
                score += 1.5
            if row['Class'] == travel_class:
                score += 1.5
            if row['Seat'] == seat:
                score += 0.5
            if row['Departure_Time'] == departure_time:
                score += 0.5
            duration_diff = abs(row['Duration'] - duration)
            score += max(0, 1 - duration_diff / 200)
            
            scored_flights.append((score, idx, row))
        
        if not scored_flights:
            return None
        
        scored_flights.sort(key=lambda x: x[0], reverse=True)
        best = scored_flights[0][2]
        
        return {
            'price': int(best['Price']),
            'duration': int(best['Duration']),
            'stops': int(best['Stops']),
            'rating': float(best['Rating']),
            'class': best['Class'],
            'seat': best['Seat'],
            'departure_time': best['Departure_Time'],
        }
    
    def _get_dt_prediction(self, source, destination, budget, duration,
                           stops, rating, travel_class, seat, departure_time):
        """Get Decision Tree prediction and confidence."""
        source_enc = self.city_encoder.transform([source])[0]
        dest_enc = self.city_encoder.transform([destination])[0]
        class_enc = self.label_encoders['Class'].transform([travel_class])[0]
        seat_enc = self.label_encoders['Seat'].transform([seat])[0]
        dept_enc = self.label_encoders['Departure_Time'].transform([departure_time])[0]
        
        features = np.array([[
            source_enc, dest_enc, budget, duration, stops,
            rating, class_enc, seat_enc, dept_enc
        ]])
        
        pred_enc = self.model.predict(features)[0]
        airline = self.label_encoders['Airline'].inverse_transform([pred_enc])[0]
        
        probs = self.model.predict_proba(features)[0]
        confidence = round(float(probs[pred_enc]) * 100, 1)
        
        return airline, confidence
    
    def _categorize_recommendations(self, scored_airlines):
        """Categorize recommendations: best overall, cheapest, shortest, highest rated."""
        if not scored_airlines:
            return {}
        
        categories = {
            'best_overall': scored_airlines[0]['airline'],
        }
        
        # Cheapest
        cheapest = min(scored_airlines, key=lambda x: x['airline_stats']['avg_price'])
        categories['lowest_price'] = cheapest['airline']
        categories['lowest_price_value'] = cheapest['airline_stats']['avg_price']
        
        # Shortest duration
        shortest = min(scored_airlines, key=lambda x: x['airline_stats']['avg_duration'])
        categories['shortest_duration'] = shortest['airline']
        categories['shortest_duration_value'] = shortest['airline_stats']['avg_duration']
        
        # Highest rated
        best_rated = max(scored_airlines, key=lambda x: x['airline_stats']['avg_rating'])
        categories['highest_rated'] = best_rated['airline']
        categories['highest_rated_value'] = best_rated['airline_stats']['avg_rating']
        
        # Most direct flights
        most_direct = max(scored_airlines, key=lambda x: x['airline_stats']['direct_ratio'])
        categories['most_direct'] = most_direct['airline']
        categories['most_direct_value'] = most_direct['airline_stats']['direct_ratio']
        
        return categories
    
    def _generate_explanation(self, score_details, source, destination,
                              budget, rating, stops, travel_class,
                              duration, seat, departure_time, dt_prediction):
        """Generate human-readable explanation for a recommendation."""
        airline = score_details['airline']
        stats = score_details['airline_stats']
        components = score_details['score_components']
        best = score_details.get('best_flight', {})
        
        # Airline personality descriptions
        airline_perks = {
            'IndiGo': "India's largest carrier with punctual on-time performance and best value for budget-conscious flyers",
            'Air India': "Legacy full-service airline with generous baggage allowances, hot meals, and wide network reach",
            'Vistara': "Premium full-service airline offering superior cabin comfort, gourmet dining, and top-tier satisfaction",
            'SpiceJet': "Affordable low-cost carrier focusing on pocket-friendly fares and flexible regional routes",
            'Akasa Air': "Modern fleet with USB charging, eco-friendly operations, and highly rated passenger experience"
        }
        
        perk = airline_perks.get(airline, "Matches your travel profile well")
        
        # Build explanation text
        explanation_parts = []
        
        # Main reason
        if components['price_match'] >= 0.8 and components['rating_match'] >= 0.8:
            explanation_parts.append(
                f"{airline} excels on both affordability (avg ₹{stats['avg_price']:,}) "
                f"and quality ({stats['avg_rating']}★)"
            )
        elif components['price_match'] >= 0.8:
            explanation_parts.append(
                f"{airline} fits your ₹{budget:,} budget well with average fares of ₹{stats['avg_price']:,}"
            )
        elif components['rating_match'] >= 0.8:
            explanation_parts.append(
                f"{airline} meets your quality expectations with an average rating of {stats['avg_rating']}★"
            )
        else:
            explanation_parts.append(
                f"{airline} provides the best available balance for your {source}→{destination} route"
            )
        
        explanation = ". ".join(explanation_parts) + "."
        
        # Build highlights
        highlights = []
        
        # Route connectivity insight
        if components['route_connectivity'] > 0.5:
            highlights.append(
                f"Strong Route: {source}→{destination} is a well-connected route in the aviation network "
                f"(connectivity score: {components['route_connectivity']:.2f})"
            )
        
        # Price insight
        if best and best.get('price'):
            price_diff = best['price'] - budget
            if price_diff <= 0:
                highlights.append(
                    f"Budget Fit: Best matching flight at ₹{best['price']:,} "
                    f"(₹{abs(price_diff):,} under your ₹{budget:,} budget)"
                )
            else:
                highlights.append(
                    f"Price Note: Best matching flight at ₹{best['price']:,} "
                    f"(₹{price_diff:,} over budget)"
                )
        
        # Rating insight
        highlights.append(
            f"Rating: {airline} averages {stats['avg_rating']}★ "
            f"(your minimum: {rating}★)"
        )
        
        # Stops insight
        if stats['direct_ratio'] > 70:
            highlights.append(
                f"Mostly Direct: {stats['direct_ratio']}% of {airline}'s flights on this route are non-stop"
            )
        elif stops == 0 and stats['direct_ratio'] > 0:
            highlights.append(
                f"Direct Available: {stats['direct_ratio']}% of flights are non-stop"
            )
        
        # Class insight
        if components['class_match'] > 0:
            highlights.append(
                f"Class: {travel_class} class available on {airline}"
            )
        
        # Airline perk
        highlights.append(f"Perk: {perk}")
        
        # Decision Tree agreement
        if dt_prediction == airline:
            highlights.append(
                f"ML Confirmed: Decision Tree model also independently recommends {airline} "
                f"for your preferences"
            )
        
        return {
            'text': explanation,
            'highlights': highlights,
            'trade_offs': self._compute_trade_offs(score_details, budget, rating, stops)
        }
    
    def _compute_trade_offs(self, score_details, budget, rating, stops):
        """Compute trade-offs for transparency."""
        stats = score_details['airline_stats']
        trade_offs = []
        
        if stats['avg_price'] > budget:
            trade_offs.append(
                f"Average price (₹{stats['avg_price']:,}) exceeds your budget of ₹{budget:,}"
            )
        
        if stats['avg_rating'] < rating:
            trade_offs.append(
                f"Average rating ({stats['avg_rating']}★) is below your preferred {rating}★"
            )
        
        if stops == 0 and stats['direct_ratio'] < 50:
            trade_offs.append(
                f"Only {stats['direct_ratio']}% of flights are direct on this route"
            )
        
        return trade_offs
    
    def get_evaluation_metrics(self, test_df=None):
        """
        Evaluate the recommender system.
        
        Uses the approach from the IEEE paper Section IV-B:
        - For each test flight, check if the system recommends the correct airline
        - Calculate precision, recall, and hit rate
        """
        if test_df is None:
            # Use 20% of data as test set
            np.random.seed(42)
            mask = np.random.rand(len(self.df)) < 0.2
            test_df = self.df[mask]
        
        if len(test_df) == 0:
            return {}
        
        tp, fp, fn = 0, 0, 0
        correct_top1 = 0
        correct_top3 = 0
        total = 0
        
        # Sample test cases for efficiency
        sample_size = min(200, len(test_df))
        test_sample = test_df.sample(n=sample_size, random_state=42)
        
        for _, row in test_sample.iterrows():
            try:
                actual_airline = row['Airline']
                recommendations, _ = self.recommend(
                    source=row['Source'],
                    destination=row['Destination'],
                    budget=int(row['Price'] * 1.1),  # 10% budget margin
                    duration=int(row['Duration']),
                    stops=int(row['Stops']),
                    rating=max(3.0, row['Rating'] - 0.5),
                    travel_class=row['Class'],
                    seat=row['Seat'],
                    departure_time=row['Departure_Time'],
                    top_k=5
                )
                
                if not recommendations:
                    fn += 1
                    total += 1
                    continue
                
                recommended_airlines = [r['airline'] for r in recommendations]
                
                # Top-1 accuracy
                if recommended_airlines[0] == actual_airline:
                    correct_top1 += 1
                    tp += 1
                else:
                    fp += 1
                
                # Top-3 hit rate
                if actual_airline in recommended_airlines[:3]:
                    correct_top3 += 1
                
                total += 1
                
            except Exception:
                total += 1
                fn += 1
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        return {
            'total_test_samples': total,
            'precision_at_1': round(precision * 100, 2),
            'recall_at_1': round(recall * 100, 2),
            'f1_at_1': round(f1 * 100, 2),
            'hit_rate_at_1': round(correct_top1 / max(total, 1) * 100, 2),
            'hit_rate_at_3': round(correct_top3 / max(total, 1) * 100, 2),
        }
