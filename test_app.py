"""
test_app.py
Automated tests for the Smart Flight Recommendation System.
Tests all API endpoints, validation, and recommendation quality.
"""

import json
import sys
sys.path.insert(0, '.')
from app import app

def run_tests():
    client = app.test_client()
    passed = 0
    failed = 0

    def test(name, fn):
        nonlocal passed, failed
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            fn(client)
            print("RESULT: PASS")
            passed += 1
        except Exception as e:
            print(f"RESULT: FAIL - {e}")
            failed += 1

    # Test 1
    def test_homepage(client):
        resp = client.get('/')
        assert resp.status_code == 200, f"Status {resp.status_code}"
        assert b'AeroMatch' in resp.data or b'Flight' in resp.data, "Missing page content"
        print(f"  HTML length: {len(resp.data)} bytes")

    # Test 2
    def test_valid_prediction(client):
        payload = {
            'source': 'Chennai', 'destination': 'Delhi',
            'budget': 5000, 'duration': 150, 'stops': 0,
            'rating': 4.0, 'travel_class': 'Economy',
            'seat': 'Window', 'departure_time': '10 AM'
        }
        resp = client.post('/predict', json=payload)
        data = resp.get_json()
        assert resp.status_code == 200, f"Status {resp.status_code}"
        assert data.get('success') == True, f"Not successful: {data.get('error')}"
        recs = data.get('recommendations', [])
        assert len(recs) >= 1, "No recommendations returned"
        print(f"  Recommendations: {len(recs)}")
        for r in recs:
            print(f"    #{r['rank']} {r['airline']:>12} - Score: {r['score']}%")
        # Check legacy format
        assert 'recommended_airline' in data, "Missing legacy field"
        assert 'explanation' in data, "Missing explanation"
        assert 'highlights' in data, "Missing highlights"
        # Check categories
        cats = data.get('meta', {}).get('categories', {})
        print(f"  Categories: {list(cats.keys())}")
        assert 'best_overall' in cats, "Missing best_overall category"

    # Test 3
    def test_same_city_validation(client):
        resp = client.post('/predict', json={
            'source': 'Delhi', 'destination': 'Delhi',
            'budget': 5000, 'duration': 150, 'stops': 0,
            'rating': 4.0, 'travel_class': 'Economy',
            'seat': 'Window', 'departure_time': '10 AM'
        })
        data = resp.get_json()
        assert resp.status_code == 400, f"Should be 400, got {resp.status_code}"
        assert 'same' in data.get('error', '').lower(), "Missing error about same city"
        print(f"  Error: {data.get('error')}")

    # Test 4
    def test_different_route(client):
        payload = {
            'source': 'Mumbai', 'destination': 'Kolkata',
            'budget': 8000, 'duration': 200, 'stops': 1,
            'rating': 3.5, 'travel_class': 'Economy',
            'seat': 'Aisle', 'departure_time': '6 PM'
        }
        resp = client.post('/predict', json=payload)
        data = resp.get_json()
        assert data.get('success') == True
        recs = data.get('recommendations', [])
        assert len(recs) >= 1
        print(f"  Recommendations: {len(recs)}")
        for r in recs:
            print(f"    #{r['rank']} {r['airline']:>12} - Score: {r['score']}%")
            # Check explanations exist
            assert r.get('explanation'), f"Missing explanation for {r['airline']}"
            assert r.get('highlights'), f"Missing highlights for {r['airline']}"

    # Test 5
    def test_business_class(client):
        payload = {
            'source': 'Delhi', 'destination': 'Mumbai',
            'budget': 25000, 'duration': 150, 'stops': 0,
            'rating': 4.5, 'travel_class': 'Business',
            'seat': 'Window', 'departure_time': '8 AM'
        }
        resp = client.post('/predict', json=payload)
        data = resp.get_json()
        assert data.get('success') == True
        recs = data.get('recommendations', [])
        print(f"  Recommendations: {len(recs)}")
        for r in recs:
            print(f"    #{r['rank']} {r['airline']:>12} - Score: {r['score']}% - Trade-offs: {len(r.get('trade_offs',[]))}")

    # Test 6
    def test_score_components(client):
        payload = {
            'source': 'Bangalore', 'destination': 'Hyderabad',
            'budget': 4000, 'duration': 70, 'stops': 0,
            'rating': 4.0, 'travel_class': 'Economy',
            'seat': 'Window', 'departure_time': '10 AM'
        }
        resp = client.post('/predict', json=payload)
        data = resp.get_json()
        assert data.get('success') == True
        recs = data.get('recommendations', [])
        for r in recs[:1]:
            sc = r.get('score_components', {})
            print(f"  Score components for {r['airline']}:")
            for k, v in sc.items():
                print(f"    {k}: {v}")
            assert 'route_connectivity' in sc, "Missing route_connectivity"
            assert 'price_match' in sc, "Missing price_match"

    # Test 7
    def test_feedback_submit(client):
        payload = {
            'source': 'Chennai', 'destination': 'Delhi', 'budget': 5000,
            'travel_class': 'Economy', 'recommended_airline': 'IndiGo',
            'rating': 5, 'comment': 'Automated test feedback'
        }
        resp = client.post('/feedback', json=payload)
        data = resp.get_json()
        assert data.get('success') == True, f"Feedback failed: {data}"
        print(f"  Message: {data.get('message')}")

    # Test 8
    def test_feedback_retrieve(client):
        resp = client.get('/api/feedback')
        data = resp.get_json()
        assert resp.status_code == 200
        print(f"  Source: {data.get('source')}")
        print(f"  Records: {len(data.get('data', []))}")

    # Test 9
    def test_routes_api(client):
        resp = client.get('/api/routes')
        data = resp.get_json()
        assert resp.status_code == 200
        assert len(data.get('cities', [])) == 6, "Should have 6 cities"
        assert len(data.get('routes', [])) == 30, f"Expected 30 routes, got {len(data.get('routes', []))}"
        print(f"  Cities: {data.get('cities')}")
        print(f"  Routes: {len(data.get('routes', []))}")
        print(f"  Classes: {data.get('classes')}")

    # Test 10
    def test_evaluation_api(client):
        resp = client.get('/api/evaluation')
        data = resp.get_json()
        assert resp.status_code == 200
        assert data.get('success') == True
        metrics = data.get('metrics', {})
        model_info = data.get('model_info', {})
        print(f"  Metrics: {json.dumps(metrics, indent=4)}")
        print(f"  Model: {model_info.get('type')}")
        print(f"  Graph nodes: {model_info.get('graph_nodes')}")
        print(f"  Clusters: {model_info.get('n_clusters')}")

    # Test 11
    def test_all_routes(client):
        cities = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
        success_count = 0
        for src in cities:
            for dst in cities:
                if src == dst:
                    continue
                payload = {
                    'source': src, 'destination': dst,
                    'budget': 6000, 'duration': 150, 'stops': 0,
                    'rating': 4.0, 'travel_class': 'Economy',
                    'seat': 'Window', 'departure_time': '10 AM'
                }
                resp = client.post('/predict', json=payload)
                data = resp.get_json()
                if data.get('success') and len(data.get('recommendations', [])) > 0:
                    success_count += 1
        print(f"  Successful routes: {success_count}/30")
        assert success_count == 30, f"Only {success_count}/30 routes returned recommendations"

    # Run all tests
    test("Homepage loads", test_homepage)
    test("Valid recommendation (Chennai->Delhi)", test_valid_prediction)
    test("Same city validation", test_same_city_validation)
    test("Different route (Mumbai->Kolkata)", test_different_route)
    test("Business class high budget", test_business_class)
    test("Score components present", test_score_components)
    test("Feedback submission", test_feedback_submit)
    test("Feedback retrieval", test_feedback_retrieve)
    test("Routes API", test_routes_api)
    test("Evaluation API", test_evaluation_api)
    test("All 30 routes work", test_all_routes)

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} PASSED, {failed} FAILED out of {passed+failed} tests")
    print(f"{'='*60}")

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
