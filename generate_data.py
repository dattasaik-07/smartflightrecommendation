"""
generate_data.py
Generates a realistic synthetic dataset for the Airline Recommendation System.
Dataset specifications:
- Records: 10,000 synthetic rows
- 6 Indian Cities: Chennai, Delhi, Mumbai, Kolkata, Bangalore, Hyderabad
- 5 Airlines: IndiGo, Air India, SpiceJet, Vistara, Akasa Air
- 9 Features: Source, Destination, Price, Duration, Stops, Rating, Class, Seat, Departure_Time
- Target: Airline
"""

import numpy as np
import pandas as pd

def generate_airline_dataset(num_records=10000, random_seed=42):
    np.random.seed(random_seed)

    cities = ['Chennai', 'Delhi', 'Mumbai', 'Kolkata', 'Bangalore', 'Hyderabad']
    classes = ['Economy', 'Premium Economy', 'Business']
    seats = ['Window', 'Aisle', 'Middle']
    departure_slots = ['6 AM', '8 AM', '10 AM', '12 PM', '2 PM', '4 PM', '6 PM', '8 PM', '10 PM']
    
    # Distance / duration mapping approximation between cities (in minutes direct)
    city_durations = {
        ('Chennai', 'Delhi'): 160,
        ('Chennai', 'Mumbai'): 120,
        ('Chennai', 'Kolkata'): 135,
        ('Chennai', 'Bangalore'): 55,
        ('Chennai', 'Hyderabad'): 75,
        ('Delhi', 'Mumbai'): 130,
        ('Delhi', 'Kolkata'): 130,
        ('Delhi', 'Bangalore'): 170,
        ('Delhi', 'Hyderabad'): 135,
        ('Mumbai', 'Kolkata'): 150,
        ('Mumbai', 'Bangalore'): 100,
        ('Mumbai', 'Hyderabad'): 90,
        ('Kolkata', 'Bangalore'): 155,
        ('Kolkata', 'Hyderabad'): 125,
        ('Bangalore', 'Hyderabad'): 70
    }
    
    def get_base_duration(c1, c2):
        pair = (c1, c2) if (c1, c2) in city_durations else (c2, c1)
        return city_durations.get(pair, 120)

    # 5 Airlines with distinct pricing, class mix, and customer ratings
    airlines = ['IndiGo', 'Air India', 'SpiceJet', 'Vistara', 'Akasa Air']
    
    records = []
    
    for _ in range(num_records):
        airline = np.random.choice(airlines, p=[0.32, 0.22, 0.16, 0.18, 0.12])
        
        # Pick source and destination (ensuring source != destination)
        source = np.random.choice(cities)
        dest = np.random.choice([c for c in cities if c != source])
        
        base_dur = get_base_duration(source, dest)
        
        # Stops distribution based on airline type
        if airline in ['IndiGo', 'Akasa Air']:
            stops = np.random.choice([0, 1, 2], p=[0.76, 0.21, 0.03])
        elif airline == 'Vistara':
            stops = np.random.choice([0, 1, 2], p=[0.72, 0.24, 0.04])
        elif airline == 'Air India':
            stops = np.random.choice([0, 1, 2], p=[0.55, 0.35, 0.10])
        else: # SpiceJet
            stops = np.random.choice([0, 1, 2], p=[0.65, 0.30, 0.05])
            
        # Duration based on direct or stops + small variation
        if stops == 0:
            duration = int(base_dur + np.random.normal(0, 10))
        elif stops == 1:
            duration = int(base_dur + np.random.uniform(90, 180))
        else:
            duration = int(base_dur + np.random.uniform(200, 320))
        duration = max(45, duration)

        # Travel class distribution
        if airline == 'IndiGo':
            travel_class = np.random.choice(classes, p=[0.94, 0.06, 0.00])
        elif airline == 'Akasa Air':
            travel_class = np.random.choice(classes, p=[0.97, 0.03, 0.00])
        elif airline == 'SpiceJet':
            travel_class = np.random.choice(classes, p=[0.92, 0.08, 0.00])
        elif airline == 'Air India':
            travel_class = np.random.choice(classes, p=[0.68, 0.14, 0.18])
        else: # Vistara
            travel_class = np.random.choice(classes, p=[0.52, 0.28, 0.20])
            
        # Seat preference
        seat = np.random.choice(seats, p=[0.42, 0.40, 0.18])
        
        # Departure time slot
        departure_time = np.random.choice(departure_slots)
        
        # Rating & Price calculation with realistic patterns
        if airline == 'SpiceJet':
            rating = round(np.clip(np.random.normal(3.5, 0.35), 2.8, 4.3), 1)
            base_price = 3200 + (base_dur * 8)
            if travel_class == 'Premium Economy':
                base_price *= 1.4
            price = int(np.random.normal(base_price, 450))
            
        elif airline == 'Akasa Air':
            rating = round(np.clip(np.random.normal(4.3, 0.28), 3.7, 4.9), 1)
            base_price = 3600 + (base_dur * 9)
            if travel_class == 'Premium Economy':
                base_price *= 1.35
            price = int(np.random.normal(base_price, 500))
            
        elif airline == 'IndiGo':
            rating = round(np.clip(np.random.normal(4.2, 0.30), 3.5, 4.8), 1)
            base_price = 4200 + (base_dur * 10)
            if travel_class == 'Premium Economy':
                base_price *= 1.45
            price = int(np.random.normal(base_price, 600))
            
        elif airline == 'Air India':
            rating = round(np.clip(np.random.normal(3.9, 0.40), 3.0, 4.6), 1)
            base_price = 4800 + (base_dur * 12)
            if travel_class == 'Premium Economy':
                base_price *= 1.6
            elif travel_class == 'Business':
                base_price *= 3.0
            price = int(np.random.normal(base_price, 900))
            
        else: # Vistara
            rating = round(np.clip(np.random.normal(4.7, 0.20), 4.1, 5.0), 1)
            base_price = 5600 + (base_dur * 13)
            if travel_class == 'Premium Economy':
                base_price *= 1.7
            elif travel_class == 'Business':
                base_price *= 3.4
            price = int(np.random.normal(base_price, 1000))

        # Adjust price slightly for stops
        if stops == 1:
            price = int(price * 0.92)  # connecting often cheaper
        elif stops == 2:
            price = int(price * 0.85)
            
        price = max(2200, price)
        # Round price to neat hundreds
        price = int(round(price / 100.0)) * 100

        records.append({
            'Price': price,
            'Duration': duration,
            'Stops': stops,
            'Rating': rating,
            'Class': travel_class,
            'Seat': seat,
            'Source': source,
            'Destination': dest,
            'Departure_Time': departure_time,
            'Airline': airline
        })

    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    print("Generating synthetic dataset of 10,000 airline records...")
    df = generate_airline_dataset(10000)
    output_file = 'airline_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"Dataset generated successfully and saved to '{output_file}'!")
    print(f"Total rows: {len(df)}")
    print("\nSample records:")
    print(df.head())
    print("\nAirline distribution:")
    print(df['Airline'].value_counts())
