''' Distance matrix calculation'''

import pandas as pd
import numpy as np

def calculate_distance_matrix(file_path):
   
    df = pd.read_csv(MapUpDA assesment/datasets/dataset-2.csv)

    toll_ids = pd.concat([df['Toll_A'], df['Toll_B']]).unique()
    toll_ids.sort()  
    dist_matrix = pd.DataFrame(np.inf, index=toll_ids, columns=toll_ids)

    np.fill_diagonal(dist_matrix.values, 0)

    for _, row in df.iterrows():
        toll_a = row['Toll_A']
        toll_b = row['Toll_B']
        distance = row['Distance']

        dist_matrix.at[toll_a, toll_b] = distance
        dist_matrix.at[toll_b, toll_a] = distance

    toll_ids = dist_matrix.index
    for k in toll_ids:
        for i in toll_ids:
            for j in toll_ids:
                
                if dist_matrix.at[i, j] > dist_matrix.at[i, k] + dist_matrix.at[k, j]:
                    dist_matrix.at[i, j] = dist_matrix.at[i, k] + dist_matrix.at[k, j]

    return dist_matrix


'''Unroll distance matrix'''

import pandas as pd
import itertools

def unroll_distance_matrix(distance_df):
    if not all(col in distance_df.columns for col in ['id_start', 'id_end', 'distance']):
        raise ValueError("Input DataFrame must contain 'id_start', 'id_end', and 'distance' columns.")
    unique_ids = distance_df['id_start'].unique()
    combinations = [(start, end) for start in unique_ids for end in unique_ids if start != end]
    combinations_df = pd.DataFrame(combinations, columns=['id_start', 'id_end'])
    result_df = pd.merge(combinations_df, distance_df, on=['id_start', 'id_end'], how='left')
    
    return result_df


'''Finding IDS with percentage threshold'''

import pandas as pd
import numpy as np

def find_ids_within_ten_percentage_threshold(distance_df, reference_id):
    if not all(col in distance_df.columns for col in ['id_start', 'id_end', 'distance']):
        raise ValueError("Input DataFrame must contain 'id_start', 'id_end', and 'distance' columns.")
    
    reference_distances = distance_df[distance_df['id_start'] == reference_id]['distance']
    if reference_distances.empty:
        raise ValueError(f"No distances found for id_start = {reference_id}")
    
    average_distance = reference_distances.mean()
    ten_percent_threshold = average_distance * 0.1
    lower_bound = average_distance - ten_percent_threshold
    upper_bound = average_distance + ten_percent_threshold
    average_distances = distance_df.groupby('id_start')['distance'].mean()

    ids_within_threshold = average_distances[(average_distances >= lower_bound) & (average_distances <= upper_bound)]

    return sorted(ids_within_threshold.index.tolist())


'''Calculate tollgates'''
import pandas as pd

def calculate_toll_rate(distance_df):
    if 'distance' not in distance_df.columns:
        raise ValueError("Input DataFrame must contain 'distance' column.")
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle_type, coefficient in rate_coefficients.items():
        distance_df[vehicle_type] = distance_df['distance'] * coefficient

    return distance_df




'''Calculate time-based tollgates'''
import pandas as pd
import numpy as np
import datetime

def calculate_time_based_toll_rates(distance_df):
    required_columns = ['id_start', 'id_end', 'distance', 'moto', 'car', 'rv', 'bus', 'truck']
    if not all(col in distance_df.columns for col in required_columns):
        raise ValueError("Input DataFrame must contain 'id_start', 'id_end', 'distance', and vehicle columns.")

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    weekday_time_slots = {
        '00:00:00': '10:00:00',  
        '10:00:00': '18:00:00',  
        '18:00:00': '23:59:59'   
    }
    weekday_discount_factors = {
        '00:00:00': 0.8,
        '10:00:00': 1.2,
        '18:00:00': 0.8
    }

    weekend_discount_factor = 0.7

    distance_df['startDay'] = np.random.choice(days_of_week, size=len(distance_df))
    distance_df['endDay'] = np.random.choice(days_of_week, size=len(distance_df))
    distance_df['starTime'] = pd.to_datetime(np.random.choice([f"{h:02}:00:00" for h in range(24)]), format='%H:%M:%S').dt.time
    distance_df['endTime'] = pd.to_datetime(np.random.choice([f"{h:02}:00:00" for h in range(24)]), format='%H:%M:%S').dt.time

    for index, row in distance_df.iterrows():
        startDay = row['start_day']
        starTime = row['start_time']
        
        if start_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:  
            if start_time >= datetime.time(0, 0) and start_time < datetime.time(10, 0):
                discount = weekday_discount_factors['00:00:00']
            elif start_time >= datetime.time(10, 0) and start_time < datetime.time(18, 0):
                discount = weekday_discount_factors['10:00:00']
            else:
                discount = weekday_discount_factors['18:00:00']
        else:  
            discount = weekend_discount_factor

        distance_df.at[index, 'moto'] *= discount
        distance_df.at[index, 'car'] *= discount
        distance_df.at[index, 'rv'] *= discount
        distance_df.at[index, 'bus'] *= discount
        distance_df.at[index, 'truck'] *= discount

    return distance_df








