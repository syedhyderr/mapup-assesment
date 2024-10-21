''' Question 1- Reverese List by N elements'''
def reverse_in_groups(lst, n):
    result = []
    length = len(lst)

    for i in range(0, length, n):
        
        end = min(i + n, length)
        
        for j in range(end - 1, i - 1, -1):
            result.append(lst[j])

    return result





'''Question 2- List and dictionary'''

def group_strings_by_length(strings):
    grouped = {}

    for string in strings:
        length = len(string)
        if length not in grouped:
            grouped[length] = []
        grouped[length].append(string)

    sorted_grouped = dict(sorted(grouped.items()))

    return sorted_grouped






'''Flatten a nested dictonary'''

def flatten_dictionary(nested_dict, parent_key='', sep='.'):
    items = {}

    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            
            items.update(flatten_dictionary(value, new_key, sep=sep))
        elif isinstance(value, list):
            
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_dictionary(item, f"{new_key}[{index}]", sep=sep))
                else:
                    items[f"{new_key}[{index}]"] = item
        else:
            
            items[new_key] = value

    return items




'''Generate unoque permutations'''
def permute_unique(nums):
    def backtrack(start):
        if start == len(nums):
            results.append(nums[:])  
            return

        seen = set()  
        for i in range(start, len(nums)):
            if nums[i] in seen:  
                continue
            seen.add(nums[i])
            
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)  
            
            nums[start], nums[i] = nums[i], nums[start]

    results = []
    nums.sort()  
    backtrack(0)
    return results





'''Find all dates in text'''
import re

def find_all_dates(text):
    
    date_pattern = r'(\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b)'
    matches = re.findall(date_pattern, text)
    
    return matches






'''Decode polyline,convert to dataframes with distances'''
import polyline
import pandas as pd
import math

def haversine(lat1, lon1, lat2, lon2):
   
    R = 6371000  
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def decode_polyline_and_calculate_distance(polyline_str):
    
    coordinates = polyline.decode(polyline_str)

    
    data = {
        'latitude': [],
        'longitude': [],
        'distance': []
    }

   
    previous_lat, previous_lon = None, None

    for lat, lon in coordinates:
        data['latitude'].append(lat)
        data['longitude'].append(lon)
        
        if previous_lat is None or previous_lon is None:
            
            data['distance'].append(0.0)
        else:
            
            distance = haversine(previous_lat, previous_lon, lat, lon)
            data['distance'].append(distance)
        
        
        previous_lat, previous_lon = lat, lon

    
    df = pd.DataFrame(data)

    return df





'''Matrix rotation and transformation'''


def rotate_and_transform(matrix):
    n = len(matrix)

    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])  
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) 
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  

    return final_matrix



'''Time check'''

import pandas as pd


def verify_timestamp_completeness(df):
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    grouped = df.groupby(['id', 'id_2'])

    def check_completeness(group):
        unique_days = group['start'].dt.date.nunique()
        full_day_coverage = (group['end'].max() - group['start'].min()).days >= 1
        return not (full_day_coverage and unique_days >= 7)

    result = grouped.apply(check_completeness)

    return result



