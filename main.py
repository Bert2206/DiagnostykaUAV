import pandas as pd
import re
import  glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Wczytanie danych z pliku
folder_path = r'C:\Users\kubas\PycharmProjects\DiagnostykaUAV\Parrot_Bebop_2\Range_data'
file_pattern = folder_path + '\\*.csv'
print(file_pattern)
data_frames = []

for file_path in glob.glob(file_pattern):
    data = pd.read_csv(file_path)
    match = re.search(r'_(\d{4}).csv$', file_path)
    if match:
        fault_code = match.group(1)
        data['Fault_Code'] = fault_code
    else:
        print(f"Nieprawidłowa nazwa pliku: {file_path}")
    data_frames.append(data)

all_data = pd.concat(data_frames, ignore_index=True)
print(all_data.head())