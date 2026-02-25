import numpy as np
import pandas as pd 

def replace_categorical_by_numerical(df):
    df = df.copy()

    df['Levy'] = df['Levy'].apply(pd.to_numeric , errors='coerce' )
    df['Levy'].fillna(df['Levy'].min() ,inplace=True)
    
    df['Engine volume'] = df['Engine volume'].str.replace('Turbo' , '')
    df['Engine volume'] = pd.to_numeric(df['Engine volume'])
    
    df['Mileage'] = df['Mileage'].str.replace('km'.lower() , '')
    df['Mileage'] = pd.to_numeric(df['Mileage'])
    
    return df


def clean_outliers(df, cols):
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df


# Add New Features That Will Enhance The Model
def engineer_features(df):
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Prod. year']
        
    return df


def preprocessing_pipeline(df: pd.DataFrame):
    print("Preprocessing Started....")
    
    print(f"Initial Shape : {df.shape}")
    
    df.drop_duplicates(inplace=True)    
    print(f"After dropping duplicates: {df.shape}")

    print("Replacing categorical values...")
    df = replace_categorical_by_numerical(df)
    
    df = clean_outliers(df, ['Price', 'Levy', 'Engine volume', 'Mileage']) # clean outliers since we want to predict normal prices (we don't want the model to learn wrong prices)
    print(f"After cleaning outliers: {df.shape}")
    
    print("Feature engineering...")
    df = engineer_features(df)
    
    print("Dropping columns...")
    df = df.drop(['ID', 'Doors', 'Prod. year'], axis=1)

    print("Final shape:", df.shape)

    return df