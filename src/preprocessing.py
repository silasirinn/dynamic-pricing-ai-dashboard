import pandas as pd

def handle_missing_values(df):
    """
    Veri setindeki eksik değerleri (NaN) kontrol eder.
    Varsa bu değerlere sahip satırları temizleyerek döner.
    """
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Eksik değerler bulundu: Toplam {missing_count} adet. Eksik veri içeren satırlar siliniyor.")
        df = df.dropna().reset_index(drop=True)
    else:
        print("Veri setinde eksik değer bulunmadı.")
    return df

def convert_to_datetime(df, date_column='date'):
    """
    Belirtilen tarih sütununu Pandas datetime formatına çevirir.
    """
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        print(f"'{date_column}' sütunu datetime formatına başarıyla çevrildi.")
    else:
        print(f"Uyarı: '{date_column}' sütunu bulunamadı.")
    return df

def drop_unnecessary_columns(df, columns_to_drop):
    """
    Listede belirtilen ve gereksiz olduğu düşünülen sütunları veri setinden çıkarır.
    """
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)
        print(f"Çıkarılan sütunlar: {existing_columns}")
    else:
        print("Çıkarılacak sütunlar zaten veri setinde mevcut değil.")
    return df
