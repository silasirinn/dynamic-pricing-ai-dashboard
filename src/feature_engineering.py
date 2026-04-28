import pandas as pd
import numpy as np

def extract_date_features(df, date_column='date'):
    """
    Tarih (datetime) sütunundan 'month', 'day_of_week' ve 'is_weekend' özelliklerini üretir.
    Böylece model, zaman içindeki örüntüleri anlayabilir.
    """
    if date_column in df.columns:
        df['month'] = df[date_column].dt.month
        df['day_of_week'] = df[date_column].dt.dayofweek
        # Hafta sonu bilgisi (5: Cumartesi, 6: Pazar)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Döngüsel özellikleri de ekleyebiliriz (Ay için)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        print("Tarih tabanlı özellikler ('month', 'day_of_week', 'is_weekend', döngüsel aylar) başarıyla üretildi.")
    else:
        print(f"Uyarı: '{date_column}' sütunu bulunamadığı için tarih özellikleri üretilemedi.")
    return df

def add_advanced_features(df):
    """
    Model performansını artırmak için EDA bulgularına dayalı, ancak VERİ SIZINTISI İÇERMEYEN
    ekstra özellikleri ekler:
    - age_group: Yaş aralıkları
    - category_quantity_mean: Kategorilere göre ortalama miktar
    """
    # 1. Yaş Gruplandırması
    if 'age' in df.columns:
        bins = [0, 25, 35, 50, 100]
        labels = ['18-25', '26-35', '36-50', '51+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)
        
    # 2. Kategori Bazlı Ortalamalar
    # Sadece quantity ortalaması alıyoruz, price_per_unit verisinden tamamen kaçınıyoruz.
    if 'product_category' in df.columns:
        if 'quantity' in df.columns:
            df['category_quantity_mean'] = df.groupby('product_category')['quantity'].transform('mean')
            
    print("Sızıntı içermeyen gelişmiş özellikler (age_group, category_quantity_mean) başarıyla eklendi.")
    return df

def apply_one_hot_encoding(df, categorical_columns=['gender', 'product_category', 'age_group']):
    """
    Belirtilen kategorik sütunlar için One-Hot Encoding işlemi uygular.
    drop_first=True parametresi ile dummy variable trap (çoklu doğrusal bağlantı) durumu önlenir.
    """
    existing_columns = [col for col in categorical_columns if col in df.columns]
    if existing_columns:
        df = pd.get_dummies(df, columns=existing_columns, drop_first=True)
        print(f"One-Hot Encoding uygulanan sütunlar: {existing_columns}")
    else:
        print("Uyarı: Belirtilen kategorik sütunlar veri setinde bulunamadı, encoding atlandı.")
    return df
