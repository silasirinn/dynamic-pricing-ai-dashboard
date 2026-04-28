import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# src altındaki diğer modülleri içeri aktarıyoruz
from data_loader import load_data
from preprocessing import handle_missing_values, convert_to_datetime, drop_unnecessary_columns
from feature_engineering import extract_date_features, add_advanced_features, apply_one_hot_encoding

def main():
    """
    Veri yükleme, ön işleme, sızıntı içermeyen özellik mühendisliği 
    ve sabit hiperparametreler ile model eğitimi süreçlerini çalıştırır. 
    En iyi modeli outputs/models klasörüne kaydeder.
    """
    # 1. Veri Yükleme
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_sales_dataset.csv')
    df = load_data(data_path)
    
    if df is None:
        print("Uyarı: Veri yüklenemedi.")
        return
        
    print("\n--- Veri Ön İşleme ve Özellik Mühendisliği Başlıyor ---")
    df = handle_missing_values(df)
    df = convert_to_datetime(df, date_column='date')
    
    # Sızıntı içermeyen gelişmiş özellikleri ekliyoruz
    df = add_advanced_features(df)
    df = extract_date_features(df, date_column='date')
    
    # Kategorik değişkenleri encode ediyoruz
    df = apply_one_hot_encoding(df, categorical_columns=['gender', 'product_category', 'age_group'])
    
    # Modelde kullanılmayacak sütunları çıkarıyoruz (Sızıntıyı Önleme)
    # price_per_unit, total_amount'u doğrudan belirlediği için çıkarılır.
    columns_to_drop = ['transaction_id', 'customer_id', 'date', 'price_per_unit']
    df = drop_unnecessary_columns(df, columns_to_drop=columns_to_drop)
        
    print("\n--- Model İçin Veri Hazırlığı ---")
    target_column = 'total_amount'
    if target_column not in df.columns:
        print(f"Hata: Hedef değişken '{target_column}' bulunamadı.")
        return
        
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    rf_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 4,
        'random_state': 42
    }

    print("\n=== Senaryo 1: Train/Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Eğitim verisi: {X_train.shape[0]} satır, Test verisi: {X_test.shape[0]} satır")
    
    print("\n--- Model Eğitimi (Leakage-Free) Başlıyor ---")
    rf = RandomForestRegressor(**rf_params)
    
    rf.fit(X_train, y_train)
    
    # Test seti üzerinde değerlendirme
    predictions = rf.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("\n--- Senaryo 1 Sonuçları ---")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    print("\n=== Senaryo 2: 5-Katlı Çapraz Doğrulama ===")
    print(
        "İkinci senaryoda modelin daha güvenilir biçimde değerlendirilmesi için "
        "5 katlı çapraz doğrulama (5-Fold Cross Validation) uygulanmıştır."
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        RandomForestRegressor(**rf_params),
        X,
        y,
        cv=cv,
        scoring={
            'mae': 'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error',
            'r2': 'r2'
        },
        return_train_score=False
    )

    cv_results_df = pd.DataFrame({
        'Fold': [f'Fold {i}' for i in range(1, 6)],
        'MAE': -cv_results['test_mae'],
        'RMSE': -cv_results['test_rmse'],
        'R2': cv_results['test_r2']
    })

    print("\n--- Fold Bazlı Sonuçlar ---")
    print(cv_results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    cv_mae_mean = -cv_results['test_mae'].mean()
    cv_rmse_mean = -cv_results['test_rmse'].mean()
    cv_r2_mean = cv_results['test_r2'].mean()
    cv_mae_std = cv_results['test_mae'].std()
    cv_rmse_std = cv_results['test_rmse'].std()
    cv_r2_std = cv_results['test_r2'].std()

    print("\n--- Senaryo 2 Ortalama Sonuçları ---")
    print(f"Ortalama R-squared (R2): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"Ortalama MAE: {cv_mae_mean:.2f} ± {cv_mae_std:.2f}")
    print(f"Ortalama RMSE: {cv_rmse_mean:.2f} ± {cv_rmse_std:.2f}")

    scenario_comparison = pd.DataFrame({
        'Senaryo': ['Senaryo 1 - Train/Test Split', 'Senaryo 2 - 5-Fold CV Ortalama'],
        'R2': [r2, cv_r2_mean],
        'MAE': [mae, cv_mae_mean],
        'RMSE': [rmse, cv_rmse_mean]
    })

    print("\n--- Senaryo 1 ve Senaryo 2 Karşılaştırması ---")
    print(scenario_comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Modeli Kaydetme
    print("\n--- Modeli Kaydetme ---")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'final_rf_model_no_leakage.joblib')
    joblib.dump(rf, model_path)
    print(f"Sızıntısız model başarıyla kaydedildi: {model_path}")

if __name__ == "__main__":
    main()
