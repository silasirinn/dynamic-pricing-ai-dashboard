import pandas as pd

def load_data(file_path):
    """
    Verilen dosya yolundan CSV verisini okur ve sütun isimlerini snake_case formatına çevirir.
    Eğer sütun isimlerinde boşluk varsa, bu boşluklar alt çizgi (_) ile değiştirilir.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Sütun isimlerini küçük harfe çevir ve boşlukları alt çizgi ile değiştir
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        
        print("Veri başarıyla yüklendi. Sütun isimleri 'snake_case' formatına dönüştürüldü.")
        return df
    except FileNotFoundError:
        print(f"Hata: {file_path} dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return None
    except Exception as e:
        print(f"Veri yüklenirken beklenmeyen bir hata oluştu: {e}")
        return None
