# 🤖 AI Pricing Assistant (Dynamic Pricing Dashboard)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)

Profesyonel bir veri bilimi ve makine öğrenmesi (Machine Learning) projesi. Perakende satış verilerini kullanarak geliştirilen **Random Forest** tabanlı bu model, müşterilerin davranışlarına ve ürün özelliklerine göre optimum fiyat önerisi sunan akıllı bir karar destek ve dinamik fiyatlandırma (Dynamic Pricing) sistemidir.

## ✨ Öne Çıkan Özellikler

- **Gelişmiş AI Tahmini:** Seçilen yaş, cinsiyet, miktar ve kategori senaryosuna göre kârlılığı optimize eden en ideal fiyatı saniyeler içinde önerir.
- **Akıllı Risk Analizi:** Gerçek pazar verileriyle kıyaslama yaparak önerilen fiyatın uygulanabilirliği hakkında risk/fırsat durumunun değerlendirilmesini (satışa dönme ihtimalini) sunar.
- **Dinamik Duyarlılık (What-If) Simülasyonu:** Fiyatı manuel olarak değiştirdiğinizde tahmini talebin ve elde edilecek projeskiyon gelirinin nasıl etkileneceğini gösteren etkileşimli "fiyat-talep" modelleme eğrisi.
- **Modern ve Premium UI/UX:** Koyu tema (Dark Mode) üzerine kurulu, gradyan (soft purple & blue) animasyonlara sahip, fütüristik bir Streamlit arayüzü.
- **Model Analitiği:** Arka planda çalışan algoritmanın performans metriklerinin (Train/Test & 5-Fold Cross Validation), özellik önem derecelerinin (Feature Importance) ve veri seti özetinin (EDA) canlı olarak sergilenmesi.

## 🛠️ Teknoloji Yığını

- **Makine Öğrenmesi Modeli:** Scikit-Learn (Random Forest Regressor)
- **Veri İşleme ve Analizi:** Pandas, NumPy
- **Arayüz (Frontend):** Streamlit, Özel CSS & HTML Animasyonları (UI Injection)
- **Veri Görselleştirme:** Plotly Express & Graph Objects

## 🚀 Kurulum ve Çalıştırma

Projeyi bilgisayarınızda yerel olarak (localhost) çalıştırmak için aşağıdaki adımları izleyin:

1. **Projeyi Klonlayın:**
   ```bash
   git clone https://github.com/silasirinn/dynamic-pricing-ai-dashboard.git
   cd dynamic-pricing-ai-dashboard
   ```

2. **Gerekli Kütüphaneleri Yükleyin:**
   Bağımlılıkları kurmak için terminalinize aşağıdaki komutu yapıştırın:
   ```bash
   pip install -r requirements.txt
   ```

3. **Uygulamayı Başlatın:**
   ```bash
   streamlit run app.py
   ```

## 📂 Proje Yapısı

```text
├── data/                       # Ham ve işlenmiş veri setleri
├── notebooks/                  # EDA ve Model eğitim süreçlerini içeren Jupyter Notebook dosyaları
├── outputs/
│   ├── models/                 # Eğitilmiş makine öğrenmesi modelleri (.joblib)
├── src/                        # Veri yükleme, preprocessing, feature engineering scriptleri
├── app.py                      # Ana Streamlit Dashboard (Arayüz) kodları
├── requirements.txt            # Proje bağımlılıkları
└── README.md                   # Proje açıklaması
```

## 📊 Model Başarısı

Arka planda kullanılan Random Forest algoritması, detaylı özellik mühendisliği (Feature Engineering) adımlarından geçirilmiş ve test edilmiştir. Modeli değerlendirmek için iki aşamalı bir doğrulama yapılmıştır:

- **Hold-Out (Eğitim/Test Ayrımı):** Birincil performans kontrolü.
- **5-Katlı Çapraz Doğrulama (Cross Validation):** Modelin aşırı öğrenmeye (overfitting) düşmediğini ispatlayan tutarlılık testi.

## 🤝 İletişim

Geliştirici: **Sıla Şirin**  
Bu proje tamamen portfolyo ve eğitim amaçlı oluşturulmuştur. Herhangi bir sorunuz, öneriniz veya iş birliği fikriniz varsa benimle GitHub üzerinden iletişime geçebilirsiniz!
