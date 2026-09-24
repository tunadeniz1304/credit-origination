# İç Kredi Politikası — Bireysel Krediler (Örnek, Kurgusal)

Bu belge platformdaki karar kurallarının iş dilindeki karşılığıdır. Sayısal eşiklerin tek doğruluk kaynağı `rules/policy_v1.yaml` dosyasıdır; bu metin onları açıklar.

## 1. Kapsam ve ilkeler

Politika ihtiyaç ve taşıt kredilerini kapsar. Temel ilke, müşterinin geri ödeme gücünün belgelenmiş gelir ve mevcut borç yükü üzerinden ölçülmesidir. Kredi kararı; bağlayıcı olarak deterministik karar motoru ve yetkili kredi personeli tarafından verilir. Yapay zekâ bileşenleri yalnızca özetler, açıklar ve öneride bulunur.

## 2. Borç servis oranı (DSR)

Borç servis oranı, yeni kredinin taksiti dahil tüm aylık kredi ve kart ödemelerinin aylık net gelire bölünmesiyle hesaplanır. İhtiyaç kredisinde üst sınır yüzde 50, taşıt kredisinde yüzde 45'tir. Mevcut borç servisi tek başına sınırı aşıyorsa başvuru reddedilir. Yalnızca yeni taksit nedeniyle sınır aşılıyorsa, sınırı sağlayan daha düşük bir tutar karşı teklif olarak sunulabilir; karşı teklif talep edilen tutarın en az yüzde 30'u olmalıdır.

## 3. Kredi notu ve kredi geçmişi

KKB kredi notu 600'ün altındaki başvurular reddedilir. Yasal takipte kredisi bulunan başvurular reddedilir. Son 24 ayda gecikme kaydı bulunan başvurular model skoru ile birlikte değerlendirilir. Kredi geçmişi olmayan (ince dosya) başvuranlarda karar, açık bankacılık üzerinden elde edilen hesap hareketlerine dayanır; düzenli maaş girişi, düşük gelir değişkenliği ve pozitif tasarruf oranı olumlu değerlendirilir.

## 4. Temerrüt olasılığı eşikleri

On iki aylık temerrüt olasılığı (PD) yüzde 5 veya altındaysa ve hiçbir yönlendirme kuralı tetiklenmemişse başvuru otomatik onaylanır. PD yüzde 20 veya üzerindeyse otomatik ret uygulanır. Arada kalan gri bölgedeki başvurular uzman incelemesine yönlendirilir. PD bandı aynı zamanda limit çarpanını belirler: A bandı tam limit, B bandı yüzde 90, C bandı yüzde 75, D bandı yüzde 50.

## 5. Uzman incelemesine yönlendirme

Yaptırım veya PEP listesiyle olası eşleşme, belge sahteciliği şüphesi, beyan edilen gelir ile belge ve hesap hareketleri arasındaki uyumsuzluk, paylaşılan telefon/IBAN/adres ile oluşan başvuru halkası, son 30 günde çok sayıda başvuru, belgeden okunan alanların düşük güveni ve yüksek anomali skoru başvuruyu uzman incelemesine yönlendirir. Bu kurallar otomatik ret üretmez; insan kararını zorunlu kılar.

## 6. Yetki matrisi ve dört göz

250.000 TL'ye kadar ve PD yüzde 10'a kadar kararları krediler uzmanı verebilir. 750.000 TL'ye kadar ve PD yüzde 20'ye kadar kararlar kıdemli uzman yetkisindedir; daha büyük veya daha riskli kararlar kredi komitesine aittir. 300.000 TL üzerindeki, PD'si yüzde 12'nin üzerindeki ve motor kararını tersine çeviren her karar dört göz ilkesine tabidir: kararı öneren kişi onaylayamaz ve onaylayanın yetkisi yeterli olmalıdır. Her uzman kararında yazılı gerekçe zorunludur.

## 7. Fiyatlama

Faiz; temerrüt olasılığı, temerrüt halinde kayıp oranı ve kredi tutarından hesaplanan beklenen kayıp ile fonlama ve operasyon maliyetlerini karşılayacak ve hedef risk ayarlı sermaye getirisini (RAROC) sağlayacak şekilde belirlenir. Faiz, yasal azami akdi faiz oranını aşamaz; aşıyorsa başvuru bu gerekçeyle reddedilir. BSMV ve KKDF faiz üzerinden tahsil edilir ve taksite dahildir.

## 8. Başvurana bildirim ve itiraz

Otomatik ret kararları başvurana gerekçe kodlarıyla ve sade bir dille bildirilir. Bildirimde, sonucun değişmesini sağlayabilecek somut öneriler (karşı-olgusal açıklama) ve KVKK m.11 kapsamındaki itiraz hakkı ile itiraz yolu yer alır. İtirazlar bir kredi uzmanı tarafından insan incelemesiyle en geç 30 gün içinde sonuçlandırılır.
