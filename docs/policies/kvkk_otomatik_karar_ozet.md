# KVKK ve Otomatik Kararlar — Kısa Özet (kendi ifademizle)

6698 sayılı Kişisel Verilerin Korunması Kanunu'nun kredi süreçlerine etkisinin platform açısından özeti. Hukuki görüş değildir.

## 1. Aydınlatma ve açık rıza

Başvuru sahibi, kişisel verilerinin hangi amaçla işleneceği, kimlere aktarılacağı ve hakları konusunda başvuru sırasında aydınlatılır. KKB sorgusu, e-Devlet verileri ve açık bankacılık hesap hareketleri gibi veri kaynaklarına erişim, ilgili açık rıza veya hukuki sebep kaydı olmadan yapılmaz. Platform her entegrasyon çağrısını geçerli ve süresi dolmamış bir rıza kaydına bağlar.

## 2. Madde 11: otomatik karara itiraz

İlgili kişi, işlenen verilerinin münhasıran otomatik sistemler vasıtasıyla analiz edilmesi suretiyle aleyhine bir sonucun ortaya çıkmasına itiraz etme hakkına sahiptir. Bu nedenle otomatik ret kararları gerekçe kodlarıyla bildirilir, bildirimde itiraz hakkı ve yolu açıkça belirtilir ve itiraz halinde karar bir insan tarafından yeniden incelenir.

## 3. Başvurulara yanıt süresi

Veri sorumlusu, ilgili kişinin başvurusunu niteliğine göre en kısa sürede ve en geç otuz gün içinde sonuçlandırmalıdır. Platformda itiraz incelemesi için 30 günlük SLA tanımlıdır.

## 4. Veri minimizasyonu ve güvenlik

Kişisel veriler işlendikleri amaçla bağlantılı, sınırlı ve ölçülü olmalıdır. Kimlik numarası, ad, telefon, adres ve IBAN uygulama seviyesinde şifrelenir; arama için anahtarlı özet (blind index) kullanılır. Yapay zekâ modellerine giden metinlerde kişisel veriler takma adlarla değiştirilir, loglarda maskelenir. Reddedilen veya iptal edilen başvuruların kişisel verileri saklama süresi sonunda anonimleştirilir.

## 5. Kurul kararları

Kişisel Verileri Koruma Kurulu'nun veri güvenliğine ve ihlal bildirimine ilişkin kararları (ör. 2019/10 sayılı yeterli önlemler kararı) ile yurt dışına aktarım düzenlemeleri, bulut ve yapay zekâ hizmetlerinin kullanımında dikkate alınmalıdır. LLM katmanının OpenAI uyumlu olması, modelin kurum içinde çalıştırılmasına imkân vererek yurt dışı aktarımı ihtiyacını ortadan kaldırabilir.
