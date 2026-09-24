# BDDK Bilgi Sistemleri Düzenlemesi ve Veri Yerelleştirme — Kısa Özet (kendi ifademizle)

Bankaların bilgi sistemleri ve elektronik bankacılık hizmetlerine ilişkin BDDK düzenlemesinin yapay zekâ destekli kredi platformu açısından önemli noktaları. Resmi metnin yerine geçmez.

## 1. Birincil ve ikincil sistemlerin yurt içinde tutulması

Bankalar birincil ve ikincil bilgi sistemlerini yurt içinde bulundurmakla yükümlüdür. Müşteri verilerinin işlendiği dış hizmetler bu çerçevede değerlendirilmelidir. Platformdaki LLM katmanı OpenAI uyumlu bir arayüz kullandığı için, tek bir ayar değişikliğiyle (`LLM_BASE_URL`) kurum içindeki bir modele (vLLM veya Ollama) yönlendirilebilir.

## 2. Dış hizmet alımı

Bankalar dış hizmet sağlayıcılarından aldıkları hizmetlerin risklerini değerlendirmeli, sözleşmelerle denetim hakkını güvence altına almalı ve hizmetin sürekliliğini planlamalıdır. Harici veri sağlayıcılarına yapılan çağrılar devre kesici ve yeniden deneme mekanizmalarıyla korunur; bir sağlayıcı erişilemez olduğunda başvuru veri toplama aşamasında bekletilir ve sonradan yeniden denenir.

## 3. Denetim izi ve erişim kontrolü

Kritik işlemler kimin, ne zaman, hangi yetkiyle yaptığını gösteren değiştirilemez kayıtlarla izlenmelidir. Rol bazlı erişim kontrolü, en az yetki ilkesi ve görevler ayrılığı uygulanmalıdır. Platformdaki hash zincirli denetim kaydı, JWT tabanlı rol yönetimi ve dört göz onayı bu gereksinimlere karşılık gelir.
