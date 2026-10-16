# PaliGemma, sıfırdan

Google DeepMind'ın görsel-dil modeli PaliGemma'nın PyTorch ile sıfırdan yazılmış implementasyonu.

PaliGemma üç parçadan oluşur. SigLIP görüntüyü küçük yamalara bölüp her yamayı bir görsel token'a çevirir. Tek bir linear katman bu token'ları dil modelinin boyutuna taşır. Gemma da görsel token'ları ve metin komutunu birlikte okuyup cevabı token token yazar. Kutular, maskeler ve OCR çıktıları dahil her şey metin olarak üretilir.

Model eğitilmiyor. Mimari sıfırdan yazılıyor, Google'ın yayımladığı ağırlıklar bu modüllere yükleniyor ve her parçanın çıktısı Hugging Face `transformers` implementasyonuyla karşılaştırılıyor.

## Makale

Steiner, A., Pinto, A. S., Tschannen, M., et al. *PaliGemma 2: A Family of Versatile VLMs for Transfer.* Google DeepMind, 2024. [arXiv:2412.03555](https://arxiv.org/abs/2412.03555)

Makale aynı tarifle eğitilmiş 9 modeli karşılaştırıyor: 3 dil modeli boyutu (3B, 10B, 28B) ve 3 çözünürlük (224, 448, 896 piksel). Mimari PaliGemma 1 ile aynı, değişen tek şey dil modeli (Gemma → Gemma 2).

## Yazı

**[Göz mü, Beyin mi? Bir Görsel-Dil Modeline Nereden Yatırım Yapmalı?](https://medium.com/@mlihgenel/g%C3%B6z-m%C3%BC-beyin-mi-bir-g%C3%B6rsel-dil-modeline-nereden-yat%C4%B1r%C4%B1m-yapmal%C4%B1-342456927efa)** (Medium)

Makaleyi okumamış bir okur için yazılmış eleştirel bir inceleme. Bir görsel-dil modelini iyileştirmek için daha keskin bir göz mü (yüksek çözünürlük), yoksa daha büyük bir beyin mi (büyük dil modeli) gerektiğini soruyor. Metin ve doküman okuyan görevler çözünürlükten, çok dilli ve akıl yürütme gerektiren görevler büyük modelden kazanıyor. Yazı, makalenin hangi iddiaları rakamla desteklediğini, hangilerini desteklemediğini de tartışıyor.

## Deney

**[Türkçe tabelalar ve fişlerde 224 ile 448 piksel](https://github.com/mlihgenel/paper-to-practice/tree/main/papers/01-paligemma2/experiment)**

Makalenin çözünürlük iddiasını yeni bir veri üzerinde sınayan, ücretsiz Colab GPU'sunda tekrarlanabilir bir deney. Aynı 3B model, Türkçe tabela ve fiş fotoğraflarında iki çözünürlükte karşılaştırıldı:

- 224'ten 448'e geçince küçük yazıda doğruluk %13'ten %39'a, fişlerde toplam tutarda %46'dan %84'e çıkıyor. Bedeli soru başına yaklaşık 3,4 kat süre.
- 224 piksel, okuyamadığı metni uyduruyor: Galata kitabesindeki "29 Mayıs 1453"ü "20 Mayıs 1938" diye okuyor.
- Model Türkçe harfleri görüyor ama soru-cevapta siliyor: `ocr` modunda Türkçe harflerin %94'ünü doğru yazarken soru-cevapta "acil çıkış" yerine "acil cikis" yazıyor.

Bu repo, aynı modelin içini anlamak için mimariyi sıfırdan kuruyor.
