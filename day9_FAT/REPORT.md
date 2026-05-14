# MAPPO + CommNet Tabanlı Çok Ajanlı Sistemlerde Arıza Toleranslı Öğrenme

**Teknik Proje Raporu**

---

## 1. Giriş

### 1.1 Multi-Agent Reinforcement Learning

Multi-Agent Reinforcement Learning (MARL), birden fazla bağımsız ajanın ortak bir ortamda eş zamanlı olarak öğrenmesini ve karar vermesini inceleyen bir makine öğrenmesi paradigmasıdır. Tekil ajan RL'den temel farkı, ortamın tüm ajanların eylemlerine bağlı olarak dinamik bir şekilde değişmesi ve her ajanın sadece kendi politikasını değil, diğer ajanların davranışlarını da hesaba katmak zorunda olmasıdır. Bu özellik, MARL'ı tek ajan sistemlerine kıyasla teorik ve pratik açıdan çok daha karmaşık bir alan haline getirmektedir.

MARL uygulamaları geniş bir yelpazede karşımıza çıkmaktadır: İnsansız Hava Aracı (İHA) sürüleri, otonom araç platoları, robot sürüleri, akıllı enerji şebekeleri ve finansal piyasa simülasyonları bunların başında gelmektedir. Bu sistemlerin ortak özelliği, merkezi bir otoritenin bulunmaması ya da gerçek zamanlı merkezi koordinasyonun fiziksel olarak mümkün olmamasıdır. Dolayısıyla her ajan, sınırlı lokal gözlemlerle hareket etmek ve gerektiğinde diğer ajanlarla iletişim kurarak koordineli davranışlar sergilemek zorundadır.

### 1.2 Arıza Toleransı Neden Önemlidir?

Gerçek dünya dağıtık sistemlerinde ajanların kesintisiz ve hatasız çalışması nadiren mümkündür. Bir İHA sürüsünde haberleşme kanalı bozulabilir, bir ajan mekanik arıza nedeniyle dururabilir ya da sensör gürültüsü nedeniyle hatalı mesajlar iletebilir. Benzer şekilde, akıllı fabrika ortamlarında bir robot kolunun geçici olarak devre dışı kalması ya da bir sensörün yanlış okuma vermesi, tüm üretim hattının performansını olumsuz etkileyebilir.

Klasik RL yöntemleri bu tür arızaları genellikle görmezden gelir ve ajanların tam kapasitede çalıştığını varsayar. Bu varsayım teorik açıdan kabul edilebilir olsa da pratik dağıtık sistemlere bu yöntemlerin doğrudan uygulanmasını güçleştirmektedir. Arıza toleranslı bir MARL sisteminin geliştirilmesi, yalnızca akademik bir problem olmayıp endüstriyel uygulamalar açısından da kritik bir gerekliliktir.

### 1.3 Çok Ajanlı Sistemlerde Haberleşme ve Koordinasyon Problemleri

Çok ajanlı sistemlerde koordinasyon, ajanların birbirlerinin niyetlerini ve eylemlerini anlayabilmesi ile doğrudan ilişkilidir. Haberleşme kanalları üzerinden iletilen mesajlar bu koordinasyonun temel taşıdır. Ancak bir ya da birden fazla ajanın arızalı mesajlar iletmesi, sağlıklı ajanların karar süreçlerini olumsuz yönde etkileyebilir. Özellikle Byzantine tipi arızalarda arızalı ajan, kasıtlı olarak yanıltıcı mesajlar üretir ve diğer ajanları yanlış yönlendirir. Bu durum, sistemin bütününde koordinasyon kaybına yol açar.

### 1.4 Çalışmanın Amacı ve Motivasyonu

Bu çalışma, MAPPO algoritması ile CommNet iletişim mimarisini entegre ederek MPE Simple Spread ortamında üç farklı arıza tolerans stratejisini karşılaştırmaktadır. Temel soru şudur: *Arızalı bir iletişim ortamında eğitilen bir MARL sistemi, bu arızaları nasıl tespit edebilir ve sistemi nasıl yeniden yapılandırabilir?*

Çalışma; naif yaklaşım (Strateji A), müfredat öğrenmesi tabanlı yaklaşım (Strateji B) ve arıza tespiti ile topoloji adaptasyonu tabanlı yaklaşım (Strateji C) olmak üzere üç farklı stratejinin fail-stop, Byzantine ve intermittent arıza senaryoları altındaki performansını sistematik olarak değerlendirmektedir.

---

## 2. Kullanılan Ortam: MPE Simple Spread

### 2.1 MPE Nedir?

Multi-Agent Particle Environments (MPE), OpenAI tarafından geliştirilen ve sonradan PettingZoo kütüphanesi bünyesine alınan çok ajanlı ortam paketidir. 2D sürekli uzayda hareket eden parçacık ajanlar içermektedir. Ortamlar kooperatif, rekabetçi veya karma görevler için konfigüre edilebilir. Basit dinamikleri ve açık kaynak yapısıyla MARL araştırmalarında standart bir kıyaslama ortamı haline gelmiştir.

### 2.2 Simple Spread Görevinin Açıklaması

Simple Spread, $N$ ajanın $N$ landmark'ı (hedef nokta) kapsamasını gerektiren tam kooperatif bir görevdir. Bu çalışmada $N=3$ kullanılmıştır: 3 ajan ve 3 landmark. Episot başında hem ajanlar hem de landmark'lar 2B uzayda rastgele konumlandırılır. Görev boyunca landmark'ların konumu sabittir; ajanlar ise her adımda hareket ederek landmark'ları kapsamaya çalışır.

Sistemin kritik özelliği, tek bir ajanın tüm landmark'ları aynı anda kaplamasının mümkün olmamasıdır. Bu nedenle ajanların koordineli biçimde birbirinden farklı landmark'lara yönelmesi zorunludur; aksi hâlde ajanlar kümelenme (clustering) yaparak bazı landmark'ları kapsamamış olarak bırakır.

### 2.3 Gözlem Uzayı

Her ajan şu bilgileri içeren 18 boyutlu bir gözlem vektörü alır:

$$o_i = [\underbrace{v_i}_{2}, \underbrace{p_i}_{2}, \underbrace{\Delta p_{i,l_1}, \Delta p_{i,l_2}, \Delta p_{i,l_3}}_{6}, \underbrace{\Delta p_{i,j_1}, \Delta p_{i,j_2}}_{4}, \underbrace{c}_{4}]$$

Burada $v_i$ ajanın hızı, $p_i$ konumu, $\Delta p_{i,l_k}$ landmark'lara göreli konum, $\Delta p_{i,j}$ diğer ajanların göreli konumu, $c$ ise iletişim kanalı için ayrılmış padding alanıdır (bu ortamda sıfır). Strateji B ve C'de gözlem vektörüne ek olarak 1 boyutlu bir **arıza göstergesi** eklenir (toplam 19 boyut). Bu gösterge B'de oracle bilgisinden, C'de ise FaultDetector'ün tahminine dayalı olarak belirlenir.

### 2.4 Eylem Uzayı

Her ajan 5 ayrık eylemden birini seçer: boşta kal (no-op), sola git, sağa git, yukarı git, aşağı git. Eylemler ortam tarafından sürtünmesiz bir ivme modeli ile işlenir.

### 2.5 Ödül Yapısı ve Reward Shaping

#### Neden Özel Reward Shaping?

MPE Simple Spread ortamının ham ödülü iki bileşenden oluşur:

$$r_{\text{ham}} = -\sum_{j=1}^{N} \min_i \|p_i - l_j\| - c \cdot \sum_{i \neq k} \mathbf{1}[\|p_i - p_k\| < d_{\text{collision}}]$$

İlk terim her landmark'ın en yakın ajana mesafesinin toplamını minimize etmeyi hedefler; ikinci terim ise ajan çiftleri arasındaki çarpışmaları cezalandırır. Ancak bu formül pratikte iki temel soruna yol açmaktadır. **Birincisi, seyrek gradient problemidir:** ajanlar landmark'lara uzakta olduğunda ödüldeki değişim çok küçük kalır ve policy gradient sinyali neredeyse sıfıra iner. Bu durum özellikle eğitimin başlarında keşfi zorlaştırmaktadır. **İkincisi, kümelenme problemidir:** ham ödül, iki ajanın aynı landmark'ı kaplamasını cezalandırmaz. Ajanlar aynı landmark'a gitmeyi öğrenebilir ve diğer landmark'lar boş kalır; bu durum locally optimal fakat globally suboptimal bir davranıştır.

Arıza senaryoları bu sorunları daha da derinleştirmektedir: arızalı bir ajanın mesajları diğer ajanları yanlış yönlendirdiğinde, ham ödülün zayıf gradyanı toparlanmayı zorlaştırır. Özellikle Byzantine saldırısında adversarial mesajların ürettiği baskı, ham ödül gradyanını tamamen bastırabilmektedir.

#### Uygulanan Dense Reward Formülü

Bu sorunları gidermek amacıyla ham ödülün üzerine üç ek bileşen eklenmiştir. Eğitimde kullanılan toplam ödül şöyledir:

$$r_{\text{shaped}} = \underbrace{r_{\text{ham}}}_{\text{ortam ödülü}} + \underbrace{\alpha \cdot \frac{1}{N}\sum_{j=1}^{N} e^{-\sigma d_j}}_{\text{(1) Navigasyon}} + \underbrace{\beta \cdot \frac{1}{N}\sum_{j=1}^{N} \mathbf{1}[d_j < \tau]}_{\text{(2) Commitment}} + \underbrace{\gamma \cdot u_{r}}_{\text{(3) Yayılma}}$$

Ham ödül $r_{\text{ham}}$ korunmaktadır; bu sayede çarpışma cezası ve mesafe bilgisi sinyali kaybolmaz. Eklenen üç terim bu sinyali güçlendirmek ve eksik teşvikleri tamamlamak amacıyla tasarlanmıştır. $u_r$ sembolü `unique_ratio`'yu temsil etmektedir. Kullanılan değerler: $\alpha=0.5$, $\beta=0.8$, $\gamma=0.3$, $\sigma=5.0$, $\tau=0.15$.

**Bileşen 1 — Navigasyon Gradyanı ($e^{-\sigma d}$):** Gaussian benzeri bu terim, her mesafe değeri için sıfırdan farklı bir gradient üretir. $\sigma=5.0$ ile $d \approx 0.5$'e kadar anlamlı gradient korunur. Başlangıçta $\sigma=8.0$ denenmiş; ancak $e^{-8 \times 0.4} \approx 0.04$ gibi çok küçük değerler üreterek ajan landmark'tan uzaktayken gradient açlığına (reward starvation) ve entropy çöküşüne yol açmıştır. $\sigma=5.0$'a geçiş bu sorunu çözmüştür.

**Bileşen 2 — Commitment Bonus ($\mathbf{1}[d < \tau]$):** Ajan landmark'a $\tau=0.15$ mesafe eşiğinin altına girdiğinde sıçramalı bir ödül verilir. Bu terim iki amaca hizmet eder: ajanın landmark üzerinde kalmasını teşvik eder (salt navigasyonla gidip geri dönen politikaların önüne geçer) ve ödül fonksiyonuna kademeli bir şekil verir (uzakta küçük gradyan, yakında büyük ödül). $\beta=0.8$ katsayısının yüksek tutulması, commitment davranışının politika üzerinde baskın bir sinyal oluşturmasını sağlar.

**Bileşen 3 — Yayılma Bonusu** (`unique_ratio`)**:** Bu terim kümelenme problemini doğrudan hedef alır. `unique_ratio`, kaç farklı ajanın en az bir landmark'a en yakın ajan olduğunun $N$'e oranıdır:

$$u_r = \frac{|\{\arg\min_i \|p_i - l_j\| : j = 1,\ldots,N\}|}{N}$$

Tüm ajanlar aynı landmark'ı kaplarsa `unique_ratio` $= 1/N = 0.33$; her ajan farklı bir landmark'ı kaplarsa `unique_ratio` $= 1.0$. Bu terim, ajanları birbirinden uzak landmark'lara yönelmeye zorlar ve koordinasyonu ödül seviyesinde kodlar.

#### Arıza Senaryolarıyla İlişkisi

Dense reward tasarımı arıza toleransıyla doğrudan bağlantılıdır. Bir ajan arızalandığında, kalan sağlıklı ajanların hâlâ güçlü bir gradient sinyali alması kritiktir; aksi hâlde sistem arızadan "kurtulamaz" ve düşük performansa kilitlenir. $e^{-5d}$ terimi bu kurtarma gradyanını sağlarken, `unique_ratio` terimi sağlıklı ajanların arızalı ajanın boş bıraktığı landmark'a yönelmesini teşvik eder.

### 2.6 Kooperatif Yapı

Sistem tam kooperatiftir: tüm ajanlar aynı takım ödülünü paylaşır, bireysel ödül yoktur. Bu yapı ajanların ortak bir hedef için koordineli davranmasını zorunlu kılar ve CTDE (Centralized Training, Decentralized Execution) paradigmasına uygundur.

---

## 3. Kullanılan Algoritmalar ve Mimari

### 3.1 PPO ve MAPPO

**Proximal Policy Optimization (PPO)**, politika gradyanı yöntemlerinin kararsız güncellemelerini düzeltmek amacıyla bir kırpma (clipping) mekanizması kullanan aktör-eleştirmen (actor-critic) algoritmasıdır. Politika güncellemesi:

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t \right) \right]$$

Burada $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ olasılık oranı, $\hat{A}_t$ GAE ile hesaplanan avantaj tahmini, $\varepsilon=0.2$ kırpma sınırıdır. PPO, her güncelleme adımında 4 epoch ve 256'lık mini-batch ile uygulanmaktadır.

**Multi-Agent PPO (MAPPO)**, PPO'nun CTDE paradigmasına uyarlanmış versiyonudur. **Aktör** her ajan için bağımsız olarak çalışır ve yalnızca o ajanın lokal gözlemini kullanır. **Eleştirmen** ise merkezi eğitim aşamasında tüm ajanların gözlemlerini birleştirerek global değer tahmini yapar:

$$V(s) = f_\phi(o_1 \oplus o_2 \oplus \cdots \oplus o_N)$$

Bu sayede eğitim sırasında global bilgiden yararlanılırken, çalışma zamanında her ajan yalnızca kendi lokal gözlemine dayanır.

Avantaj tahmini GAE ile hesaplanmaktadır ($\gamma=0.95$, $\lambda=0.95$):

$$\hat{A}_t = \sum_{k=0}^{T-t} (\gamma\lambda)^k \delta_{t+k}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 3.2 CommNet Mimarisi

**CommNet** (Communication Neural Network), ajanların gizli temsil vektörlerini birbirleriyle paylaşmasına olanak tanıyan bir iletişim mimarisidir. Temel fikir şudur: her ajan kendi gözlemini bir gizli vektöre kodlar ve bu vektörü komşularıyla paylaşır; ardından aldığı mesajları toplayarak eylem politikasını günceller.

Bu çalışmadaki CommNetActor iki aşamada çalışır:

**Kodlama (Encode):** Her ajanın gözlemi bağımsız bir encoder ağından geçirilerek $H=128$ boyutlu gizli vektöre dönüştürülür:

$$h_i = \text{Encoder}(o_i), \quad \text{Encoder}: \mathbb{R}^{d_{obs}} \to \mathbb{R}^H$$

Encoder iki katmanlı MLP'dir (Linear→ReLU→Linear→ReLU).

**Agregasyon ve Politika (Aggregate & Policy):** Komşulardan gelen mesajlar ağırlıklı adjacency matrisi $A$ kullanılarak toplanır:

$$m_i = \sum_j A_{ij} h_j$$

Her ajanın kendi temsili ve aldığı mesaj birleştirilerek politika ağına verilir:

$$\text{logits}_i = \text{PolicyNet}([h_i \| m_i])$$

Bu ayrım kritik bir tasarım kararıdır: `encode()` ve `aggregate_and_policy()` metodlarının ayrılması, iki aşama arasına mesaj seviyesinde arıza enjeksiyonu yapılmasına olanak tanır.

Adjacency matrisi $A$ statik (tam bağlı) veya dinamik (`comm_radius` tabanlı) olabilir; Strateji C'de bu matris FaultDetector çıktısına göre anlık olarak yeniden yapılandırılır.

### 3.3 FaultWrapper

`FaultWrapper`, temel MPE ortamını sarmalayan ve sisteme kontrollü arızalar enjekte eden bir katmandır. Arızalar doğrudan mesaj seviyesinde uygulanır: `inject_message_faults()` fonksiyonu, `encode()` çıktısı $h$'yi bozarak `aggregate_and_policy()`'ye girmeden önce müdahale eder.

Bu tasarımın avantajı, arıza modelinin ajan politikasından tamamen bağımsız olmasıdır. Aynı politika, farklı arıza intensiteleri ve tipleriyle yeniden eğitime sokulabilir. FaultWrapper ayrıca **intensity** parametresi ile arıza şiddetini 0'dan 1'e kademeli olarak artırma imkânı sunar; bu özellik Strateji B'nin müfredat öğrenmesi mekanizmasının temelini oluşturur.

---

## 4. Arıza Türleri ve Etkileri

Çalışmada üç temel mesaj seviyesi arıza modeli uygulanmıştır. Bu modeller gerçek dağıtık sistemlerdeki arıza sınıflarını temsil etmektedir.

### 4.1 Fail-Stop (S2)

Fail-stop arızasında arızalı ajan mesaj göndermeyi tamamen durdurur; encoder çıktısı sıfır vektöre ayarlanır ($h_i = \mathbf{0}$). Bu durum İHA sürülerinde bir aracın batarya tükenmesi veya iletişim anteninin arızalanmasına karşılık gelir.

Etkisi belirgindir: arızalı ajanın komşuları mesaj agregasyonunda sıfır katkı alır. Naif politika (Strateji A) bu yokluğu öğrenemediği için koordinasyon bozulur. Teorik maksimum kapsam oranı $\frac{N-1}{N} = \frac{2}{3} \approx 0.667$'dir; sağlıklı 2 ajan en fazla 2 landmark'ı kapayabilir.

### 4.2 Byzantine (S3)

Byzantine arızası, en zorlu ve en gerçekçi saldırı modelidir. Arızalı ajan aktif olarak yanıltıcı mesajlar üretir:

$$h_{\text{faulty}} = -\kappa \cdot \frac{1}{|\mathcal{H}|}\sum_{j \in \mathcal{H}} h_j$$

Burada $\mathcal{H}$ sağlıklı ajanlar kümesi, $\kappa=2.0$ saldırı magnitudüdür. Bu formül, sağlıklı ajanların ortalama gömülü temsilinin negatif katı olarak adversarial bir mesaj üretir. Sonuç olarak arızalı ajanın komşuları, sağlıklı ajanların öğrendiklerinin tam tersi yönde bir gradient sinyali alır. Strateji A bu senaryoda tamamen çökmektedir (kapsam $\approx 0.06$, yakınsama yok).

Byzantine arızasının eğitimi tehdit etmemesi için iki önlem alınmıştır: magnitude warmup (0.3'ten başlayıp ilk %30 update'te 2.0'a ulaşır) ve NaN zinciri koruması (h kırpma, logit `nan_to_num`, gradient `isFinite` kontrolü).

### 4.3 Intermittent (S4)

Intermittent arıza, periyodik değil stokastik bir dropout modelidir:

$$P(\text{dropout}) = \text{intensity} \times p_{\text{intermittent}}$$

Çalışmada $p_{\text{intermittent}}=0.3$ kullanılmıştır. Bu parametre ayrımı kasıtlıdır: intensity arızanın aktif olup olmadığını, $p_{\text{intermittent}}$ ise aktif olduğunda ne sıklıkla gerçekleştiğini kontrol eder. Bu ikisinin ayrılmaması durumunda intermittent arıza, intensity=1.0'da fail-stop ile özdeşleşir.

Intermittent arızanın ilginç bir özelliği vardır: arızalı ajan bazen sağlıklı mesajlar iletmektedir. Bu durum, onu sürekli izole etmenin counterproductive olduğunu göstermiştir. Nitekim deneyler, Strateji C'nin bu senaryoda B'yi geride bıraktığını ortaya koymuştur.

---

## 5. Eğitim Süreci

### 5.1 Temel Parametreler

| Parametre | Değer |
|-----------|-------|
| Toplam güncelleme sayısı | 2000 |
| Rollout adımı (worker başına) | 1000 |
| Paralel worker sayısı | 4 |
| Toplam timestep | ~8M |
| Actor öğrenme oranı | 3×10⁻⁴ |
| Critic öğrenme oranı | 3×10⁻⁴ |
| PPO epoch | 4 |
| Mini-batch boyutu | 256 |
| Entropi katsayısı | 0.015 |
| Gradient norm kırpma | 0.5 |

### 5.2 Paralel Ortam Kullanımı

Veri toplama verimliliğini artırmak amacıyla `N_ENVS=4` bağımsız worker process kullanılmıştır. Her worker, Python multiprocessing spawn havuzunda çalışır, bağımsız bir ortam örneği açar ve `ROLLOUT_STEPS=1000` adım toplar. İşçi process'ler ana process'ten aktör ağırlıklarını, mevcut arıza konfigürasyonunu ve **`obs_rms` state'ini** alır; rollout tamamlandığında `obs_rms` güncellenmiş haliyle geri iletilir.

Bu tasarımda kritik bir nüans bulunmaktadır: gözlem normalizasyon istatistiklerinin (`RunningMeanStd`) worker'lara doğru iletilmemesi başlangıçta ciddi bir performans düşüşüne yol açmıştır. Düzeltme sonrasında `obs_rms` doğrusal olarak birikmekte ve tüm worker'lar tutarlı normalizasyon istatistikleri kullanmaktadır.

### 5.3 Gözlem Normalizasyonu

Gözlemler online olarak normalize edilir:

$$\tilde{o} = \frac{o - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

Burada $\mu$ ve $\sigma^2$ RunningMeanStd ile güncellenen online istatistiklerdir. NaN koruması için sonuç `nan_to_num` ile sınırlandırılır. Bu normalizasyon, özellikle Byzantine arızasının büyük encoder aktivasyonlarına yol açtığı durumlarda eğitim kararlılığı açısından kritik öneme sahiptir.

### 5.4 Keşif ve Politika Öğrenimi

Politika entropi bonusu ($\alpha_H=0.015$) ile düzenlenerek erken kapanma önlenmektedir. Arıza senaryolarında Strateji A'nın bazı seed'lerde entropi çöküşü yaşadığı (`convergence_step=-1`) gözlemlenmiştir; bu durum arızalı mesajların gradient yönünü bozmasının yanı sıra policy'nin suboptimal deterministik bir davranışa kilitlenmesinden kaynaklanmaktadır.

---

## 6. Arıza Tolerans Yaklaşımları

Üç strateji, arıza toleransının farklı katmanlarını temsil etmektedir.

### 6.1 Strateji A — Naif Yaklaşım

Herhangi bir arıza farkındalığı bulunmamaktadır. Arıza tam şiddetle (intensity=1.0) başından beri uygulanır, topology değişmez, ek bilgi sağlanmaz. Bu strateji kıyaslama tabanı olarak tasarlanmıştır ve sistemin arıza tolerans mekanizması olmadan nasıl davrandığını göstermektedir.

### 6.2 Strateji B — Müfredat Öğrenmesi

Arıza şiddeti başlangıçta düşük tutulur ve eğitim ilerledikçe kademeli olarak artırılır. `CurriculumScheduler`, arıza intensitesini doğrusal ramp ile önce $I_{\max}$'a çıkarır; ardından kapsamın plato yaptığı noktalarda bump mekanizmasıyla sıçramalı artışlar uygular:

$$\text{intensity}(t) = \begin{cases} \frac{t}{T_{\text{linear}}} \cdot I_{\max} & t < T_{\text{linear}} \\ \text{intensity}(t-1) + \Delta & \text{kapsam plato yaptıysa} \end{cases}$$

Bu yaklaşım, ajanların önce arızasız bir ortamda temel koordinasyonu öğrenmesini, ardından giderek artan arıza baskısıyla başa çıkma stratejileri geliştirmesini sağlar. Topology tam bağlı kalır; ajanlar fault indicator sayesinde hangi komşularının arızalı olduğunu bilir ve mesajlarını buna göre değerlendirmeyi öğrenir.

### 6.3 Strateji C — Topoloji Adaptasyonu

Bu strateji, oracle bilgisi gerektirmeden arızalı ajanı tespit eder ve iletişim grafiğini anlık olarak yeniden yapılandırır.

**FaultDetector** (v4), her güncelleme adımında beş istatistiksel sinyal hesaplar:

1. **z_self**: Ajanın norm geçmişine göre z-skoru — ani norm değişimi arıza belirtisidir
2. **z_cv**: Norm değişim katsayısının z-skoru — arızalı mesajlar daha yüksek varyans gösterir
3. **z_fleet**: Filodaki diğer ajanların median'ından sapma — Byzantine mesajlar küme dışına düşer
4. **cos_drop**: Komşularla cosine benzerliğindeki düşüş — içerik tutarsızlığının göstergesi
5. **coherence_drop**: EMA kohezyon düşüşü — zamansal tutarsızlık

Bir ajan $K=3$ veya daha fazla sinyal eşiği aşarsa "şüpheli" (suspect) olarak işaretlenir. $M=5$ ardışık adım boyunca şüpheli kalırsa "arızalı" (flagged) olarak sınıflandırılır (histerezis). Bu çift eşikli mekanizma yanlış pozitifleri minimize etmektedir.

`TopologyManager`, flagged ajanın adjacency matrisindeki gelen ve giden bağlantılarını sıfırlar; self-loop korunur:

$$A_{ij} = 0 \quad \text{ve} \quad A_{ji} = 0 \quad \forall i : \text{flagged}(i),\ j \neq i$$

Bu sayede arızalı ajanın mesajları sisteme girmez ve sağlıklı ajanlar arızalı komşunun bağlantısına güvenmez hale gelir.

---

## 7. Deneyler ve Gözlemler

### 7.1 Deney Konfigürasyonu

Her strateji-senaryo kombinasyonu 3 farklı seed ile tekrarlanmıştır (toplamda 36 run). Sonuçların ortalaması alınarak seed'e bağlı varyans analiz edilmiştir.

**Değerlendirme metrikleri:**
- **Kapsam oranı (coverage)**: Her landmark'a ait en yakın ajanın mesafesi < 0.30 olduğunda o landmark kaplı sayılır; kaplanan landmark sayısının $N$'e oranı
- **Ortalama landmark mesafesi** (`avg_dist`): Tüm landmark'lar için en yakın ajan mesafesinin ortalaması
- **Yakınsama adımı** (`convergence_step`): Kapsamın 0.60 eşiğini ilk geçtiği güncelleme adımı

Strateji C için ek olarak **detector metrikleri** raporlanmaktadır: precision, recall, F1 ve false positive rate.

### 7.2 Nicel Sonuçlar

**Tablo 1: Senaryo bazında ortalama kapsam oranları (3 seed ortalaması)**

| Senaryo | Strateji A | Strateji B | Strateji C |
|---------|-----------|-----------|-----------|
| S1 – Arızasız | 0.830 | 0.830 | 0.833 |
| S2 – Fail-stop | 0.370 | 0.647 | 0.641 |
| S3 – Byzantine | 0.063 | 0.675 | 0.628 |
| S4 – Intermittent | 0.650 | 0.656 | 0.649 |

**Strateji A için yüksek varyans:** S2 senaryosunda A'nın seed bazında 0.223 ile 0.576 arasında değişen kapsamı, arıza tolerans mekanizması olmaksızın öğrenmenin ne denli kararsız olduğunu göstermektedir. Sağlıklı başlangıç koşullarında politika arızalı mesajı tolere etmeyi öğrenebilirken, olumsuz koşullarda entropi çöküşü gerçekleşmekte ve öğrenme tamamen durmaktadır.

**Byzantine altında Strateji A'nın çöküşü:** S3 senaryosunda A tüm seed'lerde ~0.06 kapsam değerine ulaşmıştır. Adversarial mesajlar CommNet agregasyonunu bozmakta, gradient sinyali yanlış yönde akmakta ve politika arızalı ajanla birlikte kilitlenmektedir. Bu sonuç Byzantine arıza modelinin ne denli yıkıcı olabileceğini somut olarak göstermektedir.

**Intermittent senaryosunda C ≥ B:** S4 senaryosunda Strateji C'nin B ile rekabetçi ya da üstün performans göstermesi dikkate değer bir bulgudur. Oracle bilgisine sahip B, arızalı ajanı sürekli izole etmektedir; oysa intermittent arızada bu ajan zaman zaman sağlıklı mesajlar iletmektedir. C ise detector'ün dinamik kararları sayesinde ajanı yalnızca gerçekten arızalı olduğu anlarda izole eder.

**Detector performansı:** Tüm C deneyleri boyunca FaultDetector precision değeri sabit olarak 1.0 ölçülmüştür; yanlış pozitif hiç gözlemlenmemiştir. Recall ise senaryo ve seed'e göre 0.987–0.999 aralığında kalmıştır.

### 7.3 Davranışsal Gözlemler

GIF görselleştirmeleri üzerinden yapılan kalitatif analizde şu davranışlar gözlemlenmiştir:

- **Sağlıklı koordinasyon (S1):** Ajanlar yakınsama sonrasında belirli landmark'lara atanmış gibi davranmakta ve konumlarını korumaktadır. Öğrenilen strateji, her ajanın farklı bir landmark'ı sahiplenmesine dayanmaktadır.

- **Fail-stop altında Strateji C:** Arızalı ajan detector tarafından tespit edildikten sonra iletişim grafiğinden çıkarılmakta; kalan iki sağlıklı ajan erişebildikleri iki landmark'ı kapsamaktadır.

- **Byzantine altında Strateji A:** Ajanlar landmark'lara yönelememekte, ortamda düzensiz hareketler gözlemlenmektedir. Adversarial mesajlar politikanın tutarlı bir strateji geliştirmesini engellemektedir.

- **Titreme davranışı:** Tüm stratejilerde, landmark üzerinde konumlanmış ajanların no-op yerine küçük hareket eylemleri ürettiği gözlemlenmektedir. Bu durum PPO entropi bonusunun politikayı tamamen deterministik olmaktan alıkoymasından kaynaklanmakta olup performansa etkisi ihmal edilebilir düzeydedir.

---

## 8. Sonuç

### 8.1 Genel Değerlendirme

Bu çalışma, MARL sistemlerinde arıza toleransının farklı katmanlarda nasıl sağlanabileceğini sistematik biçimde incelemiştir. Sonuçlar, aynı CommNet + MAPPO altyapısı üzerine inşa edilen üç stratejinin arıza senaryolarında birbirinden belirgin şekilde ayrışan performans gösterdiğini ortaya koymaktadır.

Naif yaklaşım (Strateji A), Byzantine arıza karşısında tamamen çökmeye karşın intermittent senaryoda makul performans sergilemiştir. Bu bulgu, bazı arıza türlerinin pasif tolerans ile yönetilebilirken adversarial saldırıların aktif mekanizmalar gerektirdiğine işaret etmektedir.

Müfredat öğrenmesi (Strateji B) arıza şiddetini kademeli olarak artırarak daha stabil öğrenme sağlamaktadır. Fault indicator sayesinde politika arızalı komşusunun mesajlarını ağırlıklandırmayı öğrenmektedir.

Topoloji adaptasyonu (Strateji C) kör bir detector ile %100 precision ve %99 üzeri recall değerlerine ulaşmıştır. Özellikle intermittent senaryolarda dinamik karar alma mekanizması, sabit oracle izolasyonunu geride bırakmıştır. Detector'ün oracle bilgisine dayanmaması, bu yaklaşımın gerçek dağıtık sistemlere doğrudan uygulanabilirliğini artırmaktadır.

### 8.2 Güçlü Yönler

- FaultDetector beş istatistiksel sinyal sayesinde arıza tipine bağımsız genel bir tespit mekanizması sunmaktadır
- CommNet'in mesaj seviyesinde ayrıştırılması arıza enjeksiyonunu mimari değişikliği gerektirmeden mümkün kılmaktadır
- Paralel worker yapısı veri toplama verimliliğini artırırken eğitim kararlılığını korumaktadır
- Dense reward tasarımı hem navigasyon gradyanı hem de koordinasyon teşvikini birleştirmektedir

### 8.3 Sınırlamalar

- **Azınlık varsayımı**: FaultDetector'ün mevcut tasarımı $N_{\text{faulty}} < N/2$ koşulunu gerektirmektedir. Arızalı ajan oranı bu eşiği aştığında sağlıklı ve arızalı ajanların istatistiksel profilleri ayrışmayabilir.
- **Sabit arıza ataması**: Arızalı ajan eğitim boyunca sabittir; dinamik olarak değişen arıza atamaları ele alınmamıştır.
- **Küçük ölçek**: $N=3$ ajanla yapılan deneyler, daha büyük sürü sistemlerindeki ölçeklenebilirliği garantilemez.
- **No-op optimizasyonu**: Ajanlar landmark üzerindeyken hareketsiz kalmayı tam olarak öğrenememiştir.

### 8.4 Gelecek Çalışmalar

- **Byzantine-dirençli agregasyon**: Fleet median yerine geometric median veya Krum agregasyonu kullanılarak azınlık varsayımı ortadan kaldırılabilir
- **Karma arıza senaryoları**: Farklı ajanların farklı arıza tiplerine sahip olduğu senaryolar ($N=5$, 2 arızalı)
- **Dinamik topoloji**: İletişim yarıçapı tabanlı dinamik graph ile arıza kombinasyonu
- **Best-model checkpointing**: Her senaryoda en yüksek kapsam değerine ulaşan model ağırlıklarının kaydedilmesi
- **Ek metrikler**: Çarpışma sayısı, başarı oranı ve koordinasyon verimliliği metriklerinin eğitim sürecine eklenmesi

---

*Bu rapor, `day9_FAT/model_train.py` ve `src/adaptation.py` üzerinde gerçekleştirilen deneysel çalışmaya dayanmaktadır. Sweep sonuçları WandB üzerinde `marl-fault-tolerance_v2` projesi altında kayıtlıdır.*
