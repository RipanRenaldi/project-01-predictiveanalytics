# Laporan Proyek *Machine Learning Laptop Prices* - Ripan Renaldi

## Domain Proyek  
Perkembangan teknologi pada revolusi industri 4.0 ini mengakibatkan banyak orang memanfaatkan teknologi informasi yang ada saat ini. Teknologi tersebut tidak lagi sulit untuk didapatkan, mengingat dulu tidaklah mudah untuk mendapatkan teknologi komputer yang ada seperti sekarang ini. Manfaat komputer atau laptop ini sangatlah banyak, salah satunya dalam pencarian informasi yang tersedia pada internet. Jika dilihat dari sejarah perkembangan komputer, kini komputer memiliki performa yang jauh lebih cepat dibandingkan dengan dulu. Canggihnya lagi, kini komputer memiliki ukuran yang jauh lebih kecil dan dapat kita bawa kemana-mana. Dengan ukuran yang kecil itulah laptop yang ada saat ini lebih praktis untuk dapat digunakan dimanapun dan kapanpun.

Setiap laptop yang ada, memiliki merk dan spesifikasi yang berbeda-beda, spesifikasi tersebut berpengaruh terhadap performa atau kinerja laptop saat digunakan. Tentunya, dengan laptop yang memiliki spesifikasi tinggi akan mempengaruhi performa laptop saat digunakan menjadi lebih cepat. Kini, penggunaan laptop tidak hanya untuk pencarian informasi saja, laptop juga dapat digunakan untuk bermain *game*, menonton video, pembuatan aplikasi, hingga mendesain suatu produk tertentu. Tentunya, tiap kegunaan tersebut memiliki syarat spesifikasi tertentu agar dapat digunakan dengan optimal. Spesifikasi tersebut meliputi kapasitas memori yang digunakan, prosesor, *graphic card*, dan lainnya.  

Berdasarkan hal tersebut, spesifikasi laptop sangat mempengaruhi kenyamanan serta kinerja laptop saat digunakan oleh pengguna. Namun seperti yang kita ketahui, semakin tinggi spesifikasi laptop yang kita inginkan, maka semakin tinggi pula harga yang harus kita bayar nantinya.

Kini laptop telah bertebaran dimana-mana, pekerja hingga mahasiswa pun mulai membutuhkan laptop untuk membantu pekerjaan mereka. Namun tidak sedikit dari mereka kesulitan mencari laptop yang memiliki spesifikasi mempuni yang sesuai dengan pekerjaan mereka dengan harga yang diinginkan.

## Business Understanding

Berdasarkan latar belakang di atas. Berikut beberapa hal yang akan diselesaikan :  

### Problem Statements

- Bagaimana membantu membuat keputusan bagi penjual laptop dalam menetapkan keuntungan dari harga jual laptop berdasarkan spesifikasi tertentu?
- Berapa harga jual pasar produk laptop berdasarkan spesifikasi tertentu. 

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun model *machine learning* guna memprediksi harga laptop berdasarkan spesifikasi tertentu guna membantu membuat keputusan penjual laptop dalam menentukan harga jual laptop untuk mendapat keuntungan
- Mengetahui harga jual pasar produk laptop berdasarkan spesifikasi tertentu dengan membangun model regresi prediksi harga laptop.

    ### Solution statements
         Adapun cara untuk meraih tujuan tersebut yaitu : 
         1. Membangun model menggunakan algoritma *Random Forest* untuk menyelesaikan kasus regresi.
         2. Membangun model menggunakan algoritma *Boosting* dengan metode *Adaptive Boosting* untuk menyelesaikan kasus regresi.
         3. Membangun model menggunakan algoritma *Linear Regression* untuk menyelesaikan kasus regresi.
         
ketiga solusi tersebut akan dibandingkan dengan mengukur tingkat error menggunakan metrik evaluasi mae atau *mean absolute error* seminim mungkin. Ketika salah satu model paling optimal ditemukan, *baseline* model tersebut akan ditingkatkan lagi dengan *hyperparameter tuning* menggunakan gridsearchcv pada *package* sklearn.

## Data Understanding
data yang akan digunakan yaitu dataset yang dapat diunduh dari kaggle melalui link berikut : [Kaggle laptop price datasets](https://www.kaggle.com/datasets/kuchhbhi/latest-laptop-price-list).  
Dataset di atas memiliki 896 baris data dan 23 kolom. Dataset tersebut belum melewati proses *cleaning*, jadi data yang ada masih berantakan dan perlu dibersihkan terlebih dahulu sebelum diproses oleh algoritma *machine learning*. Dataset di atas merupakan dataset yang menunjukan harga laptop berdasarkan spesifikasi atau fitur-fitur tertentu yang mempengaruhi harga.

### Variabel-variabel pada *laptop prices* dataset adalah sebagai berikut:
- brand: Merek laptop, seperti Asus, Lenovo, Samsung, dan lainnya.
- model: Model dari laptop, seperti ROG, Ryzen, IdeaPad, dan lainnya.
- processor_brand: merk dari processor, seperti AMD, Intel, Qualcom, M1, dan lainnya.
- processor_name: Nama processor.
- processor_gnrtn: Generasi processor, tiap generasi pada prosesor memiliki peningkatan performa yang berbeda.
- ram_gb: Ukuran ram pada laptop dalam satuan *Giga Byte*.
- ram_type: Tipe ram yang digunakan pada laptop, seperti DDR4, DDR5, dan lainnya.
- hdd : Ukuran memori hdd, adapun variasi ukuran hdd pada laptop yaitu ada yang 1024GB, 512GB, dan 2048GB
- os : merupakan *operating system* yang digunakan pada laptop, seperti Windows, Mac, dan DOS.
- os_bit : Bit dari *operating system* pada laptop. Ada 64 bit, dan 32 bit.
- graphic_card_gb : Ukuran dari *graphic card* yang digunakan pada laptop.
- weight : Tipe laptop berdasarkan bobot atau berat, ada yang casual, thinlight, dan gaming.
- display_size : Merupakan ukuran layar dalam satuan inci.
- warranty : Merupakan garansi dalam satuan tahun.
- Touchscreen : Merupakan fitur yang menunjukan apakah laptop menerapkan fitur *touchscreen* atau tidak.
- msoffice : Merupakan fitur yang menunjukkan apakah laptop sudah terinstall msoffice didalamnya atau tidak.
- old_price : Harga laptop original ketika baru *launching* dalam satuan indian rupees(INR).
- latest_price : harga laptop terbaru dalam satuan indian rupees(INR).
- discount : Persentase harga diskon pada laptop.
- star_rating : *Rating* pengguna terhadap laptop dari 0-5.
- ratings : Total *rating* yang diberikan terhadap laptop.
- reviews : Jumlah ulasan pada laptop.

Dalam pemahaman data yang ada, saya melakukan eksplorasi terhadap data dengan teknik *exploratory data analysis* guna mengetahui apakah ada *missing values*, deskripsi dari variabel, dan melakukan *univariate analysis*. Hal tersebut saya lakukan guna menganalisis karakteristik serta mendapatkan asumsi awal pada data.  
- *Brand* yang paling sering muncul pada data :  
![brand counts](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/eda_brand.png)  
Gambar 1. Distribusi *Brand*  

Berdasarkan gambar 1, Asus merupakan *brand* yang paling sering muncul pada data.
- Jenis ukuran ram yang digunakan pada laptop :  
![Ram Size](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/eda_ram.png)  
Gambar 2. Distribusi *Ram Size*  

Berdasarkan gambar 2, Laptop memiliki variasi ukuran ram yang beragam, ada yang 4 GB, 8 GB, 16GB, dan 32 GB. Namun, kebanyakan laptop menggunakan ukuran ram sebesar 8 GB.
- Berbagai merek prosesor yang digunakan pada laptop :  
![Merk Processor](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/eda_processor_brand.png)  
Gambar 3. Distribusi *Processor Brand*  

Berdasarkan gambar 3, Kebanyakan laptop menggunakan merk prosesor intel.
- Ragam generasi prosesor yang digunakan pada laptop :  
![Processor Generation](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/eda_processor_gnrtn.png)  
Gambar 4. Distribusi *Processor Generation*  

Lebih dari 50% laptop yang ada menggunakan prosesor generasi 10 dan 11, dan sisanya menggunakan generasi prosesor selain itu.  
- Jenis ukuran memori ssd dan hdd yang digunakan pada laptop :  
![HDD](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/eda_hdd.png)  
Gambar 5. Distribusi *HDD*  

![SSD](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/eda_ssd.png)  
Gambar 6. Distribusi *SSD*    

Berdasarkan gambar 5 dan 6, Laptop menggunakan variasi ukuran memori yang beragam, mulai dari 0 GB - 3072 GB.
- Laptop yang menerapkan fitur *Touchscreen* didalamnya :  
![Touch Screen](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/eda_touchscreen.png)   
Gambar 7. Distribusi *Touch Screen*  

Berdasarkan gambar 7, Kebanyakan laptop belum mengimplementasikan fitur *touchscreen* pada laptopnya.
- Jenis prosesor yang digunakan pada laptop :  
![Processor Name](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/eda_processor_name.png)  
Gambar 8. Distribusi *Processor Name*  

Berdasarkan gambar 8, Kebanyakan laptop yang ada menggunakan jenis prosesor intel core i5, diikuti dengan core i3, core i7, ryzen 5, dan ryzen 9.
- Laptop yang telah terinstall MsOffice didalamnya :  
![MsOffice](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/eda_msoffice.png)  
Gambar 9. Distribusi *MsOffice*  

Berdasarkan gambar 9, Kebanyakan laptop belum terinstall msoffice didalamnya.
- Tipe ram yang digunakan pada laptop :  
![Ram Type](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/eda_ram_type.png)  
Gambar 10. Distribusi *Ram Type*  

Berdasarkan gambar 10, lebih dari 70% laptop yang ada, menggunakan DDR4 sebagai tipe ramnya.
- Variasi ukuran *graphic card* dalam satuan GB pada laptop :  
![Graphic Card](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/eda_graphic_card.png)  
Gambar 11. Distribusi *Graphic Card*  

Berdasarkan gambar 11, Laptop yang menggunakan graphic card dengan ukuran di atas 2GB berjumlah sedikit.  

Berdasarkan hasil *Exploratory Data Analysis* yang telah dilakukan, ternyata sebagian besar laptop merupakan laptop menengah. Hal ini ditunjukkan oleh penggunaan laptop yang sebagian besar menggunakan ssd sebagai penyimpanannya, menggunakan ukuran ram sebesar 8 GB, menggunakan jenis prosesor intel core i5, dan sebagian besar laptop yang ada menggunakan prosesor generasi 10 dan 11.

Seperti yang telah dijelaskan pada latar belakang di atas, bahwa spesifikasi pada laptop cukup berpengaruh terhadap harga laptop. Berdasarkan informasi yang telah saya dapatkan, kesimpulannya adalah bahwa lebih dari 50% laptop yang ada merupakan laptop yang memiliki spesifikasi menengah dan merupakan laptop keluaran terbaru. Hal ini dibuktikan melalui : 
- lebih dari 50% laptop yang ada menggunakan ukuran ram 8 GB
- lebih dari 50% laptop yang ada menggunakan prosesor dengan generasi 10 dan 11. Hal ini juga menunjukkan bahwa laptop-laptop yang ada merupakan laptop keluaran terbaru.
- lebih dari 50% laptop yang ada menerapkan memori SSD dengan kapasitas ukuran 512 GB dan 256 GB. Yang artinya bahwa penerapan memori ssd memiliki performa lebih cepat baik dalam write maupun read, sehingga hal itu menunjukkan bahwa laptop memiliki performa yang terbilang cukup cepat jika dibandingkan dengan hdd.


## Data Preparation
Dataset yang kita punya belum melalui proses *cleaning*. Oleh sebab itu, sebelum data benar-benar siap diolah oleh algoritma *machine learning*, saya akan melakukan proses pembersihan data terlebih dahulu. Pembersihan data disini mencakup : 
- Menangani *missing value* dengan teknik *arbitrary imputation*
  setelah data berhasil di *load* dan ditampilkan. dengan menggunakan fungsi `df.isna().sum()`, Data menunjukkan bahwa tidak ada data yang hilang. Hal ini disebabkan karna fungsi `isna()` hanya akan memeriksa jika suatu baris bernilai NaN, sehingga sistem tidak menunjukkan terdapat *missing value* pada data. Namun, jika kita lihat 5 data teratas, terlihat bahwa pada kolom "display_size" terdapat nilai "Missing" yang berarti *missing value* pada data. Sehingga perlu kita tangani data tersebut. Hal yang akan saya lakukan yaitu, mengubah semua baris "Missing" pada data menjadi NaN terlebih dahulu, hal ini dilakukan agar dapat mendeteksi kembali apakah terdapat *missing value* pada kolom yang lain. Lalu langkah selanjutnya yaitu mengganti *missing value* tersebut dengan teknik imputasi.
  
  Teknik imputasi ini merupakan teknik yang digunakan untuk mentetapkan nilai lain guna mengganti nilai yang hilang, tidak benar, atau tidak sesuai. Adapun beberapa teknik imputasi yang dapat digunakan seperti imputasi mean, median, modus, *zero value*, ataupun *arbitrary*. Dalam penerapan teknik imputasi ini saya akan menggunakan teknik imputasi *arbitrary*(suka-suka). Pada saat pengecekan *missing value*, terlihat bahwa terdapat 328 data yang hilang pada kolom "display_size", 238 data yang hilang pada kolom "processor_gnrtn", dan 98 data yang hilang pada kolom "model". Oleh sebab itu, hal yang pertama kali saya lakukan yakni menghapus kolom yang memiliki *missing value* paling banyak, dalam hal ini yaitu kolom "display_size". Hal tersebut saya lakukan karena banyak data yang hilang tersebut akan berimbas terhadap hasil prediksi model nantinya. Mengingat kita memiliki 23 kolom keseluruhan, penghapusan satu buah kolom tersebut tidak menjadikan dataset yang kita punya menjadi tidak informatif. 
  
  Selanjutnya, kita memiliki 2 buah kolom yang memiliki *missing value*, yakni kolom "processor_gnrtn", dan "kolom model". Mengingat 2 fitur tersebut merupakan fitur kategorikal, sehingga saya menerapkan teknik imputasi *arbitrary* dengan menambahkan kategori baru yakni "x" yang merepresentasikan kategori hilang. Hal tersebut bertujuan agar saya tidak kehilangan informasi jika melakukan penghapusan pada 2 buah kolom yang memiliki *missing value* seperti penghapusan kolom "display_size" tersebut.

- menghilangkan karakter string pada data sekaligus mengubahnya menjadi tipe data *number*
  Pada tahapan ini, saya akan mengubah beberapa fitur yang sebenarnya sudah merepresentasikan angka, seperti fitur ram_gb, hdd dan ssd. Jika kita melihat sekilas terhadap data yang ada, kolom ram_gb, ssd, dan hdd sudah merepresentasikan satuannya yaitu GB. Sehingga pada kolom ram_gb, baris data dengan akhiran "GB GB" akan kita ganti menggunakan fungsi `replace()` dengan mengganti kata "GB GB", menjadi string kosong "". Hal ini saya lakukan agar dapat mengubahnya menjadi tipe data *numerik*. Begitu pula dengan kolom SSD dan HDD, saya akan mengganti baris data yang memiliki kata "GB" dengan string kosong "", dengan tujuan yang sama.
- Mengubah data ordinal menjadi dengan teknik *label encoder*.  
Dalam penerapannya, *label encoder* ini digunakan untuk mengubah data kategorikal menjadi *numerik* dengan catatan bahwa urutan dari kategori diperhatikan. Dalam data di atas, terdapat beberapa data ordinal atau data yang berurutan seperti pada kolom ram_type, dan processor_gnrtn. Seperti yang kita tahu, processor_gnrtn memiliki urutan mulai dari urutan terendah hingga yang paling tinggi mulai dari generasi 4,7,8,9,10,11, dan 12.
- Mengubah data binary menjadi angka
*Binary data* yang dimaksud yakni data yang hanya memiliki 2 buah kategori saja, seperti *Yes* | *No*, 64 bit | 32 Bit, dan lainnya. Pada dataset di atas, kita memiliki 3 buah kolom yang berdata ordinal, yakni kolom  *Touchscreen*, msoffice, dan os_bit. Oleh sebab itu saya akan mengubah data ordinal tersebut menjadi data *numerik* yang berisikan angka 0 dan 1 saja. 0 merepresentasikan *No* dan 32bit, sedangkan 1 merepresentasikan *Yes* dan 64bit.
- Mengubah data nominal menjadi angka menggunakan teknik *one hot encoding*.
Pada dataset *laptop price* di atas, kita memiliki banyak data kategorikal. Oleh sebab itu, kita perlu menanganinya dengan mengubah data kategorikal menjadi data *numerik*. Untuk data nominal atau data yang tidak memiliki urutan ini, kita dapat menanganinya dengan teknik *one hot encoding*. Teknik ini akan membuat kolom baru yang merepresentasikan masing-masing kategori yang ada dengan angka satu pada kolom acuan.

   Hal di atas perlu dilakukan mengingat algoritma *machine learning* akan lebih mudah mengelola data berupa angka.

- Membagi data latih dan data uji
Lalu selanjutnya kita perlu membagi data menjadi data latih dan data uji. Hal ini perlu kita lakukan, guna membagi dataset di atas menjadi beberapa bagian yaitu data latih untuk melatih model, dan data uji yang berfungsi sebagai data baru atau *input* baru untuk diujikan pada model yang telah kita buat. Hal tersebut bertujuan untuk mengukur seberapa baik model yang kita buat dapat men-generalisasi data yang baru tersebut. Adapun porsi pembagian data latih dan data uji yang akan saya bagi yakni 80% data latih, dan 20% data uji. 

## Modeling
Setelah data selesai melalui tahap *preprocessing* dan *data cleaning*. Selanjutnya saya akan membuat model untuk memprediksi label "latest_price" untuk menjawab rumusan masalah di atas. Dalam pembuatan model ini, saya akan membandingkan ketiga algoritma yang telah disebutkan di atas. Adapun metrik evaluasi yang akan digunakan pada kesempatan kali ini yaitu metrik evaluasi *Mean Absolute Error* (MAE). Tujuannya yaitu mencari model paling optimal yang dapat menghasilkan skor mae sekecil mungkin.
- *Random Forest*
Algoritma yang pertama, saya akan menggunakan algoritma *Random Forest*. Kasus kali ini, saya akan memprediksi data kontinu sehingga kasus ini yaitu kasus regresi. Dalam suatu penelitian dijelaskan bahwa *Random Forest* merupakan metode *ensemble learning* yang menggunakan *decision tree* sebagai *baseline model* yang dikombinasikan [1]. Lebih lanjut, dalam suatu penelitian disebutkan bahwa algoritma ini memiliki kelebihan seperti mampu menghasilkan *error* yang relatif rendah, dan dapat mengatasi jumlah data yang besar dalam proses pelatihan dengan lebih efisien [2]. Parameter yang akan saya gunakan yaitu n_estimators (jumlah pohon) sebanyak 80, dan max_depth(kedalaman) sebanyak 16. Parameter tadi saya pilih secara acak, guna mendapatkan model awal secepatnya untuk membandingkan seberapa baik algoritma dalam prediksi. Dalam praktiknya, algoritma ini mampu mendapatkan skor mae sebesar 1517 terhadap data latih.
- *Boosting*
Algoritma *Boosting* ini merupakan model *ensemble* juga, *Boosting* bekerja dengan tujuan untuk meningkatkan performa dengan mengubah *weak learner model* menjadi *strong learner* [3]. Perbedaan algoritma *Boosting* dengan *Random Forest* yaitu bahwa *Random Forest* bekerja secara paralel, sedangkan *Boosting* bekerja secara berurutan.  
Pada praktiknya, saya akan menggunakan algoritma *Boosting* dengan metode *Adaptive Boosting*. Adapun parameter yang saya gunakan yaitu n_estimators sebanyak 50, dan learning_rate sebesar 0.005. Adapun skor mae yang didapatkan menggunakan algoritma ini sebesar 13073 terhadap data latih.  

- *Linear Regression*
ALgoritma yang terakhir yaitu regresi linear, pemilihan algoritma ini saya lakukan untuk melihat seberapa baik algoritma *ensemble* dengan algoritma biasa. Adapun skor mae yang didapatkan pada model ini sebesar 8870 terhadap data latih.  
Berikut merupakan distribusi prediksi dengan label sebenernya menggunakan algoritma *Random Forest*, *Adaptive Boosting*, dan *Linear Regression* secara berurutan :  
![Random Forest Distribution](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/distribusi_rf.png)  
Gambar 12. Distribusi Hasil Prediksi dengan Label Asli menggunakan *Random Forest Algorithm*  

![Boosting Distribution](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/distribusi_boosting.png)  
Gambar 13. Distribusi Hasil Prediksi Dengan Label Asli menggunakan *Boosting Algorithm*  

![Linear Regression Distribution](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/distribusi_linreg.png)  
Gambar 14. Distribusi Hasil Prediksi Dengan Label Asli menggunakan *Linear Regression Algorithm*  

Sekilas terlihat bahwa algoritma *Random Forest* pada gambar 12 membentuk pola garis lurus. Hal ini menunjukkan bahwa hasil prediksi model dengan label asli tidak melenceng terlalu jauh.

Oleh sebab itu, dari ketiga algoritma tersebut algoritma *Random Forest* memiliki tingkat error yang paling minim jika dibandingkan dengan algoritma yang lain. Berdasarkan hal tersebut model dengan algoritma *Random Forest* merupakan model terbaik untuk memprediksi harga laptop dengan tingkat *error* yang kecil dan akurat.  
Untuk itu, saya akan melakukan peningkatan terhadap model ini dengan tujuan dapat memperkecil tingkat *error* agar prediksi model lebih akurat lagi. Adapun teknik yang akan saya gunakan yaitu menggunakan *GridSearchCV* pada *library* sklearn.  

Dalam praktiknya, saya akan mengumpulkan parameter-parameter yang akan diujikan. Dalam tahap ini, saya akan menguji parameter n_estimator sebanyak 50, 60, 70, 80, 90,dan 100. Adapun untuk parameter max_depthnya saya akan mencoba nilai 4, 8, 16, 32, dan 64.  
Selanjutnya saya akan mencari parameter terbaik dengan mengirimkan parameter terhadap fungsi `GridSearchCV()` berupa model yang akan dicari parameter terbaiknya,  dalam hal ini yaitu *Random Forest*, serta parameter terbaik yang akan dicarinya, kemudian *cross validation* atau tiap berapa kali pengujian akan dilakukan, dan metrik evaluasi apa yang akan dicari skor terbaiknya, dalam hal ini saya akan mencari metrik evaluasi mae sebagai skornya. Setelah tahapan tersebut dilakukan, selanjutnya yaitu menampilkan parameter terbaik dengan menjalankan kode `best_params_`.  

Selanjutnya kita latih kembali data dengan model *Random Forest* menggunakan parameter terbaik yang telah didapatkan sebelumnya. Kemudian kita cek skor mae menggunakan data uji dan data latih. Hasilnya, model memperoleh score mae sebesar 3259 pada data uji. Hasil ini mengalami peningkatan mengingat skor mae pada data uji sebelumnya yaitu sebesar 3272.

## Evaluation
Pada studi kasus ini, metrik yang akan digunakan yaitu *mean absolute error*(MAE).  
### *Mean Absolute Error*
![MAE](https://www.statisticshowto.com/wp-content/uploads/2016/10/MAE.png)  
Gambar 15. Formula MAE  

Sederhananya, metriks ini menghitung absolut atau nilai mutlak dari selisih hasil sebenarnya dengan hasil prediksi. Jika dibandingkan dengan mse, mse akan menghitung kuadrat dari selisih nilai sebenarnya dengan nilai prediksi. Yang artinya, jika hasil prediksi besar, maka mse akan memiliki nilai yang jauh lebih besar. Begitu pula sebaliknya, jika selisih antara nilai sebenarnya dengan prediksi kecil, mse akan memperkecil nilai tersebut.  
Berikut perbandingan ketiga algoritma berdasarkan hasil skor mae yang diperolehnya :  
![Hasil Mae](https://github.com/RipanRenaldi/project-01-predictiveanalytics/blob/main/img/hasil_algoritma_mae.png)  
Gambar 16. Skor MAE pada Masing-Masing Algoritma    


Tabel 1. Evaluasi MAE Tiga Algoritma
|     Algoritma     | Train MAE | Test MAE |
|-------------------|-----------|----------|
| Random Forest     | 1529.49   | 3259.10  |
| Adaptive Boosting | 13277.22  | 13489.85 |
| Linear Regression | 8870.70   | 12176.50 |

Pada gambar 16 dan tabel 1 dapat terlihat bahwa algoritma *Random Forest* menghasilkan skor mae yang paling kecil diantara 3 algoritma yang digunakan.  
Berikut perbandingan hasil prediksi model dengan label asli pada 2 buah baris data :  

Tabel 2. Perbandingan Prediksi Dengan Label Asli
|   | Label Asli | Hasil Prediksi |
|---|------------|----------------|
|   | 113590     | 108780.0       |
|   | 169990     | 158299.8       |  

Berdasarkan grafik dan tabel di atas, model dapat memberikan gambaran terkait harga laptop pada pasar berdasarkan spesifikasi tertentu. Dengan hasil prediksi yang diberikan, model dapat membantu *stake holder* dalam mengambil keputusan.

## Kesimpulan
Algoritma *Random Forest* ini memiliki tingkat error yang paling minim jika dibandingkan dengan kedua algoritma di atas. Dengan demikian, model yang telah dibuat dapat memprediksi harga laptop berdasarkan spesifikasi tertentu dengan tingkat *error* yang tidak terlalu jauh. Berdasarkan hasil prediksi yang diberikan, para pemangku keputusan baik itu penjual laptop, maupun *customer* yang akan membeli laptop mendapatkan gambaran terkait harga pasar laptop yang ada, dan dapat menentukan harga jual laptop untuk meraih keuntungan.  


## Daftar Pustaka
[1]	A. Primajaya and S. B. Nurina, “Random Forest Algorithm for Prediction of Precipitation,” Indonesian Journal of Artificial Intelligence and Data Mining (IJAIDM), vol. 1, no. 1, pp. 27–31, Mar. 2018.  
[2]	Y. Religia, A. Nugroho, and W. Hadikristanto, “Analisis Perbandingan Algoritma Optimasi pada Random Forest untuk  Klasifikasi Data Bank Marketing,” JURNAL RESTI, vol. 5, no. 1, pp. 187–192, Feb. 2021.  
[3]	R. M. Yanuar, W. S. Adhi, and S. D. Madya, “ANALISIS KINERJA HAND TRACKING-BY-DETECTIONUNTUKHOLOGRAM INTERAKTIF MENGGUNAKAN MODEL ADAPTIVE BOOSTING,” e-Proceeding of Engineering, vol. 7, no. 1, pp. 340–347, Apr. 2020.  

