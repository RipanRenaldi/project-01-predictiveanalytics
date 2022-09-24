# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Perkembangan teknologi pada revolusi industri 4.0 ini mengakibatkan banyak orang memanfaatkan teknologi informasi yang ada saat ini. Teknologi tersebut tidak lagi sulit untuk didapatkan, mengingat dulu tidaklah mudah untuk mendapatkan teknologi komputer yang ada seperti sekarang ini. Manfaat komputer atau laptop ini sangatlah banyak, salah satunya dalam pencarian informasi yang tersedia pada internet. Jika dilihat dari sejarah perkembangan komputer, kini komputer sudah memiliki ukuran yang sangat jauh lebih kecil dan memiliki performa yang jauh lebih cepat dibandingkan dengan dulu. Canggihnya lagi, kini komputer memiliki ukuran yang jauh lebih kecil dan  dapat kita bawa kemana-mana. Dengan ukuran yang kecil itulah laptop yang ada saat ini lebih praktis untuk dapat digunakan dimanapun dan kapanpun.

Setiap laptop yang ada, memiliki merk dan spesifikasi yang berbeda-beda, spesifikasi tersebut berpengaruh terhadap performa atau kinerja laptop saat digunakan. Tentunya, dengan laptop yang memiliki spesifikasi tinggi akan mempengaruhi performa laptop saat digunakan menjadi lebih cepat. Kini, penggunaan laptop tidak hanya untuk pencarian informasi saja. Laptop juga dapat digunakan untuk bermain game, menonton video, pembuatan aplikasi, hingga mendesain suatu produk tertentu. Tentunya, tiap kegunaan tersebut memiliki syarat spesifikasi tertentu agar dapat digunakan dengan optimal. Spesifikasi tersebut meliputi memory yang digunakan, processor, graphic card, dan lainnya.  

Berdasarkan hal tersebut, spesifikasi laptop sangatlah mempengaruhi kenyamanan serta kinerja laptop saat digunakan. Namun seperti yang kita ketahui, semakin tinggi spesifikasi laptop yang kita inginkan, maka semakin tinggi pula harga yang harus kita bayar nantinya. Walaupun spesifikasi bukanlah indikator pasti dalam penentuan harga laptop, namun ada banyak faktor yang memengaruhi harga laptop seperti merk, tampilan, desain, dan perangkat lunak yang terinstall pada laptop.

Kini laptop telah bertebaran dimana-mana, pekerja hingga mahasiswa pun mulai membutuhkan laptop untuk membantu pekerjaan mereka. Namun tidak sedikit dari mereka kesulitan mencari laptop yang memiliki spesifikasi mempuni yang sesuai dengan pekerjaan mereka dengan harga tertentu.

## Business Understanding

Berdasarkan latar belakang di atas. Berikut beberapa hal yang akan diselesaikan :  

### Problem Statements

- Bagaimana membantu membuat keputusan bagi penjual laptop dalam menetapkan keuntungan dari harga jual laptop berdasarkan spesifikasi tertentu?
- Berapa harga jual pasar produk laptop berdasarkan spesifikasi tertentu. 

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun model machine learning guna memprediksi harga laptop berdasarkan spesifikasi tertentu guna membantu membuat keputusan penjual laptop dalam menentukan harga jual laptop untuk mendapat keuntungan
- Mengetahui harga jual pasar produk laptop berdasarkan spesifikasi tertentu dengan membangun model regresi prediksi harga laptop.

    ### Solution statements
         Adapun cara untuk meraih tujuan tersebut yaitu : 
         1. Membangun model menggunakan algoritma Random Forest untuk menyelesaikan kasus regresi
         2. Membangun model menggunakan algoritma Boosting(adaboost) untuk menyelesaikan kasus regresi
         3. Membangun model menggunakan algoritma Linear Regression untuk menyelesaikan kasus regresi.
         
ketiga solusi tersebut akan dibandingkan dengan mengukur tingkat error menggunakan metrik evaluasi mae atau mean absolute error seminim mungkin. Ketika salah satu model paling optimal ditemukan, baseline model tersebut akan ditingkatkan lagi dengan hyperparameter tuning menggunakan gridsearchcv pada package sklearn.

## Data Understanding
data yang akan digunakan yaitu dataset yang dapat diunduh dari kaggle melalui link berikut : [Kaggle laptop price datasets](https://www.kaggle.com/datasets/kuchhbhi/latest-laptop-price-list).  
Dataset di atas memiliki 896 baris data dan 23 kolom. Dataset tersebut belum melewati proses cleaning, jadi data yang ada masih berantakan dan perlu dibersihkan terlebih dahulu sebelum diproses oleh algoritma machine learning. Dataset di atas merupakan dataset yang menunjukan harga laptop berdasarkan spesifikasi atau fitur-fitur tertentu yang mempengaruhi harga.

### Variabel-variabel pada laptop prices dataset adalah sebagai berikut:
- brand: Merek laptop, seperti Asus, Lenovo, Samsung, dan lainnya.
- model: Model dari laptop, seperti ROG, Ryzen, IdeaPad, dan lainnya.
- processor_brand: merk dari processor, seperti AMD, Intel, Qualcom, M1, dan lainnya.
- processor_name: Nama processor.
- processor_gnrtn: Generasi processor, tiap generasi pada processorr memiliki peningkatan performa yang berbeda.
- ram_gb: Ukuran ram pada laptop dalam satuan Giga Byte.
- ram_type: Tipe ram yang digunakan pada laptop, seperti DDR4, DDR5, dan lainnya.
- hdd : Ukuran memori eksternal hdd, adapun variasi ukuran hdd pada laptop yaitu ada yang 1024GB, 512GB, dan 2048GB
- os : merupakan operating system yang digunakan pada laptop, seperti Windows, Mac, dan DOS.
- os_bit : Bit dari operating system pada laptop. Ada 64 bit, dan 32 bit.
- graphic_card_gb : Ukuran dari graphic card yang digunakan pada laptop.
- weight : Tipe laptop berdasarkan bobot atau berat, ada yang casual, thinlight, dan gaming.
- display_size : Merupakan ukuran layar dalam satuan inci.
- warranty : Merupakan garansi dalam satuan tahun.
- Touchscreen : Merupakan fitur yang menunjukan apakah laptop menerapkan fitur touchscreen atau tidak.
- msoffice : Merupakan fitur yang menunjukkan apakah laptop sudah terinstall msoffice didalamnya atau tidak.
- old_price : Harga laptop original ketika baru launching dalam satuan indian rupees.
- latest_price : harga laptop terbaru dalam satuan indian rupees.
- discount : Persentase harga diskon pada laptop.
- star_rating : Rating pengguna terhadap laptop dari 0-5.
- ratings : Total rating yang diberikan terhadap laptop.
- reviews : Jumlah reviewa atau ulasan pada laptop.

Dalam pemahaman data yang ada, saya melakukan eksplorasi terhadap data dengan teknik exploratory data analysis mencakup penanganan missing values, deskripsi variabel, dan univariate anaklysis. Hal tersebut saya lakukan guna menganalisis karakteristik serta mendapatkan asumsi awal pada data.
Adapun informasi yang akan didapatkan berguna untuk menjawab beberapa pertanyaan berikut:  
- Brand atau merk produk laptop apa yang paling banyak muncul dalam data?
- Ada berapa variasi ukuran ram yang digunakan pada laptop?
- Merk processor apa saja yang digunakan pada laptop?
- Processor generasi berapa saja yang disisipkan pada laptop?
- Ada berapa variasi kapasitas memori ssd yang digunakan pada laptop?
- Ada berapa variasi kapasitas memori hdd yang digunakan pada laptop?
- Apakah laptop yang ada menerapkan fitur touch screen?
- Jenis processor apa saja yang digunakan pada laptop?
- Apakah kebanyakan laptop telah terinstall msoffice didalamnya?
- Tipe ram apa saja yang digunakan pada laptop?
- Ukuran VGA (dalam GB) yang digunakan pada laptop?

Seperti yang telah dijelaskan pada latar belakang di atas, bahwa spesifikasi pada laptop cukup berpengaruh terhadap harga laptop. Berdasarkan hal tersebut, pada exploratory data analysis yang telah saya lakukan bertujuan untuk memahami apakah data laptop yang ada memiliki spesifikasi yang tinggi atau menengah. Berdasarkan informasi yang telah saya dapatkan, kesimpulannya adalah bahwa lebih dari 50% laptop yang ada merupakan laptop yang memiliki spesifikasi menengah dan merupakan laptop keluaran terbaru. Hal ini dibuktikan melalui : 
- lebih dari 50% laptop yang ada memiliki ram 8 GB
- lebih dari 50% laptop yang ada menggunakan processor dengan generasi 10 dan 11. Hal ini juga menunjukkan bahwa laptop-laptop yang ada merupakan laptop keluaran terbaru.
- lebih dari 50% laptop yang ada menerapkan memori eksternal SSD dengan kapasitas ukuran 512 GB dan 256 GB. Yang artinya bahwa penerapan memori ssd memiliki performa lebih cepat baik dalam write maupun read, sehingga hal tersebut menunjukkan bahwa laptop memiliki performa yang terbilang cukup cepat jika dibandingkan dengan hdd.

Seperti yang telah dijelaskan sebelumnya, bahwa dataset ini belum melalui proses cleaning. Oleh sebab itu, sebelum masuk ke tahap preprocessing, saya akan melakukan proses pembersihan data terlebih dahulu. Pembersihan data disini mencakup : 
- Penanganan missing value.  
  setelah data berhasil di load dan ditampilkan. dengan menggunakan fungsi `df.isna().sum()`, Data menunjukkan bahwa tidak ada data yang hilang. Hal ini disebabkan karna fungsi isna() hanya akan memeriksa jika suatu baris memiliki nilai NaN, sehingga sistem tidak menunjukkan terdapat missing value pada data. Namun, jika kita lihat 5 data teratas, terlihat bahwa pada kolom "display_size" terdapat nilai "Missing" yang berarti hal ini merupakan missing value pada data. Sehingga perlu kita tangani data tersebut. Hal yang akan saya lakukan yaitu, mengubah semua baris "Missing" pada data menjadi NaN terlebih dahulu, hal ini dilakukan agar dapat mendeteksi kembali apakah terdapat missing value pada kolom yang lain. Lalu langkah selanjutnya yaitu mengganti missing value tersebut dengan teknik imputasi.
  
  Teknik imputasi ini merupakan teknik yang digunakan untuk mentetapkan nilai lain guna mengganti nilai yang hilang, tidak benar, dan tidak sesuai [1]. Adapun beberapa teknik imputasi yang dapat digunakan seperti imputasi mean, median, modus, zero value, ataupun arbitrary. Dalam penerapan teknik imputasi ini saya akan menggunakan teknik imputasi arbitrary(suka-suka). Pada saat pengecekan missing value, saya mendapati bahwa terdapat 328 data yang hilang pada kolom display size, 238 data yang hilang pada kolom processor_gnrtn, dan 98 data yang hilang pada kolom model. Oleh sebab itu, hal yang pertama kali saya lakukan yakni menghapus kolom yang memiliki data hilang paling banyak, dalam hal ini yaitu kolom "display_size". Hal tersebut saya lakukan karena banyak data yang hilang tersebut akan berimbas terhadap hasil prediksi model nantinya. Mengingat kita memiliki 23 kolom keseluruhan, sehingga penghapusan satu buah kolom tersebut tidak menjadikan data tidak informatif. 
  
  Selanjutnya, kita memiliki 2 buah kolom yang memiliki missing value, yakni kolom processor_gnrtn, dan kolom model. Mengingat 2 fitur tersebut merupakan fitur kategorikal, sehingga saya menerapkan teknik imputasi arbitrary dengan menambahkan kategori baru yakni "x" untuk merepresentasikan kategori yang hilang. Hal tersebut bertujuan agar saya tidak kehilangan informasi jika melakukan penghapusan pada 2 buah kolom tersebut.

- Mengubah tipe data dari kategorikal menjadi numerikal
  Pada tahapan ini, saya akan mengubah beberapa fitur yang sebenarnya sudah merepresentasikan angka, seperti fitur ram_gb, hdd dan ssd. Jika kita melihat sekilas terhadap data yang ada, kolom ram_gb, ssd, dan hdd sudah merepresentasikan satuannya yaitu GB. Sehingga pada kolom ram_gb, baris data dengan akhiran "GB GB" akan kita ganti menggunakan fungsi `replace` dengan mengganti kata "GB GB", menjadi string kosong "". Hal ini saya lakukan agar dapat mengubahnya menjadi kolom numerik, karna seperti yang kita tahu, bahwa algoritma machine learning dapat mengolah data angka. Begitu pula dengan kolom SSD dan HDD, saya akan mengganti baris data yang memiliki kata "GB" dengan string kosong "", dengan tujuan yang sama.
 
- Melihat korelasi fitur kategorical terhadap label latest_price.  
  Kesimpulan yang saya dapatkan yaitu :  
  1. Fitur processor_gnrtn memiliki pengaruh terhadap label latest_price.  
  2. Fitur Touchscreen memiliki pengaruh yang tinggi terhadap label latest_price.

## Data Preparation
Kini saya akan melakukan tahap preprocessing data sepenuhnya agar data benar-benar siap diolah oleh algoritma machine learning.
Adapun teknik data preparation yang akan dilakukan yakni : 

- Mengubah data ordinal menjadi angka dengan menggunakan teknik label encoder.  
Dalam penerapannya, label encoder ini digunakan untuk mengubah data kategorikal menjadi numerik dengan catatan bahwa urutan dari kategori adalah hal yang penting. Dalam data di atas, terdapat beberapa data ordinal atau data yang berurutan seperti pada kolom ram_type, dan processor_gnrtn. Seperti yang kita tahu, processor_gnrtn memiliki urutan mulai dari urutan terendah hingga yang paling tinggi mulai dari generasi 4,7,8,9,10,11, dan 12.
- Mengubah data binary menjadi angka dengan menerapkan fungsi `lambda` untuk mengubah data binary
Data binary yang dimaksud yakni data yang hanya memiliki 2 buah kategori saja, seperti Yes | no, 64 bit | 32 Bit, dan lainnya. Pada dataset di atas, kita memiliki 3 buah kolom yang berdata ordinal, yakni kolom  Touchscreen, msoffice, dan os_bit. Oleh sebab itu saya akan mengubah data ordinal tersebut menjadi data numerik yang berisikan angka 0 dan 1 saja. 0 merepresentasikan No dan 32bit, sedangkan 1 merepresentasikan Yes dan 64bit.
- Mengubah data nominal menjadi angka menggunakan teknik one hot encoding.
Pada dataset laptop price di atas, kita memiliki banyak data kategorikal. Oleh sebab itu, kita perlu menanganinya dengan mengubah data kategorikal menjadi data numerik. Untuk data nominal atau data yang tidak memiliki urutan ini, kita dapat menanganinya dengan teknik one hot encoding. Teknik one hot encoding ini akan membuat kolom baru yang merepresentasikan masing-masing kategori yang ada dengan angka satu pada kolom acuan.

   ketiga hal di atas perlu dilakukan mengingat algoritma machine learning akan lebih mudah mengelola data berupa angka.

- Membagi data latih dan data uji
Lalu selanjutnya kita perlu membagi data menjadi data latih dan data uji. Hal ini perlu kita lakukan, guna membagi dataset di atas menjadi beberapa bagian yaitu data latih untuk melatih model, dan data uji yang berfungsi sebagai data baru atau input baru untuk model. Hal tersebut berguna untuk mengukur seberapa baik model yang kita buat dapat men-generalisasi data yang baru tersebut. Adapun porsi pembagian data latih dan data uji yang akan saya bagi yakni 80% data latih, dan 20% data uji. 


## Modeling
Setelah data selesai melalui tahap preprocessing dan data cleaning. Selanjutnya saya akan membuat model untuk memprediksi label "latest_price" untuk menjawab rumusan masalah di atas. Dalam pembuatan model ini, saya akan membandingkan ketiga algoritma yang telah disebutkan di atas. 
- Random Forest
Algoritma yang pertama saya akan menggunakan algoritma Random Forest. Kasus kali ini, saya akan memprediksi data kontinu sehingga kasus ini yaitu kasus regresi. Adapun parameter yang akan saya gunakan yaitu n_estimators sebanyak 80, dan max_depth sebanyak 16. Parameter tadi saya pilih secara acak, guna mendapatkan base model secepatnya untuk membandingkan seberapa baik algoritma dalam memprediksi model. Dalam praktiknya, algoritma ini mampu mendapatkan score mae atau mean absolute error sebesar 1527 terhadap data latih dan 3579 pada data uji.
- Boosting
Model selanjutnya yaitu saya akan mencoba menggunakan algoritma Boosting menggunakan metode Adaptive Boosting. Adapun parameter yang saya gunakan yaitu n_estimators sebanyak 50, dan learning_rate sebesar 0.005. Dalam praktiknya, score mae yang didapatkan menggunakan algoritma ini sebesar 13396 terhadap data uji.  

Kedua algoritma di atas, baik Random Forest maupun Boosting Algortihm merupakan model ensemble. Artinya bahwa model tersebut memiliki beberapa model yang bekerja sama untuk mendapatkan hasil prediksi yang diinginkan. Namun perbedaannya, tiap model pada Random Forest akan bekerja secara paralel, sedangkan tiap model pada boosting akan bekerja secara berurutan.

- Linear Regression
ALgoritma yang terakhir yaitu regresi linear, pemilihan algoritma ini saya lakukan untuk melihat seberapa baik algoritma ensemble dengan algoritma biasa. Adapun score mae yang didapatkan pada model ini sebesar 8800 terhadap data latih.  
**Untuk lebih detailnya mengenai metric mae ini akan dijelaskan di bab evaluation.

Dari ketiga algoritma tersebut, algoritma Random Forest memiliki tingkat error yang paling minim jika dibandingkan dengan yang lainnya. Oleh sebab, itu model dengan algoritma Random Forest merupakan model terbaik untuk memprediksi harga laptop dengan tingkat error seminim dan seakurat mungkin.  
Untuk itu, saya akan melakukan peningkatan terhadap model ini dengan tujuan dapat memperkecil tingkat error agar prediksi model lebih akurat lagi. Adapun teknik yang akan saya gunakan yaitu menggunakan GridSearchCV pada library sklearn.   
Dalam praktiknya, saya akan mengumpulkan parameter-parameter yang akan diujikan. Dalam tahap ini, saya akan menguji parameter n_estimator dengan sebanyak 50, 60, 70, 80, 90,dan 100. Adapun untuk kedalaman pohonnya / max_depthnya saya akan mencoba nilai 4, 8, 16, 32, dan 64.   
Source Code :  
`grid_params = {"n_estimators" : [50,60,70,80,90,100], "max_depth" : [4,8,16,32,64]}`  
Selanjutnya saya akan mencari parameter terbaik menggunakan fungsi `GridSearchCV()` dengan mengirimkan parameter berupa model regressor yang akan dicari parameter terbaiknya dalam hal ini yaitu `RandomForestRegressor()`, grid parameternya atau parameter yang akan dicarinya, kemudian cross validation atau tiap berapa kali pengujian akan dilakukan, dan metrik evaluasi apa yang akan dicari score terbaiknya dalam hal ini saya akan mencari metrik evaluasi mae sebagai scoringnya. Setelah tahapan tersebut dilakukan, selanjutnya yaitu menampilkan parameter terbaik dengan menjalankan kode `best_params_`.  
Selanjutnya kita buat kembali model dengan algoritma Random Forest menggunakan parameter yang telah didapatkan sebelumnya. Kemudian kita cek score mae menggunakan data uji dan data latih. Hasilnya, model memperoleh score mae sebesar 3429 pada data uji. Hasil ini mengalami peningkatan mengingat score mae pada data uji sebelumnya yaitu sebesar 3579.

## Evaluation
Pada studi kasus ini, metrik yang saya gunakan yaitu mean absolute error.  
![MAE](https://www.statisticshowto.com/wp-content/uploads/2016/10/MAE.png)  
Sederhananya, metriks ini menghitung absolut atau nilai mutlak dari selisih hasil sebenarnya dengan hasil prediksi. Jika dibandingkan dengan mse, mse akan menghitung kuadrat dari selisih nilai sebenarnya dengan nilai prediksi. Yang artinya, jika hasil prediksi besar, maka mse akan memiliki nilai yang jauh lebih besar. Begitu pula sebaliknya, jika selisih antara nilai sebenarnya dengan prediksi kecil, mse akan memperkecil nilai tersebut, mengingat formula mse adalah kuadrat dari selisih hasil.
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.


Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
