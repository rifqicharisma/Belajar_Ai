# Belajar_Ai
Berisi tentang semua hal yang saya pelajari dalam Ai

## -<(Requirement)>-
- Download dan install anaconda environment : https://www.anaconda.com/products/individual
- Pakai text editor python : `Jupyter Notebook, VS Code, PyCharm` (saya biasa menggunakan VS Code dan Jupyter Notebook). Jika anda tidak ingin menginstall anaconda anda bisa menggunakan `Google Colaboratory`
- **Note : Jika anda menggunakan Google Colab, beberapa code mungkin akan eror, itu dikarenakan penulisan code antara google colab dengan jupyter notebook sedikit berbeda**
- Setiap projek membutuhkan library yang berbeda-beda, install library sesuai petunjuk pada `Readme` setiap projek
> Jika anda tidak menemukan `Readme` pada folder projek, kemungkinan besar projek tersebut dalam masa pengembangan/masih belum sempura

## -<(Jupyter Notebok Extensions (nbextensions))>-
Ekstensi untuk jupyter notebook, di dalam library ini terdapat banyak sekali ekstensi jadi cukup praktis tinggal install satu library saja. Saya menemukan library ini ketika saya ingin mencari `autocomplete` pada jupyter notebook dan saya kaget ternyata banyak sekali ekstensi yang saya dapat ketika saya menginstall library ini

Hasilnya seperti ini :
 ![image](https://user-images.githubusercontent.com/58881125/136680487-ef0278e7-98ce-43d3-a0c2-1306ad0dda0e.png)

 Bisa dilihat terdapat banyak sekali ekstensi namun kekurangannya tidak ada penjelasan mengenai kegunaan ekstensinya. Anda dapat melihat penjelasannya melalui website pecipta : https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions.html

### Cara Install
- Buka CMD Anaconda, lalu ketik : `pip install jupyter_contrib_nbextensions`
- install into the user’s home jupyter directories : `jupyter contrib nbextension install --user`
- Enable user : `jupyter nbextensions_configurator enable --user`
