# HW4｜熵編碼

## 作業目標

以灰階 lena.png 為輸入，實作一個簡化的 JPEG-like 壓縮流程：

1. 將影像切成 8×8 區塊。
2. 執行二維 DCT。
3. 使用量化表量化係數。
4. 以 Zigzag 順序掃描。
5. 使用 Run-Length Encoding（RLE）編碼與解碼。
6. 反量化並執行 IDCT，重建影像。

程式比較兩組量化表的壓縮結果，並輸出重建影像與比較圖。

## 執行環境

~~~
pip install numpy pillow scipy matplotlib
python3 entropy_coding.py
~~~

主要輸出：

- recon_q1.png：量化表 1 的重建影像
- recon_q2.png：量化表 2 的重建影像
- comparison.png：原圖與重建結果比較

