# HW1｜色彩空間轉換

## 作業目標

將 lena.png 以公式轉換為 RGB、YUV 與 YCbCr，並輸出八張灰階影像：

- R.png、G.png、B.png
- Y.png、U.png、V.png
- Cb.png、Cr.png

本作業不使用現成的色彩轉換函式；影像讀寫使用 OpenCV，數值處理使用 NumPy。

## 執行方式

~~~
pip install numpy opencv-python
python3 color_transform.py
~~~

程式會讀取同一資料夾內的 lena.png，並將結果寫入 outputs/。

