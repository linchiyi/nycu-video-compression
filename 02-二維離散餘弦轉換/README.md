# HW2｜二維離散餘弦轉換

## 作業目標

不依賴現成 DCT 函式，完成 2D-DCT 與 2D-IDCT，並比較兩種實作方式：

1. 直接依照二維公式，以巢狀迴圈計算。
2. 利用 DCT 的可分離性，先對每一列、再對每一行執行一維 DCT。

程式同時比較兩種方法的執行時間、重建影像與 PSNR，並產生係數視覺化結果。

## 執行環境

~~~
pip install numpy opencv-python matplotlib tqdm
python3 dct.py
~~~

程式會互動式詢問分析尺寸；影像尺寸較大時，可選擇是否執行計算量較高的直接二維 DCT。

