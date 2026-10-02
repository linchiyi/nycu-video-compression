# 視訊壓縮課程作業

本 repository 整理視訊壓縮課程中的程式作業與期末專題，內容涵蓋影像色彩表示、DCT、動態估測、熵編碼與學習式影像壓縮。

## 作業總覽

| 作業 | 主題 | 主要內容 |
| --- | --- | --- |
| [HW1｜色彩空間轉換](./01-色彩空間轉換/) | RGB、YUV、YCbCr | 以公式完成色彩轉換，輸出八張灰階影像 |
| [HW2｜二維離散餘弦轉換](./02-二維離散餘弦轉換/) | 2D-DCT / IDCT | 比較直接二維公式與可分離的兩次一維 DCT |
| [HW3｜動態估測與補償](./03-動態估測與補償/) | Motion Estimation | 實作 FSBM 與 TSS，完成運動補償與 PSNR 分析 |
| [HW4｜熵編碼](./04-熵編碼/) | JPEG-like pipeline | 串接 DCT、量化、Zigzag、RLE、反量化與 IDCT |
| [期末專題｜影像壓縮模型比較](./05-期末專題/) | JPEG、Cheng2020、MLIC++、HiFiC | 以 PSNR、SSIM、LPIPS、BPP 比較不同壓縮方法 |

## 技術

- Python
- NumPy、OpenCV、Pillow、SciPy、Matplotlib
- DCT / IDCT、量化、Zigzag scan、Run-Length Encoding
- Full Search Block Matching、Three-Step Search
- PSNR、SSIM、LPIPS、Rate–Distortion analysis

## 公開版本說明

本 repository 保留程式碼、範例輸入輸出與實驗統計；未放入含姓名或學號的報告 PDF、影片、模型權重、虛擬環境、壓縮 bitstream 與大型產出檔。

期末專題中的 high-fidelity-generative-compression 保留原專案的 LICENSE 與 README；其中部分程式碼來自既有開源專案，使用時請遵守原專案授權與引用要求。

