# 視訊壓縮課程作業

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


