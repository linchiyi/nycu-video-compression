import os
import argparse
from pathlib import Path
from PIL import Image

def compress_images(input_dir, output_dir, quality_levels):
    """
    壓縮影像資料夾中的所有影像
    
    Args:
        input_dir: 輸入影像資料夾
        output_dir: 輸出資料夾
        quality_levels: JPEG 品質等級列表
    """
    # 建立輸出資料夾
    compressed_dir = os.path.join(output_dir, 'compressed')
    reconstructed_dir = os.path.join(output_dir, 'reconstructed')
    os.makedirs(compressed_dir, exist_ok=True)
    os.makedirs(reconstructed_dir, exist_ok=True)
    
    # 取得所有影像檔案
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    image_files = sorted(image_files)
    
    if not image_files:
        raise ValueError(f"在 {input_dir} 中找不到影像檔案")
    
    print(f"找到 {len(image_files)} 張影像")
    print(f"品質等級: {quality_levels}")
    print(f"總共處理: {len(image_files) * len(quality_levels)} 次\n")
    
    total = len(image_files) * len(quality_levels)
    current = 0
    
    for img_path in image_files:
        # 讀取影像
        img = Image.open(img_path).convert('RGB')
        img_name = img_path.stem
        
        for quality in quality_levels:
            current += 1
            print(f"[{current}/{total}] {img_name} (quality={quality})")
            
            # 定義輸出路徑
            compressed_path = os.path.join(compressed_dir, f'{img_name}_q{quality}.jpg')
            reconstructed_path = os.path.join(reconstructed_dir, f'{img_name}_q{quality}.png')
            
            # 壓縮
            img.save(compressed_path, 'JPEG', quality=quality, optimize=True)
            
            # 重建（讀取壓縮檔案並存成 PNG）
            reconstructed = Image.open(compressed_path)
            reconstructed.save(reconstructed_path, 'PNG')
    
    print(f"\n✅ 完成！")
    print(f"壓縮檔案: {compressed_dir}")
    print(f"重建檔案: {reconstructed_dir}")


def main():
    parser = argparse.ArgumentParser(description='JPEG 圖像壓縮')
    parser.add_argument('--input', type=str, required=True, help='輸入影像資料夾')
    parser.add_argument('--output', type=str, required=True, help='輸出資料夾')
    parser.add_argument('--quality', type=int, nargs='+', default=[10, 30, 50, 70, 90], help='JPEG 品質等級 (預設: 10 30 50 70 90)')
    args = parser.parse_args()
    
    compress_images(args.input, args.output, args.quality)


if __name__ == '__main__':
    main()