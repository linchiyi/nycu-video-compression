import os
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Metric Calculation
# ============================================================================

def calculate_psnr(original, reconstructed):
    """計算 PSNR"""
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(original, reconstructed):
    """計算 SSIM"""
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(original, reconstructed, channel_axis=2 if len(original.shape) == 3 else None, data_range=255)
    except ImportError:
        return None


def calculate_ms_ssim(original, reconstructed):
    """計算 MS-SSIM"""
    try:
        from skimage.metrics import structural_similarity as ssim
        min_dim = min(original.shape[0], original.shape[1])
        if min_dim < 176:
            return None
        
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        levels = min(5, int(np.log2(min_dim)) - 3)
        ms_ssim_value = 1.0
        
        for i in range(levels):
            ssim_val = ssim(original, reconstructed, win_size=3+i*2, channel_axis=2 if len(original.shape) == 3 else None, data_range=255)
            ms_ssim_value *= ssim_val ** weights[i]
        return ms_ssim_value
    except:
        return None


def calculate_lpips(original, reconstructed):
    """計算 LPIPS"""
    try:
        import torch
        import lpips
        loss_fn = lpips.LPIPS(net='alex')
        
        # 轉換為 tensor
        orig_tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        recon_tensor = torch.from_numpy(reconstructed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            return loss_fn(orig_tensor, recon_tensor).item()
    except:
        return None


def format_size(bytes_size):
    """格式化檔案大小"""
    if bytes_size < 1024:
        return f"{bytes_size} bytes"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.2f} KB"
    else:
        return f"{bytes_size / (1024 * 1024):.2f} MB"


# ============================================================================
# Single Image Evaluation
# ============================================================================

def evaluate_single_image(original_path, reconstructed_path, compressed_path, method_name):
    """評估單張影像，返回結果字典"""
    # 讀取影像
    original_img = Image.open(original_path).convert('RGB')
    reconstructed_img = Image.open(reconstructed_path).convert('RGB')
    
    # 轉換為 numpy 並確保尺寸一致
    original_arr = np.array(original_img)
    reconstructed_arr = np.array(reconstructed_img)
    if original_arr.shape != reconstructed_arr.shape:
        reconstructed_img = reconstructed_img.resize(original_img.size, Image.LANCZOS)
        reconstructed_arr = np.array(reconstructed_img)
    
    # 計算檔案大小
    width, height = original_img.width, original_img.height
    original_size = width * height * 3
    compressed_size = os.path.getsize(compressed_path) if compressed_path and os.path.exists(compressed_path) else None
    
    # 計算所有指標
    return {
        'method': method_name,
        'image_name': Path(original_path).stem,
        'reconstructed_name': Path(reconstructed_path).stem,
        'width': width,
        'height': height,
        'psnr': calculate_psnr(original_arr, reconstructed_arr),
        'ssim': calculate_ssim(original_arr, reconstructed_arr),
        'ms_ssim': calculate_ms_ssim(original_arr, reconstructed_arr),
        'lpips': calculate_lpips(original_arr, reconstructed_arr),
        'bpp': (compressed_size * 8) / (width * height) if compressed_size else None,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': original_size / compressed_size if compressed_size else None,
        'original_path': str(original_path),
        'reconstructed_path': str(reconstructed_path),
        'compressed_path': str(compressed_path) if compressed_path else None
    }


def print_result(result):
    """列印單張評估結果"""
    print("\n" + "="*70)
    print(f"評估結果 - {result['method']}")
    print("="*70)
    print(f"原始影像:   {result['original_path']}")
    print(f"重建影像:   {result['reconstructed_path']}")
    print(f"壓縮檔案:   {result['compressed_path']}")
    print(f"影像尺寸:   {result['width']} × {result['height']}")
    print("-"*70)
    print("檔案大小:")
    print(f"  原始大小:     {format_size(result['original_size'])} ({result['original_size']:,} bytes)")
    if result['compressed_size']:
        print(f"  壓縮大小:     {format_size(result['compressed_size'])} ({result['compressed_size']:,} bytes)")
        if result['compression_ratio']:
            print(f"  壓縮率:       {result['compression_ratio']:.2f}x")
            print(f"  節省空間:     {(1 - 1/result['compression_ratio']) * 100:.2f}%")
    print("-"*70)
    print("品質指標:")
    print(f"  PSNR:     {result['psnr']:.4f} dB" if result['psnr'] != float('inf') else "  PSNR:     inf")
    print(f"  SSIM:     {result['ssim']:.6f}" if result['ssim'] else "  SSIM:     N/A")
    print(f"  MS-SSIM:  {result['ms_ssim']:.6f}" if result['ms_ssim'] else "  MS-SSIM:  N/A")
    print(f"  LPIPS:    {result['lpips']:.6f}" if result['lpips'] else "  LPIPS:    N/A")
    print(f"  BPP:      {result['bpp']:.6f}" if result['bpp'] else "  BPP:      N/A")
    print("="*70 + "\n")


# ============================================================================
# Batch Evaluation
# ============================================================================

def batch_evaluate(original_dir, reconstructed_dir, compressed_dir, method_name, output_dir):
    """批次評估所有影像並產生報告"""
    original_dir = Path(original_dir)
    reconstructed_dir = Path(reconstructed_dir)
    compressed_dir = Path(compressed_dir)
    
    reconstructed_files = sorted(list(reconstructed_dir.glob('*')))
    if not reconstructed_files:
        raise ValueError(f"在 {reconstructed_dir} 中找不到重建影像")
    
    print(f"\n找到 {len(reconstructed_files)} 個重建影像，開始評估...")
    
    results = []
    for i, recon_file in enumerate(reconstructed_files, 1):
        print(f"[{i}/{len(reconstructed_files)}] 評估: {recon_file.name}")
        
        # 解析檔名找原始影像
        base_name = recon_file.stem
        for sep in ['_q', '_quality', '_lambda']:
            if sep in base_name:
                base_name = base_name.split(sep)[0]
                break
        
        # 找對應檔案
        original_candidates = list(original_dir.glob(f'{base_name}.*'))
        if not original_candidates:
            print(f"  ⚠️  找不到原始影像: {base_name}")
            continue
        
        compressed_candidates = list(compressed_dir.glob(f'{recon_file.stem}.*'))
        
        try:
            result = evaluate_single_image(
                original_candidates[0],
                recon_file,
                compressed_candidates[0] if compressed_candidates else None,
                method_name
            )
            results.append(result)
        except Exception as e:
            print(f"  ❌ 錯誤: {e}")
    
    print(f"\n✅ 評估完成！")
    
    # 儲存結果
    save_results(results, output_dir)
    plot_rd_curve(results, output_dir, method_name)


def save_results(results, output_dir):
    """儲存 CSV 和摘要"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 儲存 CSV
    csv_path = os.path.join(output_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['image_name', 'reconstructed_name', 'method', 'width', 'height', 'psnr', 'ssim', 'ms_ssim', 'lpips', 'bpp', 'original_size', 'compressed_size', 'compression_ratio']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                'image_name': r['image_name'],
                'reconstructed_name': r['reconstructed_name'],
                'method': r['method'],
                'width': r['width'],
                'height': r['height'],
                'psnr': f"{r['psnr']:.4f}" if r['psnr'] != float('inf') else 'inf',
                'ssim': f"{r['ssim']:.6f}" if r['ssim'] else 'N/A',
                'ms_ssim': f"{r['ms_ssim']:.6f}" if r['ms_ssim'] else 'N/A',
                'lpips': f"{r['lpips']:.6f}" if r['lpips'] else 'N/A',
                'bpp': f"{r['bpp']:.6f}" if r['bpp'] else 'N/A',
                'original_size': r['original_size'],
                'compressed_size': r['compressed_size'] if r['compressed_size'] else 'N/A',
                'compression_ratio': f"{r['compression_ratio']:.4f}" if r['compression_ratio'] else 'N/A'
            })
    
    print(f"詳細結果已儲存至: {csv_path}")
    
    # 計算統計並儲存摘要
    stats = {}
    for metric in ['psnr', 'ssim', 'ms_ssim', 'lpips', 'bpp', 'compression_ratio']:
        valid = [r[metric] for r in results if r[metric] is not None and r[metric] != float('inf')]
        if valid:
            stats[metric] = (np.mean(valid), np.std(valid))
    
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"評估摘要 - {results[0]['method']}\n")
        f.write("="*60 + "\n\n")
        f.write(f"總影像數: {len(results)}\n\n")
        if 'psnr' in stats:
            f.write(f"平均 PSNR:         {stats['psnr'][0]:.4f} ± {stats['psnr'][1]:.4f} dB\n")
        if 'ssim' in stats:
            f.write(f"平均 SSIM:         {stats['ssim'][0]:.6f} ± {stats['ssim'][1]:.6f}\n")
        if 'ms_ssim' in stats:
            f.write(f"平均 MS-SSIM:      {stats['ms_ssim'][0]:.6f} ± {stats['ms_ssim'][1]:.6f}\n")
        if 'lpips' in stats:
            f.write(f"平均 LPIPS:        {stats['lpips'][0]:.6f} ± {stats['lpips'][1]:.6f}\n")
        if 'bpp' in stats:
            f.write(f"平均 BPP:          {stats['bpp'][0]:.6f} ± {stats['bpp'][1]:.6f}\n")
        if 'compression_ratio' in stats:
            f.write(f"平均壓縮率:        {stats['compression_ratio'][0]:.4f}x ± {stats['compression_ratio'][1]:.4f}x\n")
        f.write("\n" + "="*60 + "\n")
    
    print(f"摘要統計已儲存至: {summary_path}")

def plot_rd_curve(results, output_dir, method_name):
    """繪製 Rate-Distortion 曲線"""
    images = {}
    for r in results:
        if r['image_name'] not in images:
            images[r['image_name']] = []
        images[r['image_name']].append(r)
    
    has_multi_points = any(len(points) > 1 for points in images.values())
    
    if has_multi_points:
        plt.figure(figsize=(12, 8))
        for img_name, points in images.items():
            if len(points) < 2:
                continue
            points = sorted(points, key=lambda x: x['bpp'] if x['bpp'] else 0)
            bpp = [p['bpp'] for p in points if p['bpp']]
            psnr = [p['psnr'] for p in points if p['psnr'] and p['psnr'] != float('inf')]
            if len(bpp) >= 2:
                plt.plot(bpp, psnr, marker='o', label=img_name, alpha=0.7)
        
        plt.xlabel('BPP (Bits Per Pixel)', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.title(f'Rate-Distortion Curves - {method_name}', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        valid = [r for r in results if r['bpp'] and r['psnr']]
        if not valid:
            return
        plt.figure(figsize=(10, 6))
        plt.scatter([r['bpp'] for r in valid], [r['psnr'] for r in valid], s=100, alpha=0.6)
        plt.xlabel('BPP (Bits Per Pixel)', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.title(f'Rate-Distortion - {method_name}', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    rd_path = os.path.join(output_dir, 'rd_curve.png')
    plt.savefig(rd_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rate-Distortion 曲線已儲存至: {rd_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='統一評估模組 - 單張或批次評估')
    parser.add_argument('--original', type=str, required=True, help='原始影像路徑或資料夾')
    parser.add_argument('--reconstructed', type=str, required=True, help='重建影像路徑或資料夾')
    parser.add_argument('--compressed', type=str, required=True, help='壓縮檔案路徑或資料夾')
    parser.add_argument('--method', type=str, required=True, help='方法名稱 (JPEG, Cheng2020, HiFiC, MLIC)')
    parser.add_argument('--batch', action='store_true', help='批次模式')
    parser.add_argument('--output', type=str, help='輸出資料夾（批次模式必須）')
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.output:
            parser.error("批次模式需要指定 --output 資料夾")
        
        print("\n" + "="*70)
        print("批次評估模式")
        print("="*70)
        print(f"方法: {args.method}")
        print(f"原始影像: {args.original}")
        print(f"重建影像: {args.reconstructed}")
        print(f"壓縮檔案: {args.compressed}")
        print(f"輸出路徑: {args.output}")
        print("="*70)
        
        batch_evaluate(args.original, args.reconstructed, args.compressed, args.method, args.output)
    else:
        result = evaluate_single_image(args.original, args.reconstructed, args.compressed, args.method)
        print_result(result)


if __name__ == '__main__':
    main()