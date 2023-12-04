import numpy as np
import os
from math import log10
from skimage.metrics import structural_similarity as ssim

def MSE(img1, img2):
    print("img1 shape:", img1.shape)
    print("img2 shape:", img2.shape)
    # Ensure both images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for MSE calculation.")

    # Calculate MSE
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

    return mse


def PSNR(mse):
    if mse == 0:
        return float('inf')  # Return infinity for perfect reconstruction

    return 10 * log10((255 * 255) / mse)

def SSIM(img1, img2):
    # Chuyển đổi sang kiểu float64 để tính toán chính xác
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Tính toán SSIM cho từng kênh màu và trung bình kết quả
    ssim_values = []
    for channel in range(img1.shape[-1]):
        channel_ssim = ssim(img1[:, :, channel], img2[:, :, channel], data_range=img2.max() - img2.min())
        ssim_values.append(channel_ssim)

    # Trả về giá trị SSIM trung bình của tất cả các kênh
    return np.mean(ssim_values)


def Compression_Ratio(filepath):
    Ori_img = os.stat(filepath).st_size
    Ori_img = Ori_img / 1024
    Com_img = os.path.getsize('decompressed.jpg')
    Com_img = Com_img / 1024
    CR = Ori_img / float(Com_img)
    return CR

def calculate_ssim(img1, img2):
    # Ensure both images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for SSIM calculation.")

    # Specify an appropriate window size (odd value)
    win_size = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    win_size = win_size if win_size % 2 == 1 else win_size - 1

    # Calculate SSIM with the specified window size and considering three channels
    ssim_value, _ = ssim(img1.astype(np.float64), img2.astype(np.float64), win_size=win_size, data_range=img2.max() - img2.min(), multichannel=True, channel_axis=2)

    return ssim_value

def calculate_bpp(image, compression_ratio):
    # Assuming image.dtype is uint8 (8 bits per channel)
    bpp = (8 * image.size) / (compression_ratio * image.shape[0] * image.shape[1])
    return bpp