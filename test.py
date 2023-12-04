from core import *
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
import time
import pandas as pd
from calculate import *




directory = './data/'
sorted_image_files = []
mse_values = []
psnr_values = []
ssim_values = []
compression_ratio_values = []
bpp_values = []
sorted_psnr_values = []
sorted_bpp_values = []
sorted_ssim_values = []
sorted_comratio_values = []
image_names = []
time_compress = []
sorted_time_compress = []

image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# image_files = ['vidu_1.png']

for image_file in image_files:
    # Construct the full file path
    filepath = os.path.join(directory, image_file)

    # Read the image
    image = cv2.imread(filepath)

    
    start = time.time()
    result = np.zeros_like(image)
    for i in range(3):
        result[:,:,i] = compress_channel(image[:,:,i])
    stop = time.time()  
    
    
    # print(img)
    # Chuyển đổi ảnh YCbCr lại thành ảnh màu
    # new_img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    new_img = cv2.cvtColor(result,cv2.COLOR_YCrCb2RGB)
    # Hiển thị ảnh gốc và ảnh giải nén
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Original Image')
    # plt.subplot(122), plt.imshow(new_img), plt.axis('off'), plt.title('Reconstructed Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Recovered Image')
    plt.show()

    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    # Lưu ảnh sau khi giải nén
    cv2.imwrite('decompressed.jpg', new_img)
    
    mse = MSE(image, new_img)
    psnr = PSNR(mse)
    ssim = SSIM(image, new_img)
    compression_ratio = Compression_Ratio(filepath)
    bpp = calculate_bpp(image, compression_ratio)
    image_files = [os.path.basename(filepath) for filepath in image_files]
    
    mse_values.append(mse)
    psnr_values.append(psnr)
    ssim_values.append(ssim)
    time_compress.append(stop-start)
    compression_ratio_values.append(compression_ratio)
    bpp_values.append(bpp)
    
    sorted_indices = np.argsort(bpp_values)
    sorted_bpp_values = np.array(bpp_values)[sorted_indices]
    sorted_psnr_values = np.array(psnr_values)[sorted_indices]
    sorted_ssim_values = np.array(ssim_values)[sorted_indices]
    sorted_comratio_values = np.array(compression_ratio_values)[sorted_indices]
    sorted_image_files = np.array(image_files)[sorted_indices]
    sorted_time_compress = np.array(time_compress)[sorted_indices]

    print(filepath)
    print('Time Compress: ',stop - start)
    print('Compression Ratio: ', compression_ratio)
    print("loading...")

data = {
'Image': sorted_image_files,
'BPP':  sorted_bpp_values,
'PSNR': sorted_psnr_values,
'SSIM': sorted_ssim_values,
'Compression Ratio': sorted_comratio_values,
'Time Compress' : sorted_time_compress,
}

pd.set_option('display.max_columns', None)
df = pd.DataFrame(data)
print(df)

sum = np.sum(sorted_time_compress)
print("Sum of time", sum)

plt.figure(figsize=(8, 6))
plt.plot(sorted_bpp_values, sorted_psnr_values, linestyle='-', marker='o', color='green', label='Line of Best Fit')
plt.title('PSNR vs Bits per Pixel (BPP) for Image Reconstruction')
plt.xlabel('Bits per Pixel (BPP)')
plt.ylabel('PSNR')
plt.grid(True)
plt.legend()
plt.show()
    