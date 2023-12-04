import cv2
import numpy as np
import math


# DCT block 8x8
def dct_block(array):
    result = np.zeros_like(array, dtype=float)

    # DCT theo hàng
    for i in range(8):
        for u in range(8):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            sum_val = 0
            for v in range(8):
                sum_val += array[i][v] * np.cos((2 * v + 1) * np.pi * u / 16)
            result[i][u] = sum_val * cu * 1 / 2

    # Tạo một mảng tạm thời để lưu trữ kết quả DCT theo hàng
    temp_result = np.zeros_like(result, dtype=float)

    # DCT theo cột từ mảng tạm thời
    for j in range(8):
        for u in range(8):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            sum_val = 0
            for v in range(8):
                sum_val += result[v][j] * np.cos((2 * v + 1) * np.pi * u / 16)
            temp_result[j][u] = sum_val * cu * 1 / 2

    return temp_result


# IDCT block 8x8
def idct_block(array):
    reconstruction = np.zeros_like(array, dtype=float)

    # IDCT theo hàng
    for i in range(8):
        for v in range(8):
            sum_val = 0
            for u in range(8):
                cu = 1 / np.sqrt(2) if u == 0 else 1
                sum_val += array[i][u] * cu * np.cos((2 * v + 1) * np.pi * u / 16)
            sum_val *= 1 / 2
            reconstruction[i][v] = sum_val

    # IDCT theo cột
    for j in range(8):
        for v in range(8):
            sum_val = 0
            for u in range(8):
                cu = 1 / np.sqrt(2) if u == 0 else 1
                sum_val += reconstruction[u][j] * cu * np.cos((2 * v + 1) * np.pi * u / 16)
            sum_val *= 1 / 2
            reconstruction[v][j] = sum_val

    return reconstruction