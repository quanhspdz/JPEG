import numpy as np

##  Zig-zag
block_size = 8
def zig_zag(input_matrix, block_size):
    z = np.empty([block_size * block_size])
    index = -1
    bound = 0
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                z[index] = input_matrix[j, i - j]
            else:
                z[index] = input_matrix[i - j, j]
    return z


def zig_zag_reverse(array):
    result = [[0] * block_size for _ in range(block_size)]
    index = 0

    for i in range(block_size + block_size - 1):
        if i % 2 == 0:  # đường chéo lên
            row = min(i, block_size - 1)
            col = max(0, i - block_size + 1)
            while row >= 0 and col < block_size:
                result[row][col] = array[index]
                index += 1
                row -= 1
                col += 1
        else:  # Đường chéo xuống
            col = min(i, block_size - 1)
            row = max(0, i - block_size + 1)
            while col >= 0 and row < block_size:
                result[row][col] = array[index]
                index += 1
                col -= 1
                row += 1
    return result