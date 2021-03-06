import math

WHITE = 1
BLACK = 0

#This method returns the prefix sum array for a given 2D array
def prefix_sum_array (array):
    prefix_sum = [[0 for x in range(len(array[0])+1) ] for y in range(len(array)+1)]
    for a in range(1, len(array)+1):
        for b in range(1, len(array[a-1])+1):
            prefix_sum[a][b] = prefix_sum[a-1][b] + prefix_sum[a][b-1] - prefix_sum[a-1][b-1] + array[a-1][b-1]
    return prefix_sum

#This method returns the sum of a given block in a 2D array
def find_sum (prefix_sum, left_row, left_coloumn, right_row, right_coloumn):
    if left_row >= right_row or left_coloumn >= right_coloumn:
        return 0

    p_sum = prefix_sum[left_row][left_coloumn] + prefix_sum[right_row][right_coloumn]
    p_sum -= prefix_sum[right_row][left_coloumn] + prefix_sum[left_row][right_coloumn]
    return p_sum

#This method finds the average of a given block in a 2D array
def average (array_2d, start_x, end_x, start_y, end_y):
    total = 0
    count = 0
    for x in range(start_x, end_x):
        for y in range(start_y, end_y):
            total += array_2d[y][x]
            count += 1
    return total/count

def percent_white (img_array):
    total_squares = 0
    white_squares = 0
    for x in img_array:
        for a in range(len(x)):
            total_squares += 1
            if x[a] == WHITE:
                white_squares += 1
    return white_squares/total_squares

#This method counts the number of white pixels in a given 2D prefix sum array and rectangle
def cnt_white (prefix_sum, left_row, left_coloumn, right_row, right_coloumn):
    '''num_white calculated by (all black matrix - actual matrix = only white matrix with entries BLACK-WHITE)
                EG: |B B|   |B W|                            |0 B-W|
                    |B B| - |B B| = |all black| - |actual| = |0 0  |
                '''
    p_s = find_sum (prefix_sum, left_row, left_coloumn, right_row, right_coloumn)
    n_w = (c_size*c_size*(BLACK) - p_s)/(BLACK-WHITE)
    return n_w

#This method returns an image converted to black and white by finding the middle colour and considering everything above the middle to be white
def to_black_white (img_array):
    min = img_array[0][0]
    max = img_array[0][0]
    for x in img_array:
        for a in range(len(x)):
            min = x[a] if x[a] < min else min
            max = x[a] if x[a] > max else max

    whiteness = 0.45
    border = (min + max)/2*whiteness

    img_cpy = [[0 for x in range(len(img_array[0]))] for y in range(len(img_array))]
    for a in range(len(img_array)):
        for b in range(len(img_array[a])):
            img_cpy[a][b] = WHITE if img_array[a][b] >= border else BLACK
    return img_cpy

#This method compresses an image to 50x50 size
def compress_to_25x25 (img_array):
    length = len(img_array[0])
    height = len(img_array)
    arr = [[0 for x in range(25)] for y in range(25)]
    if length < 25 or height < 25:
        return img_array

    prev_x = 0
    prev_y = 0
    curr_x = 0
    curr_y = 0
    for x in range(1, 51):
        curr_x = length*x//50
        for y in range(1, 51):
            prev_y = curr_y
            curr_y = height*y//50
            arr[y-1][x-1] = average(img_array, prev_x, curr_x, prev_y, curr_y)

        prev_x = curr_x
        curr_y = 0
    return arr
    prev_x = 0
    prev_y = 0
    curr_x = 0
    curr_y = 0
    for x in range(1, 51):
        curr_x = length*x//50
        for y in range(1, 51):
    for x in range(1, 26):
        curr_x = length*x//25
        for y in range(1, 26):
            prev_y = curr_y
            curr_y = height*y//25
            arr[y-1][x-1] = average(img_array, prev_x, curr_x, prev_y, curr_y)

        prev_x = curr_x
        curr_y = 0
    return arr
