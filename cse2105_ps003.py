###########################################################################
# HEADER COMMENTS (5 pts)
###########################################################################
### __filename__ = "cse2105_ps003.py"
### __Python version__ = "3.10.4"
### __author__ = "이세원"
### __copyright__ = "Copyright 2022, CSE2015 Problem Set 03"
### __credits__ = ["Jieung Kim"]
### __license__ = "GPL"
### __version__ = "1.0.0"
### __maintainer__ = "이세원"
### __email__ = "dltpdnjs2000@gmail.com"
###########################################################################

from typing import Optional
import copy
import random
import time
import sys


###########################################################################
# determinant_with_cofactor (25 pts)
###########################################################################
def determinant_with_cofactor(matrix: list[list[int]]) -> Optional[int]:
    """
    Add your explanations about this function:
    여인수 전개를 통하여 행렬식 계산하기

    param matrix:
    2차원 int 리스트 (행렬)

    return:
    입력으로 들어온 행렬의 행렬식 값
    """

    # ill-formed 예외처리
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        return None
    if len(matrix) == 1:  # 1x1 matrix
        return matrix[0][0]
    if len(matrix) == 2:  # 2x2 case
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    dim = len(matrix)  # dimension
    det = 0

    for col in range(dim):  # loop
        a = matrix[0][col]  # 필요 원소 추출
        partial = [row[:col] + row[col + 1:] for row in matrix[1:]]  # 소 행렬식 만들기
        minor_matrix = determinant_with_cofactor(partial)  # minor_matrix 로 재귀 진행
        flag = (-1) ** (col)  # sign flag 설정
        det += a * flag * minor_matrix  # 결과값 계산
    return det


###########################################################################
# determinant_with_gauss (25 pts)
###########################################################################

def determinant_with_gauss(matrix: list[list[int]]) -> Optional[int]:
    """
    Add your explanations about this function
    : 가우스 조던 소거법을 이용하여 행렬식 계산하기

    param matrix
    : 2차원 int 리스트 (행렬)

    return
    : 입력으로 들어온 행렬의 행렬식 값
    """

    # ill-formed 예외처리
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        return None
    if len(matrix) == 1:  # 1x1 matrix
        return matrix[0][0]
    if len(matrix) == 2:  # 2x2 case
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    dim = len(matrix)  # dimension
    det = 1

    row_count = 0
    for col in range(dim):  # 각 열에 대해 수행

        # 첫 행 지정
        for i in range(row_count, dim):
            if matrix[i][col] != 0:  # 첫 열이 0이 아닌 첫 행을 추출
                first_row = i  # 첫 행으로 사용
                break
        else:  # 모든 원소가 0인 경우
            return 0

        # 첫 행 첫 열 1로 처리
        if matrix[first_row][col] != 1:  # 첫 열의 원소가 1이 아닌 경우
            mul_val = matrix[first_row][col]
            matrix[first_row] = [elem / mul_val for elem in matrix[first_row]]  # 첫 행의 첫 열을 1로 변환
            det *= mul_val  # update det

        # 행 교환
        if first_row != row_count:
            det *= -1  # 행 교환시 *-1 처리 (값 변경 방지)
            matrix[first_row], matrix[row_count] = matrix[row_count], matrix[first_row]

        # 상수배 후 덧셈
        for j in range(row_count + 1, dim):
            if matrix[j][col] != 0:
                scaling_val = matrix[j][col]
                matrix[j] = [elem - scaling_val * matrix[row_count][idx] for idx, elem in enumerate(matrix[j])]

        row_count += 1

    return round(det, 2)


##########################################################################
# determinant_with_cofactor_benchmarks (12.5 pts)
##########################################################################
def determinant_with_cofactor_benchmarks(matrix: list[list[int]]) -> tuple[Optional[int], int, int]:
    """
    Add your explanations about this function:
    여인수 전개를 통하여 행렬식 계산 및 계산과정 중 덧셈연산과 곱셈연산 횟수 계산

    param matrix:
    2차원 int 리스트 (행렬)

    return:
    행렬의 행렬식, 덧셈 연산 횟수, 곱셈 연산 횟수
    """

    # ill-formed 예외처리
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        return None

    # 재귀적으로 행렬의 행렬식 및 덧셈, 곱셈 연산횟수 계산
    def cal(matrix: list[list[int]]) -> tuple[int, int, int]:

        dim = len(matrix)  # dimension
        add_count = 0  # 덧셈 연산 횟수
        mul_count = 0  # 곱셈 연산 횟수
        det = 0  # 행렬식

        if dim == 1:  # 1x1 case
            return matrix[0][0], 0, 0
        if dim == 2:  # 2x2 case
            return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1], 1, 2
        else:
            # 소 행렬식을 만들고, 재귀적으로 계산
            for col in range(dim):  # 각 열에 대해
                a = matrix[0][col]  # 첫 행 원소

                # 소 행렬식 만들기
                partial = [row[:col] + row[col + 1:] for row in matrix[1:]]

                minor_matrix, add_num, mul_num = cal(partial)  # 재귀적 계산

                sign_flag = (-1) ** (col)  # 부호 결정
                det += a * sign_flag * minor_matrix  # update det
                add_count += add_num + 1  # update add cnt
                mul_count += mul_num + 2  # update mul cnt

            return det, add_count, mul_count

    det, add_count, mul_count = cal(matrix)  # 결과값

    return det, add_count, mul_count


##########################################################################
# determinant_with_gauss_benchmarks (12.5 pts)
##########################################################################
def determinant_with_gauss_benchmarks(matrix: list[list[int]]) -> tuple[Optional[int], int, int]:
    """
    Add your explanations about this function
    : 가우스 조던법을 통하여 행렬식 계산 및 계산과정 중 덧셈연산과 곱셈연산 횟수 계산

    param matrix
    : 2차원 int 리스트 (행렬)

    return
    : 행렬의 행렬식, 덧셈 연산 횟수, 곱셈 연산 횟수
    """

    # ill-formed 예외처리
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        return None
    if len(matrix) == 1:  # 1x1 matrix
        return matrix[0][0], 0, 0
    if len(matrix) == 2:  # 2x2 case
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1], 1, 2

    dim = len(matrix)  # dimension
    add_count = 0  # 덧셈 연산 횟수
    mul_count = 0  # 곱셈 연산 횟수
    det = 1  # 행렬식

    row_count = 0
    for col_count in range(dim):  # 각 열에 대해
        # 첫 열이 0이 아닌 첫 행을 추출
        for i in range(row_count, dim):
            if matrix[i][col_count] != 0:
                first_row = i  # 첫 행으로 사용
                break
        else:  # 모든 원소가 0인 경우
            return 0, add_count, mul_count

        # 첫 행 첫 열 1로 처리
        if matrix[first_row][col_count] != 1:  # 첫 열의 원소가 1이 아닌 경우
            mul_val = matrix[first_row][col_count]
            matrix[first_row] = [elem / mul_val for elem in matrix[first_row]]
            det *= mul_val  # update det
            mul_count += dim  # update mul cnt

        # 행 교환
        if first_row != row_count:
            det *= -1  # 행 교환시 *-1 처리 (값 변경 방지)
            matrix[first_row], matrix[row_count] = matrix[row_count], matrix[first_row]  # 행 교환
            add_count += dim  # update add cnt

        # 상수배 후 덧셈
        for j in range(row_count + 1, dim):
            if matrix[j][col_count] != 0:
                scaling_val = matrix[j][col_count]
                matrix[j] = [elem - scaling_val * matrix[row_count][idx] for idx, elem in enumerate(matrix[j])]
                add_count += dim  # update add cnt
                mul_count += dim  # update mul cnt

        row_count += 1

    return round(det, 2), add_count, mul_count


##########################################################################
# run_benchmarks (15 pts)
##########################################################################
from numpy.linalg import det


def run_benchmarks():
    """
    Add your explanations about this function
    : 다양한 크기의 행렬에서 여인수 전개 방식(1), 가우스 조던 방식(2), numpy library(3)의 속도 및 연산 수 비교
    """

    matrix_sizes = range(3, 11)  # 3x3 ~ 10x10 까지 테스트

    for size in matrix_sizes:  # 각 크기에 대해서 loop
        # 결과 담을 변수 초기화
        cofactor_times = []
        gauss_times = []
        cofactor_add_counts = []
        gauss_add_counts = []
        cofactor_mul_counts = []
        gauss_mul_counts = []

        # 각 함수 10회씩 실행
        for _ in range(10):
            # 랜덤 숫자로 초기화
            random.seed(size)
            matrix = [[random.randint(0, 7) for _ in range(size)] for _ in range(size)]

            # cofactor
            start_time = time.time()
            det_val, add_count, mul_count = determinant_with_cofactor_benchmarks(matrix)
            end_time = time.time()

            # cofactor 결과 저장
            cofactor_times.append(end_time - start_time)
            cofactor_add_counts.append(add_count)
            cofactor_mul_counts.append(mul_count)

            # gauss
            start_time = time.time()
            det_val, add_count, mul_count = determinant_with_gauss_benchmarks(matrix)
            end_time = time.time()

            # gauss 결과 저장
            gauss_times.append(end_time - start_time)
            gauss_add_counts.append(add_count)
            gauss_mul_counts.append(mul_count)

        # 평균값 계산
        avg_cofactor_time = sum(cofactor_times) / len(cofactor_times)
        avg_cofactor_add_count = sum(cofactor_add_counts) / len(cofactor_add_counts)
        avg_cofactor_mul_count = sum(cofactor_mul_counts) / len(cofactor_mul_counts)

        avg_gauss_time = sum(gauss_times) / len(gauss_times)
        avg_gauss_add_count = sum(gauss_add_counts) / len(gauss_add_counts)
        avg_gauss_mul_count = sum(gauss_mul_counts) / len(gauss_mul_counts)

        # numpy
        det_times = []
        for _ in range(10):  # 10회 수행
            start_time = time.time()
            det(matrix)
            end_time = time.time()
            det_times.append(end_time - start_time)
        # 평균값 저장
        avg_det_time = sum(det_times) / len(det_times)

        # 결과 출력
        print(f"Matrix size: {size} x {size}")
        print("  Using Cofactor expansion:")
        print(f"    Average time (sec): {avg_cofactor_time}")
        print(f"    Average num. of additions: {avg_cofactor_add_count}")
        print(f"    Average num. of multiplications: {avg_cofactor_mul_count}")
        print(f"    Average num. of cycles: {avg_cofactor_add_count + 3 * avg_cofactor_mul_count}")
        print("  Using Gauss elimination:")
        print(f"    Average time (sec): {avg_gauss_time}")
        print(f"    Average num. of additions: {avg_gauss_add_count}")
        print(f"    Average num. of multiplications: {avg_gauss_mul_count}")
        print(f"    Average num. of cycles: {avg_gauss_add_count + 3 * avg_gauss_mul_count}")
        print("  Using numpy library (numpy.linalg.det):")
        print(f"    Average time (sec): {avg_det_time}")
        print()


##########################################################################
# main functions. Please do not change the following two functions
##########################################################################
def main() -> int:
    """
    Main function that runs run_benchmarks function in it.
    """
    run_benchmarks()
    return 0


if __name__ == '__main__':
    sys.exit(main())

###########################################################################
# TABLE (5 pts)
###########################################################################
# Matrix size: 3 x 3
#   Using Cofactor expansion:
#     Average time (sec): 0.0
#     Average num. of additions: 6.0
#     Average num. of multiplications: 12.0
#     Average num. of cycles: 42.0
#   Using Gauss elimination:
#     Average time (sec): 0.0
#     Average num. of additions: 9.0
#     Average num. of multiplications: 18.0
#     Average num. of cycles: 63.0
#   Using numpy library (numpy.linalg.det):
#     Average time (sec): 0.0
#
# Matrix size: 4 x 4
#   Using Cofactor expansion:
#     Average time (sec): 0.0
#     Average num. of additions: 28.0
#     Average num. of multiplications: 56.0
#     Average num. of cycles: 196.0
#   Using Gauss elimination:
#     Average time (sec): 0.0
#     Average num. of additions: 20.0
#     Average num. of multiplications: 36.0
#     Average num. of cycles: 128.0
#   Using numpy library (numpy.linalg.det):
#     Average time (sec): 0.0
#
# Matrix size: 5 x 5
#   Using Cofactor expansion:
#     Average time (sec): 0.0
#     Average num. of additions: 145.0
#     Average num. of multiplications: 290.0
#     Average num. of cycles: 1015.0
#   Using Gauss elimination:
#     Average time (sec): 0.0
#     Average num. of additions: 45.0
#     Average num. of multiplications: 70.0
#     Average num. of cycles: 255.0
#   Using numpy library (numpy.linalg.det):
#     Average time (sec): 0.0
#
# Matrix size: 6 x 6
#   Using Cofactor expansion:
#     Average time (sec): 0.000921320915222168
#     Average num. of additions: 876.0
#     Average num. of multiplications: 1752.0
#     Average num. of cycles: 6132.0
#   Using Gauss elimination:
#     Average time (sec): 0.0
#     Average num. of additions: 90.0
#     Average num. of multiplications: 120.0
#     Average num. of cycles: 450.0
#   Using numpy library (numpy.linalg.det):
#     Average time (sec): 0.0
#
# Matrix size: 7 x 7
#   Using Cofactor expansion:
#     Average time (sec): 0.007901477813720702
#     Average num. of additions: 6139.0
#     Average num. of multiplications: 12278.0
#     Average num. of cycles: 42973.0
#   Using Gauss elimination:
#     Average time (sec): 0.00010027885437011719
#     Average num. of additions: 140.0
#     Average num. of multiplications: 189.0
#     Average num. of cycles: 707.0
#   Using numpy library (numpy.linalg.det):
#     Average time (sec): 0.0
#
# Matrix size: 8 x 8
#   Using Cofactor expansion:
#     Average time (sec): 0.09028942584991455
#     Average num. of additions: 49120.0
#     Average num. of multiplications: 98240.0
#     Average num. of cycles: 343840.0
#   Using Gauss elimination:
#     Average time (sec): 4.978179931640625e-05
#     Average num. of additions: 224.0
#     Average num. of multiplications: 288.0
#     Average num. of cycles: 1088.0
#   Using numpy library (numpy.linalg.det):
#     Average time (sec): 0.0
#
# Matrix size: 9 x 9
#   Using Cofactor expansion:
#     Average time (sec): 0.49344649314880373
#     Average num. of additions: 442089.0
#     Average num. of multiplications: 884178.0
#     Average num. of cycles: 3094623.0
#   Using Gauss elimination:
#     Average time (sec): 0.0
#     Average num. of additions: 297.0
#     Average num. of multiplications: 378.0
#     Average num. of cycles: 1431.0
#   Using numpy library (numpy.linalg.det):
#     Average time (sec): 0.0
#
# Matrix size: 10 x 10
#   Using Cofactor expansion:
#     Average time (sec): 5.176206564903259
#     Average num. of additions: 4420900.0
#     Average num. of multiplications: 8841800.0
#     Average num. of cycles: 30946300.0
#   Using Gauss elimination:
#     Average time (sec): 0.0
#     Average num. of additions: 420.0
#     Average num. of multiplications: 510.0
#     Average num. of cycles: 1950.0
#   Using numpy library (numpy.linalg.det):
#     Average time (sec): 0.0
###########################################################################
