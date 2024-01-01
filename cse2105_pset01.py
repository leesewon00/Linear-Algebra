###########################################################################
# HEADER COMMENTS
###########################################################################
### __filename__ = "cse2105.py"
### __Python version__ = "3.11.0"
### __copyright__ = "Copyright 2023, CSE2105 Problem Set 01"
### __credits__ = ["Jieung Kim", "XXX"]
### __license__ = "GPL"
### __version__ = "1.0.0"
###########################################################################

"""
Please keep in mind the following items for all questions:
1. To work with valid input/output pairs, refer to the
   test cases provided in `cse2105_pset01_test.py`.
2. There are no hidden test cases. So, if you pass all test cases in
   `cse2105_pset01_test.py`, you will get the full credit
   (type `python3 cse2105_pset01_test.py`).
3. Please implement the function efficiently and consider additional helper
   functions if needed.
4. While you are free to implement additional functions, DO NOT CHANGE
   SIGNATURES OF TOP-LEVEL FUNCTIONS. YOU WILL GET NO POINTS IF YOU
   CHANGE SIGNATURES OF TOP-LEVEL FUNCTIONS.
5. DO NOT IMPORT ANY ADDITIONAL PACKAGES (e.g., numpy, scipy) IN THIS FILE.
6. DO NOT CHANGE THE NAME OF THIS FILE.
7. If there are any unclear parts while implementing these functions,
   please feel free to ask any questions through any channels available.
"""

###########################################################################
# RREF FORM with Integer (50 pts)
###########################################################################

"""
This function computes the Reduced Row Echelon Form (RREF) of the provided 
matrix and returns the result if all values in the RREF form are integers.
Otherwise, it returns None.

Parameters:
  matrix: a 2-D array (it can be either a proper form of a matrix or not)
Returns:
  option(matrix): None or a RREF matrix only with integer numbers
  
  cse2105_pset01.py 과제 안내 사항 5 변경 
5. DO NOT IMPORT ANY ADDITIONAL PACKAGES OR LIBRALIES (e.g., numpy, scipy)
IN THIS FILE except "from fractions import Fraction".

cse2105_pset01.py 내 RREF FORM with Integer 문제의 output 관련 설명 추가
option(matrix):
- It will return None when the input is not a matrix
- It will return None when the input matrix is not a square matrix
- It will return None when the RREF form of the matrix has non-integer numbers
- It will return the RREF form of the matrix when the input matrix is well-formed and all values of the RREF form are integer numbers
cse2105_pset01_test.py 내 RREF 4번째 테스트케이스 값 변경
input = [[2]]
answer = [[1]]
"""


def isSquare(matrix):
    if not matrix:
        return False
    if matrix is None or len(matrix) == 0:
        return False

    num_rows = len(matrix)
    num_cols = len(matrix[0])
    if num_rows != num_cols:
        return False

    for row in matrix:
        if len(row) != num_cols:
            return False

    return True


def hasDecimal(matrix):
    for row in matrix:
        for element in row:
            if isinstance(element, float) and element != int(element):
                return True

    return False


def changeRowWithPriority(matrix):
    d = {}
    for i in range(len(matrix)):
        flag = True

        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                if j in d.keys():
                    d[j].append(i)
                else:
                    d[j] = list()
                    d[j].append(i)
                flag = False
                break

        if flag:
            if len(matrix[i]) in d.keys():
                d[len(matrix[i])].append(i)
            else:
                d[len(matrix[i])] = list()
                d[len(matrix[i])].append(i)

    # swap
    sorted_list = sorted(d.keys())
    li = []
    for i in sorted_list:
        for j in d[i]:
            li.append(matrix[j])
    matrix = li.copy()
    return matrix


def RREFWithIntegerEntries(matrix):
    # 정방행렬 예외처리
    if not isSquare(matrix):
        return None

    pivots = []
    for row in range(len(matrix)):  # 각 행 순회
        # change row
        matrix = changeRowWithPriority(matrix)

        # scaling
        pivot_idx = 0
        for i in range(len(matrix[row])):  # 열 순회
            if matrix[row][i] != 0:
                pivot_idx = i
                break
        pivots.append(pivot_idx)

        pivot_val = matrix[row][pivot_idx]  # pivot 요소의 값
        for i in range(len(matrix[row])):
            if pivot_val != 0:
                matrix[row][i] /= pivot_val  # -> 1

        # addition
        for row2 in range(len(matrix)):
            pivot_col_val = matrix[row2][pivot_idx]  # pivot 열의 값
            if row == row2:
                continue
            for i in range(len(matrix[row2])):
                matrix[row2][i] -= matrix[row][i] * pivot_col_val  # -> 0

    # rounding
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = round(matrix[i][j], 2)

    # 소수 예외처리
    if hasDecimal(matrix):
        return None

    return matrix


###########################################################################
# Minimum Number of Multiplications for Matrices (50 pts)
###########################################################################
"""
This function calculates the minimum number of multiplications that are 
required for a matrix multiplication.

Parameters:
  matrix_sizes: a list that represents matrix sizes in a sequence. 
  If we hope to calculate the minimum number of multiplications for ABC
  when A \in M_(3, 2), B \in M_(2, 4), and C \in M(4, 3), "matrix_sizes"
  will be as follows:
  matrix_sizes = [(3, 2), (2, 4), 4, 3)]

Returns:
  option[number of required multiplications]:
  - It will return None when the input list is empty. 
  - It will return 0 when the input is a size for only one matrix (when
    the input is a singleton list)
  - It will return None when at least one of adjacent matrices are not
    conformable. (e.g., matrix_sizes = [(3, 2) (4, 3)])
  - It will return the number of minimum required multiplications 
    for the give input.
"""


def MinNumOfMul(matrix_sizes):
    n = len(matrix_sizes)

    # empty 예외처리
    if n == 0:
        return None
    # singleton list 예외처리
    if n == 1:
        return 0
    # not conformable 예외처리
    for i in range(0, n - 1):
        if matrix_sizes[i][1] != matrix_sizes[i + 1][0]:
            return None

    dp = []
    for i in range(n):
        tmp = []
        for j in range(n):
            tmp.append(0)
        dp.append(tmp)

    # 최소 연산 수 계산
    for i in range(1, len(matrix_sizes)):
        for start in range(0, len(matrix_sizes) - i):
            end = start + i
            temp = list()
            for middle in range(start, end):
                temp.append(dp[start][middle] + dp[middle + 1][end] +
                            matrix_sizes[start][0] * matrix_sizes[middle][1] * matrix_sizes[end][1])

            dp[start][end] = min(temp)

    return dp[0][-1]
