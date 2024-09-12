package main

import (
	"fmt"
	"math"
)

type Matrix struct {
	data [][]float64
	rows int
	cols int
}

func (a *Matrix) printMatrix() {
	for i := range a.rows {
		for j := range a.cols {
			fmt.Print(a.data[i][j], " ")
		}
		fmt.Println()
	}
}

func add_matrices(a *Matrix, b *Matrix) Matrix {
	if a.rows != b.rows || a.cols != b.cols {
		fmt.Println("matrices do not match for subtracting")
		return Matrix{}
	}
	res := createMatrix(a.rows, a.cols)
	for i := range a.rows {
		for j := range a.cols {
			res.data[i][j] = a.data[i][j] + b.data[i][j]
		}
	}
	return res
}

func subtract_matrices(a *Matrix, b *Matrix) Matrix {
	if a.rows != b.rows || a.cols != b.cols {
		fmt.Println("matrices do not match for subtracting")
		return Matrix{}
	}
	res := createMatrix(a.rows, a.cols)
	for i := range a.rows {
		for j := range a.cols {
			res.data[i][j] = a.data[i][j] - b.data[i][j]
		}
	}
	return res
}

func scalar_minus_matrix(scalar float64, a *Matrix) Matrix {
	res := createMatrix(a.rows, a.cols)
	for i := range a.rows {
		for j := range a.cols {
			res.data[i][j] = scalar - a.data[i][j]
		}
	}
	return res
}

func scale_matrix(scalar float64, a *Matrix) Matrix {
	res := createMatrix(a.rows, a.cols)
	for i := range a.rows {
		for j := range a.cols {
			res.data[i][j] = scalar * a.data[i][j]
		}
	}
	return res
}

func elementwise_matrix_multiplication(a *Matrix, b *Matrix) Matrix {
	if a.rows != b.rows || a.cols != b.cols {
		fmt.Println("matrices do not match for elementwise multiplication")
		return Matrix{}
	}
	res := createMatrix(a.rows, a.cols)
	for i := range a.rows {
		for j := range a.cols {
			res.data[i][j] = a.data[i][j] * b.data[i][j]
		}
	}
	return res
}

func (a *Matrix) matrixMult(b *Matrix) Matrix {
	if a.cols != b.rows {
		fmt.Println("rows and cols do not match")
		return Matrix{}
	}
	res := createMatrix(a.rows, b.cols)
	for i := range a.rows {
		for j := range b.cols {
			for k := range b.rows {
				res.data[i][j] += a.data[i][k] * b.data[k][j]
			}
		}
	}
	return res
}

func (matrix *Matrix) sigmoidActivation() {
	for i := range matrix.rows {
		for j := range matrix.cols {
			matrix.data[i][j] = 1 / (1 + math.Exp(-matrix.data[i][j]))
		}
	}
}

func sigmoidActivation(matrix *Matrix) Matrix {
	result_matrix := createMatrix(matrix.rows, matrix.cols)
	for i := range matrix.rows {
		for j := range matrix.cols {
			result_matrix.data[i][j] = 1 / (1 + math.Exp(-matrix.data[i][j]))
		}
	}
	return result_matrix
}

func (a *Matrix) scaleMatrix(factor float64) {
	for i := range a.rows {
		for j := range a.cols {
			a.data[i][j] *= factor
		}
	}
}

func (a *Matrix) transposeMatrix() {
	b := createMatrix(a.cols, a.rows)
	for i := range a.rows {
		for j := range a.cols {
			b.data[j][i] = a.data[i][j]
		}
	}
	*a = b
}

func transposeMatrix(a *Matrix) Matrix {
	b := createMatrix(a.cols, a.rows)
	for i := range a.rows {
		for j := range a.cols {
			b.data[j][i] = a.data[i][j]
		}
	}
	return b
}

func createMatrix(rows int, cols int) Matrix {
	m := Matrix{
		rows: rows,
		cols: cols,
	}
	m.data = make([][]float64, rows)
	for i := range rows {
		m.data[i] = make([]float64, cols)
	}
	return m
}
