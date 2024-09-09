package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

type Matrix struct {
	data [][]float64
	rows int
	cols int
}

func query(input_list *Matrix, weight_matrices []Matrix) []Matrix {
	output_list := make([]Matrix, 0)
	input_list_t := transposeMatrix(input_list)
	output_list = append(output_list, input_list_t)
	for i := range len(weight_matrices) {
		signal_output := (weight_matrices)[i].matrixMult(&input_list_t)
		signal_output.sigmoidActivation()
		output_list = append(output_list, signal_output)
		input_list_t = signal_output
	}
	return output_list
}

func train(input_list *Matrix, weight_matrices []Matrix, target_list *Matrix, learning_rate float64) {
	network_outputs := query(input_list, weight_matrices)
	error_output := calculateError(&network_outputs[len(network_outputs)-1], target_list)
	for i := len(weight_matrices) - 1; i >= 0; i-- {
		weight_t := transposeMatrix(&(weight_matrices)[i])
		error_hidden := weight_t.matrixMult(&error_output)
		// right here goes gradient descent
		one_subtract_o := scalar_minus_matrix(1.0, &network_outputs[i+1])
		de_dw := elementwise_matrix_multiplication(&network_outputs[i+1], &one_subtract_o)
		de_dw = elementwise_matrix_multiplication(&error_output, &de_dw)
		hidden_outputs := transposeMatrix(&network_outputs[i])
		de_dw = de_dw.matrixMult(&hidden_outputs)
		de_dw = scale_matrix(learning_rate, &de_dw)
		weight_matrices[i] = add_matrices(&(weight_matrices)[i], &de_dw)
		error_output = error_hidden
	}
}

func test(input_list *Matrix, weight_matrices []Matrix) {
	input_list_t := transposeMatrix(input_list)
	for i := range len(weight_matrices) {
		signal_output := (weight_matrices)[i].matrixMult(&input_list_t)
		signal_output.sigmoidActivation()
		input_list_t = signal_output
	}
	input_list_t.printMatrix()
}

func checkError(e error) {
	if e != nil {
		panic(e)
	}
}

func convertFileValuesToMatrix(input *string, delimiter string) (Matrix, float64) {
	strs := strings.Split(*input, delimiter)
	input_matrix := createMatrix(1, len(strs)-1)
	var target_value float64
	for i, str := range strs {
		val, err := strconv.ParseFloat(str, 64)
		checkError(err)
		if i == 0 {
			target_value = val
		} else {
			val = (val / 255 * 0.99)
			if val == 0 {
				val += 0.01
			}
			input_matrix.data[0][i-1] = val
		}
	}
	return input_matrix, target_value
}

func createTargetList(target float64) Matrix {
	target_list := Matrix{
		data: [][]float64{
			{0.01},
			{0.01},
			{0.01},
			{0.01},
			{0.01},
			{0.01},
			{0.01},
			{0.01},
			{0.01},
			{0.01},
		},
		rows: 10,
		cols: 1,
	}
	target_list.data[int(target)][0] = 0.99
	return target_list
}

func main() {
	weight_matrices := initializeWeights(784, []int{100}, 10)
	learning_rate := 0.3
	epochs := 1
	train_file, err := os.Open("C:/Users/ruffl/Desktop/mnist data set/mnist_train.csv")
	test_file, test_err := os.Open("C:/Users/ruffl/Desktop/mnist data set/mnist_test.csv")
	checkError(test_err)
	checkError(err)
	defer train_file.Close()
	defer test_file.Close()
	scanner := bufio.NewScanner(train_file)
	test_scanner := bufio.NewScanner(test_file)
	//i := 1
	if test_scanner.Scan() {
		line := test_scanner.Text()
		il, _ := convertFileValuesToMatrix(&line, ",")
		test(&il, weight_matrices)
		fmt.Println()
	} else {
		fmt.Println("Error reading file: ", test_scanner.Err())
	}
	for range epochs {
		for scanner.Scan() {
			//fmt.Println("training on: ", i)
			line := scanner.Text()
			input_list, target := convertFileValuesToMatrix(&line, ",")
			target_list := createTargetList(target)
			train(&input_list, weight_matrices, &target_list, learning_rate)
			//i++
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file: ", err)
	}
	if test_scanner.Scan() {
		line := test_scanner.Text()
		il, _ := convertFileValuesToMatrix(&line, ",")
		test(&il, weight_matrices)
		fmt.Println()
	} else {
		fmt.Println("Error reading file: ", test_scanner.Err())
	}
	if test_scanner.Scan() {
		line := test_scanner.Text()
		il, _ := convertFileValuesToMatrix(&line, ",")
		test(&il, weight_matrices)
	} else {
		fmt.Println("Error reading file: ", test_scanner.Err())
	}
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

func initializeWeights(input_layer_size int, neurons_per_layer []int, output_layer_size int) []Matrix {
	weightMatrices := make([]Matrix, 0)
	for i := range len(neurons_per_layer) + 1 {
		if i == 0 {
			w := createMatrix(neurons_per_layer[i], input_layer_size)
			weightMatrices = append(weightMatrices, w)
		} else if i == len(neurons_per_layer) {
			w := createMatrix(output_layer_size, neurons_per_layer[i-1])
			weightMatrices = append(weightMatrices, w)
		} else {
			w := createMatrix(neurons_per_layer[i], neurons_per_layer[i-1])
			weightMatrices = append(weightMatrices, w)
		}
	}
	for i := range len(weightMatrices) {
		for j := range weightMatrices[i].rows {
			for k := range weightMatrices[i].cols {
				weightMatrices[i].data[j][k] = rand.Float64() - 0.5
			}
		}
	}
	return weightMatrices
}

func calculateError(o *Matrix, t *Matrix) Matrix {
	if o.rows != t.rows && o.cols != t.cols {
		fmt.Println("output and target matrices dont match in length")
		return Matrix{}
	}
	e := createMatrix(o.rows, o.cols)
	for i := range o.rows {
		for j := range o.cols {
			e.data[i][j] = t.data[i][j] - o.data[i][j]
		}
	}
	return e
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
