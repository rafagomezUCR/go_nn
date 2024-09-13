package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

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
	learning_rate := 0.3
	hidden_layers := []int{100, 50}
	activation := "s"
	epochs := 1
	weight_matrices := initializeWeights(784, hidden_layers, 10)
	nn := NN{
		learning_rate: learning_rate,
		hidden_layers: hidden_layers,
		activation:    activation,
		epochs:        epochs,
		weights:       weight_matrices,
	}
	train_file, train_file_err := os.Open("C:/Users/ruffl/Desktop/mnist data set/mnist_train.csv")
	test_file, test_file_err := os.Open("C:/Users/ruffl/Desktop/mnist data set/mnist_test.csv")
	checkError(train_file_err)
	checkError(test_file_err)
	defer train_file.Close()
	defer test_file.Close()
	scanner := bufio.NewScanner(train_file)
	test_scanner := bufio.NewScanner(test_file)
	if test_scanner.Scan() {
		line := test_scanner.Text()
		il, _ := convertFileValuesToMatrix(&line, ",")
		nn.input_layer = il
		result := nn.query()
		result.printMatrix()
		fmt.Println()
	} else {
		fmt.Println("Error reading file: ", test_scanner.Err())
	}
	for range epochs {
		for scanner.Scan() {
			line := scanner.Text()
			input_list, target := convertFileValuesToMatrix(&line, ",")
			target_list := createTargetList(target)
			nn.input_layer = input_list
			nn.target = target_list
			nn.train()
			//train(&input_list, weight_matrices, &target_list, learning_rate)
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file: ", err)
	}
	if test_scanner.Scan() {
		line := test_scanner.Text()
		il, target := convertFileValuesToMatrix(&line, ",")
		fmt.Println(target)
		nn.input_layer = il
		result := nn.query()
		result.printMatrix()
		fmt.Println()
	} else {
		fmt.Println("Error reading file: ", test_scanner.Err())
	}
	// if test_scanner.Scan() {
	// 	line := test_scanner.Text()
	// 	il, _ := convertFileValuesToMatrix(&line, ",")
	// 	test(&il, weight_matrices)
	// } else {
	// 	fmt.Println("Error reading file: ", test_scanner.Err())
	// }
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
