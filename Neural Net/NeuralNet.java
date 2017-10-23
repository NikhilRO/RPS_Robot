import java.math.*;
public class NeuralNet {
	static double vectorNormSqrd (double [] v) {
		double len = 0;
		for (int a = 0; a < v.length; a++) {
			len += v[a]*v[a];
		}
		return len;
	}
	
	static double costFunc (int n, double [] pred, double [] actual) {
		double cost = 0;
		for (int a = 0; a < pred.length; a++) {
			cost += vectorNormSqrd(Mat.matAdd(1.0, -1.0, pred, actual));
		}
		cost /= 2*n;
		return cost;
	}
	
	public static void main(String[] args) {
		
	}
}

class Net {
	static final double e = Math.E;
	int layers;
	NeuralWeights [] weights;
	double [][] baises;
	
	Net (int input, int output, int [] neurons) {
		this.layers = neurons.length;
		weights = new NeuralWeights [layers+1];
		
		this.weights[0] = new NeuralWeights(input, neurons[0]);
		
		for (int a = 1; a < layers; a++) {
			weights[a] = new NeuralWeights(neurons[a-1], neurons[a]);
		}
		
		weights[layers] = new NeuralWeights(neurons[layers-1], output);
		
		baises = new double[layers][];
		
		for (int a = 0; a < layers; a++) {
			baises[a] = new double[neurons[a]];
		}
		
		this.randomize();
	}
	
	// Uses standard gradient decent algorithm
	void SGD (Mat [] training_data, int [] expected_output, int epochs, 
			int mini_batch_size, double learning_rate) {
		int num_data = expected_output.length;
		
		Data [] data = new Data [num_data];
		for (int a = 0; a < num_data; a++) {
			data[a] = new Data(training_data[a], expected_output[a]);
		}
		
		Data [][] batches = randomizeTestData(data, epochs, mini_batch_size);
		
		for (int a = 0; a < epochs; a++) {
			
		}
	}
	
	// Finds all outputs on a given layer
	double [] feedForward (int currLayer, double [] input) {
		double [] nextLayer = new double [weights[currLayer+1].getNeurons()];
		for (int a = 0; a < nextLayer.length; a++) {
			nextLayer[a] = activation(currLayer, a, input);
		}
		return nextLayer;
	}
	
	// Finds output of a neuron
	double activation (int prevLayer, int currNeuron, double [] input) {
		double sum = 0;
		return sigmoidFunc(Mat.dot(input, weights[prevLayer].getWeights(currNeuron)) + baises[prevLayer][currNeuron]);
	}
	
	// Creates a number of mini batches of test data all of a specified size
	Data [][] randomizeTestData (Data [] testData, int epochs, int batchSizes) {
		int nTestData = testData.length;
		int [] rand = randomizeNums(nTestData);
		
		Data [][] batches = new Data[epochs][batchSizes];
		
		int count = 0;
		for (int a = 0; a < epochs; a++) {
			for (int b = 0; b < batchSizes; b++) {
				batches[a][b] = testData[count];
				count++;
				count %= nTestData;
			}
		}
		
		return batches;
	}
	
	// The sigmoid normalization function
	static double sigmoidFunc (double z) {
		double deno = 1 + Math.pow(e, -z);
		return 1/deno;
	}
	
	// Randomizes weights and baises
	void randomize() {
		randomizeWeights();
		randomizeBaises();
	}
	
	// Randomizes weights
	void randomizeWeights() {
		for (int a = 0; a < weights.length; a++) {
			for (int b = 0; b < weights[a].getNeurons(); b++) {
				for (int c = 0; c < weights[a].getConnections(); c++) {
					weights[a].setCell(b, c, Math.random());
				}
			}
		}
	}
	
	// Randomizes baises
	void randomizeBaises() {
		for (int a = 0; a < baises.length; a++) {
			for (int b = 0; b < baises[a].length; b++) {
				baises[a][b] = Math.random();
			}
		}
	}
	
	// Randomizes numbers from 0-len
	int [] randomizeNums (int len) {
		int [] ran = new int [len];
		for (int a = 0; a < len; a++) {
			ran[a] = len;
		}
		
		for (int a = 0; a < len; a++) {
			int swapWith = (int)(len*Math.random());
			int t = ran[swapWith];
			ran[swapWith] = a;
			ran[a] = t;
		}
		
		return ran;
	}
}

class NeuralWeights {
	int neurons;
	int connections;
	double [][] mat;
	
	NeuralWeights (int neurons, int connections) {
		this.neurons = neurons;
		this.connections = connections;
		mat = new double [neurons][connections];
	}
	
	void setCell (int i, int j, double val) {
		mat[i][j] = val;
	}
	
	int getNeurons () {
		return neurons;
	}
	
	double [] getWeights (int neuron) {
		return mat[neuron];
	}
	
	int getConnections () {
		return connections;
	}
}

class Mat {
	static double [] matAdd (double c1, double c2, double [] m1, double [] m2) {
		int n = m1.length;
		double [] m3 = new double [n];
		for (int a = 0; a < n; a++) { 
			m3[a] = c1*m1[a] + c2*m2[a];
		}
		return m3;
	}
	
	// Dot prod between 2 vectors
	static double dot (double [] m1, double [] m2) {
		double sum = 0;
		for (int a = 0; a < m1.length; a++) {
			sum += m1[a]*m2[a];
		}
		return sum;
	}
}

class Data {
	Mat m;
	int r;
	
	Data (Mat m, int r) {
		this.m = m;
		this.r = r;
	}
	
	Mat getMat() {
		return m;
	}
	
	int getResult() {
		return r;
	}
}
