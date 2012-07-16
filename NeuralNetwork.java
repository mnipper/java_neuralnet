/**
 * Michael Nipper
 * 
 * 
 * Creates and trains a neural net, and returns the result of a test vector.
 * 
 * EXAMPLE USAGE
 * 		double[][] ins = { {.9, .9, .9, .9}, {.1, .1, .1, .1} };
		double[]   classes = {.9, .1};
		NeuralNetwork net = new NeuralNetwork(4, 2, 1, ins, classes);
		net.setMse(.008);
		net.trainNeuralNetwork();
		double[] inputTest = {.2, .2, .2, .2};
		System.out.println(net.classify(inputTest));
 */
import java.util.ArrayList;
import java.util.Random;

public class NeuralNetwork {

	private int numInputNodes = 0;
	private int numHiddenNodes = 0;
	private int numOutputNodes = 0;
	
	// Mean square error you are shooting for.
	private double mse = .1;
	
	// This is where known classes are provided for training purposes.
	private double[] expectedOutput;
	private double[][] expectedVectors;
	
	// Learning rate.
	private double RHO = .3;

	private ArrayList<Neuron> inputNodes = new ArrayList<Neuron>();
	private ArrayList<Neuron> hiddenNodes = new ArrayList<Neuron>();
	private ArrayList<Neuron> outputNodes = new ArrayList<Neuron>();

	// Holds the weights from nodes to nodes, this implemenation assumes one output node, since that is all we need.
	private double[][] inputToHiddenWeights;
	private double[] hiddenToOutputWeights;
	
	// Random number generator.
	private Random rng = new Random();

	// Provide the number of input, hidden, and output nodes.
	public NeuralNetwork(int i, int h, int o, double[][] eV, double[] eoV) {
		numInputNodes = i;
		numHiddenNodes = h;
		numOutputNodes = o;

		inputToHiddenWeights = new double[numInputNodes][numHiddenNodes];
		hiddenToOutputWeights = new double[numHiddenNodes];
		
		expectedOutput = eoV;
		expectedVectors = eV;

		buildNeuralNet();
	}

	
	// Create the neural net data structure and set weights to random value between 0 and .5.
	public void buildNeuralNet() {

		// Create node objects.
		for (int i = 0; i < numInputNodes; i++) {
			Neuron n = new Neuron();
			inputNodes.add(n);
		}
		for (int i = 0; i < numHiddenNodes; i++) {
			Neuron n = new Neuron();
			hiddenNodes.add(n);
		}
		for (int i = 0; i < numOutputNodes; i++) {
			Neuron n = new Neuron();
			outputNodes.add(n);
		}

		// Set random weights between 0 and .5.
		for (int i = 0; i < numInputNodes; i++)
			for (int j = 0; j < numHiddenNodes; j++)
				inputToHiddenWeights[i][j] = rng.nextDouble()/2;

		for (int i = 0; i < numHiddenNodes; i++)
			hiddenToOutputWeights[i] = rng.nextDouble()/2;
	}

	// Set the input vector to the input nodes.
	public void setInputs(double[] inputs) {
		for (int i = 0; i < numInputNodes; i++) {
			inputNodes.get(i).setInput(inputs[i]);
		}
	}

	public void trainNeuralNetwork() {

		double calcMSE = 1.0;

		while (calcMSE >= mse) {
			int randomVector = rng.nextInt(expectedOutput.length);
			backpropError(randomVector);
			calcMSE = Math.pow((getOutput(randomVector) - expectedOutput[randomVector]), 2) * .5;

		}
	}
	
	public void backpropError(int n) {
		double[] errorOutputLayer = new double[numOutputNodes];
		double[] errorHiddenLayer = new double[numHiddenNodes];
		
		double output = getOutput(n);
		
		for (int i = 0; i < numOutputNodes; i++) 
			errorOutputLayer[i] = (expectedOutput[n] - output) * sigmoidDerivative(output);
			
		for (int i = 0; i < numHiddenNodes; i++) {
			errorHiddenLayer[i] = 0;
			for (int j = 0; j < numOutputNodes; j++) {
				errorHiddenLayer[i] += errorOutputLayer[j] * hiddenToOutputWeights[i];
			}
			errorHiddenLayer[i] *= sigmoidDerivative(hiddenNodes.get(i).getOutput());
		}
			
		for (int i = 0; i < numOutputNodes; i++)
			for (int j = 0; j < numHiddenNodes; j++)
				hiddenToOutputWeights[i] += RHO * errorOutputLayer[i] * hiddenNodes.get(j).getOutput();
		
		for (int i = 0; i < numHiddenNodes; i++)
			for (int j = 0; j < numInputNodes; j++)
				inputToHiddenWeights[j][i] += RHO * errorHiddenLayer[i] * inputNodes.get(j).getInput();
		
	}

	public double getOutput(int n) {
		double output = 0.0;

		for (int i = 0; i < numHiddenNodes; i++)
			hiddenNodes.get(i).setInput(0.0);
		
		for (int i = 0; i < numOutputNodes; i++)
			outputNodes.get(i).setInput(0.0);
			
		for (int i = 0; i < numInputNodes; i++)
			for (int j = 0; j < numHiddenNodes; j++) {
				hiddenNodes.get(j).setInput( hiddenNodes.get(j).getInput() +
						expectedVectors[n][i]
								* inputToHiddenWeights[i][j]);
			}

		for (int i = 0; i < numHiddenNodes; i++) {
			output += hiddenNodes.get(i).getOutput() * hiddenToOutputWeights[i];
		}

		return output;
	}
	
	public double classify(double[] featureVector) {
		double output = 0.0;
		setInputs(featureVector);
			
		for (int i = 0; i < numInputNodes; i++)
			for (int j = 0; j < numHiddenNodes; j++) {
				hiddenNodes.get(j).setInput( hiddenNodes.get(j).getInput() +
						inputNodes.get(i).getInput()
								* inputToHiddenWeights[i][j]);
			}

		for (int i = 0; i < numHiddenNodes; i++) {
			output += hiddenNodes.get(i).getOutput() * hiddenToOutputWeights[i];
		}

		return sigmoid(output);
	}

	// Set the mean square error you wish to achieve.
	public void setMse(double mse) {
		this.mse = mse;
	}
	
	// Sigmoid derivative function, to be used in the error back-propagation.
	public double sigmoidDerivative(double u) {
		return u * (1 - u);
	}
	
	public double sigmoid(double u) {
		return 1.0 / (1.0 + Math.pow(Math.E, -u));
	}
}
