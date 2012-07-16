
public class Driver {
	
	public static void main(String[] args) {
		double[][] ins = { {.9, .9, .9, .9}, {.1, .1, .1, .1} };
		double[]   classes = {.9, .1};
		NeuralNetwork net = new NeuralNetwork(4, 2, 1, ins, classes);
		net.setMse(.008);
		net.trainNeuralNetwork();
		double[] inputTest = {.2, .2, .2, .2};
		System.out.println(net.classify(inputTest));
	}

}
