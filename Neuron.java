/**
 * Michael Nipper
 * 
 * 
 * For use with the NeuralNetwork class.  Stores the data needed for each neuron.
 *
 */
public class Neuron {
	
	private double input = 0.0;
	private double error = 0.0;

	public void setInput(double input) {
		this.input = input;
	}
	public double getInput() {
		return this.input;
	}
	public double getError() {
		return error;
	}
	public double getOutput() {
		return (1.0 / (1.0 + Math.pow(Math.E, -input)));
	}
	
	
}
