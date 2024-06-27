import java.util.Arrays;
import java.util.Random;

public class L4LayersAndObjects {
    public static void main(String[] args) {
        double[][] X = {
                {1.0, 2.0, 3.0, 2.5},
                {2.0, 5.0, -1.0, 2.0},
                {-1.5, 2.7, 3.3, -0.8}
        };

        Layer_Dense layer1 = new Layer_Dense(4, 5);
        Layer_Dense layer2 = new Layer_Dense(5, 2);

        layer1.forward(X);
        //System.out.println(Arrays.deepToString(layer1.getOutput()));
        layer2.forward(layer1.getOutput());
        System.out.println(Arrays.deepToString(layer2.getOutput()));
    }
}

class Layer_Dense {
    private static final Random random = new Random(0); //random number generator used for gaussian distribution.
    private double[][] weights; //weights of the layer
    private double[] biases; //biases of the layer
    private double[][] output; // output of the layer.

    public Layer_Dense(int n_inputs, int n_neurons) {
        this.weights = randn(n_inputs, n_neurons);
        this.biases = new double[n_neurons];
    }

    public void forward(double[][] inputs) {
        output = add(dotProduct(inputs, weights), biases);
    }

    public double[][] getOutput() {
        return output;
    }

    public double[][] randn(int rows, int cols) {
        double[][] output = new double[rows][cols];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                output[i][j] = 0.1 * random.nextGaussian();
            }
        }
        return output;
    }

    private static double[][] dotProduct(double[][] input1, double[][] input2) {
        double[][] output = new double[input1.length][input2[0].length];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                double value = 0;
                for (int k = 0; k < input1[0].length; k++) {
                    value += input1[i][k] * input2[k][j];
                }
                output[i][j] = value;
            }
        }
        return output;
    }

    private static double[][] add(double[][] input1, double[] input2) {
        double[][] output = new double[input1.length][input1[0].length];
        for (int i = 0; i < input1.length; i++) {
            for (int j = 0; j < input1[0].length; j++) {
                output[i][j] = input1[i][j] + input2[j];
            }
        }
        return output;
    }

}

