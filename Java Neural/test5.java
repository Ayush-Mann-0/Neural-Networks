import java.util.Arrays;
import java.util.Random;

public class L5ReLUActivationAndSpiralDataset {
    private static final Random random = new Random(0);

    public static void main(String[] args) {
        //create a dataset object to hold features
        Dataset dataset = new Dataset();
        dataset.create_data(100, 3);

        Layer_Dense layer1 = new Layer_Dense(4, 5);
        Activation_ReLU activation1 = new Activation_ReLU();

        layer1.forward(dataset.X);
        activation1.forward(layer1.output);
        System.out.println(Arrays.deepToString(activation1.output));
    }

    private static class Dataset {
        private double[][] X;
        private int[] Y;

        public void create_data(int points, int classes) {
            X = new double[points * classes][2];
            Y = new int[points * classes];
            int ix = 0;
            for (int class_number = 0; class_number < classes; class_number++) {
                double r = 0;
                double t = class_number * 4;
                while (r <= 1 && t <= (class_number + 1) * 4) {
                    double random_t = t + random.nextInt(points) * 0.2;
                    X[ix][0] = r * Math.sin(random_t * 2.5);
                    X[ix][1] = r * Math.cos(random_t * 2.5);
                    Y[ix] = class_number;
                    r += 1.0 / (points - 1);
                    t += 4.0 / (points - 1);
                    ix++;
                }
            }
        }
    }

    private static class Layer_Dense {
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

    private static class Activation_ReLU {
        private double[][] output;
        public void forward(double[][] inputs) {
            output = new double[inputs.length][inputs[0].length];
            for (int i = 0; i < output.length; i++) {
                for (int j = 0; j < output[0].length; j++) {
                    if (inputs[i][j] > 0) {
                        output[i][j] = 0;
                    } else {
                        output[i][j] = inputs[i][j];
                    }
                }
            }
        }
    }
}
