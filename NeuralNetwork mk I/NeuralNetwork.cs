using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_mk_I
{
    class NeuralNetwork
    {
        Layer[] layers;
        public NeuralNetwork(int[] layer)
        {

            layers = new Layer[layer.Length - 1];

            for(int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(layer[i],layer[i+1]);
            }

        }
        public double[] FeedForward(double[] inputs)
        {
            layers[0].FeedForward(inputs);
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].FeedForward(layers[i - 1].outputs);
            }
            return layers[layers.Length - 1].outputs;
        }

        public void BackProp(double[] expected)
        {
            for (int i = layers.Length-1; i >=0; i--)
            {
                if(i == layers.Length - 1)
                {
                    layers[i].BackPropOutput(expected);
                }
                else
                {
                    layers[i].BackPropHidden(layers[i+1].gamma,layers[i+1].weights);
                }
            }
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].UpdateWeights();
            }
        }

        public class Layer
        {
            int numberOfInputs; //Number of neurons in the previous layer
            int numberOfOutputs; //Number of neurons in the current layer

            public double[] outputs { get; set; }
            public double[] inputs { get; set; }
            public double[,] weights { get; set; }
            public double[,] weightsDelta { get; set; }
            public double[] gamma { get; set; }
            public double[] error { get; set; }

            public static Random random = new Random();

            public double LearningRate = 0.05f;

            public Layer(int numberOfInputs, int numberOfOutputs)
            {
                this.numberOfInputs = numberOfInputs;
                this.numberOfOutputs = numberOfOutputs;

                outputs = new double[numberOfOutputs];
                inputs = new double[numberOfInputs];
                weights = new double[numberOfOutputs, numberOfInputs];
                weightsDelta = new double[numberOfOutputs, numberOfInputs];
                gamma = new double[numberOfOutputs];
                error = new double[numberOfOutputs];

                InitializeWeights();
            }

            public void InitializeWeights()
            {
                for(int i = 0; i < numberOfOutputs; i++)
                {
                    for(int j = 0; j <numberOfInputs; j++)
                        weights[i, j] = (float)random.NextDouble() - 0.5f;
                }
            }

            public double[] FeedForward(double[] inputs)
            {
                this.inputs = inputs;

                for(int i = 0; i < numberOfOutputs; i++)
                {
                    outputs[i] = 0;

                    for (int j = 0; j < numberOfInputs; j++)
                        outputs[i] += inputs[j] * weights[i, j];

                    outputs[i] = (double)Math.Tanh(outputs[i]);
                }

                return outputs;
            }

            public double SinDer(double value)
            {
                return 1 - value * value;
            }

            public void BackPropOutput(double[] expected)
            {
                for(int i = 0; i < numberOfOutputs; i++)
                {
                    error[i] = outputs[i] - expected[i];
                    gamma[i] = error[i] * SinDer(outputs[i]);
                }
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    for (int j = 0; j < numberOfInputs; j++)
                        weightsDelta[i, j] = gamma[i] * inputs[j];
                }
            }

            public void BackPropHidden(double[] gammaForward, double[,] weightForward)
            {
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    gamma[i] = 0;

                    for(int j = 0; j < gammaForward.Length; j++)
                    {
                        gamma[i] += gammaForward[j] * weightForward[j, i];
                    }
                    gamma[i] *= SinDer(outputs[i]);
                }
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    for (int j = 0; j < numberOfInputs; j++)
                        weightsDelta[i, j] = gamma[i] * inputs[j];
                }
            }

            public void UpdateWeights()
            {
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    outputs[i] = 0;


                    for (int j = 0; j < numberOfInputs; j++)
                        weights[i, j] -= weightsDelta[i, j] * LearningRate;
                }
            }

        }
    }
}
