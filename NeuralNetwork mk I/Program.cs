using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_mk_I
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Benchmark results:");
            Console.WriteLine("Setosa:\t\t 1\t 0\t 0");
            Console.WriteLine("Versicolor:\t 0\t 1\t 0");
            Console.WriteLine("Virginica:\t 0\t 0\t 1");
            Console.WriteLine();
            for (int i = 0; i < 10; i++)
            {
                //Test_XOR();
                //Test_Iris();
                Test_Kfold_Iris();
            }
            //Test_SOM();
            //Test_SOM_Iris();
            Console.ReadKey();
        }

        static void Test_SOM_Iris()
        {
            SOM som = new SOM(4, 3, @".\iris_som.csv");

        }

        static void Test_SOM()
        {
            SOM som = new SOM(4, 3, @".\iris_som.csv");
        }

        static void Test_Kfold_Iris()
        {
            NeuralNetwork network = new NeuralNetwork(new int[] { 4, 7, 5, 3 });

            List<Tuple<double[], double[]>> input = new List<Tuple<double[], double[]>>();

            using (var reader = new StreamReader(@".\input.csv"))
            {
                reader.ReadLine();
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    input.Add(new Tuple<double[], double[]>(new double[] {Double.Parse(values[0], CultureInfo.InvariantCulture),
                                                  Double.Parse(values[1], CultureInfo.InvariantCulture),
                                                  Double.Parse(values[2], CultureInfo.InvariantCulture),
                                                  Double.Parse(values[3], CultureInfo.InvariantCulture)},
                                                  MagicSwitch(values[4])));
                }
            }

            Random rng = new Random();
            int n = input.Count;
            while (n > 1)
            {
                n--;
                int l = rng.Next(n + 1);
                Tuple<double[],double[]> value = input[l];
                input[l] = input[n];
                input[n] = value;
            }

            int k = 10;
            double score = 0;
            double[] results;
            List<Tuple<double[], double[]>>[] chunks = new List<Tuple<double[], double[]>>[10];
            int chunkSize = input.Count() / k;
            for(int i = 0; i < k; i++)
            {
                chunks[i] = input.GetRange(i*chunkSize, chunkSize);
            }
            for (int j = 0; j < k; j++)
            {
                Console.WriteLine("Chunk " + (j + 1));
                for (int i = 0; i < k; i++)
                {
                    if (i != j)
                    {
                        for (int m = 0; m < 5000; m++)
                        {
                            foreach (Tuple<double[], double[]> t in chunks[i])
                            {
                                network.FeedForward(t.Item1);
                                network.BackProp(t.Item2);
                            }
                        }
                    }
                }
                score = 0.0d;
                foreach (Tuple<double[], double[]> t in chunks[j])
                {
                    results = network.FeedForward(t.Item1);
                    List<double> r1 = results.ToList();
                    List<double> r2 = t.Item2.ToList();
                    if ( r1.IndexOf(r1.Max()) == r2.IndexOf(r2.Max()))  score += 1.0d;
                }
                score = score / chunkSize;
                Console.WriteLine("Accuracy: " + score*100 + "%");
            }
            //double[] result = network.FeedForward(new double[] { 5.8d, 4.0d, 1.2d, 0.2d });
            //foreach (double d in result)
            //{
            //    Console.Write(d + " ");
            //}
            score = 0.0d;
            foreach (Tuple<double[],double[]> t in input)
            {
                results = network.FeedForward(t.Item1);
                List<double> r1 = results.ToList();
                List<double> r2 = t.Item2.ToList();
                if (r1.IndexOf(r1.Max()) == r2.IndexOf(r2.Max())) score += 1.0d;
            }
            score = score / input.Count();
            Console.WriteLine("Accuracy: " + score * 100 + "%");
            Console.ReadKey();
        }

        static void Test_Iris()
        {
            NeuralNetwork network = new NeuralNetwork(new int[] { 4, 7, 5, 3 });

            List<Tuple<double[],double[]>> input = new List<Tuple<double[],double[]>>();

            using (var reader = new StreamReader(@".\input.csv"))
            {
                reader.ReadLine();
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    input.Add(new Tuple<double[],double[]>(new double[] {Double.Parse(values[0], CultureInfo.InvariantCulture),
                                                  Double.Parse(values[1], CultureInfo.InvariantCulture),
                                                  Double.Parse(values[2], CultureInfo.InvariantCulture),
                                                  Double.Parse(values[3], CultureInfo.InvariantCulture)},
                                                  MagicSwitch(values[4])));
                }
            }

            Random rng = new Random();
            int n = input.Count;
            while (n > 1)
            {
                n--;
                int l = rng.Next(n + 1);
                Tuple<double[], double[]> value = input[l];
                input[l] = input[n];
                input[n] = value;
            }

            for (int i = 1; i < 5000; i++)
            {
                foreach (Tuple<double[], double[]> t in input)
                {
                    network.FeedForward(t.Item1);
                    network.BackProp(t.Item2);
                }
            }

            double[] result = network.FeedForward(new double[] { 5.0, 3.4, 1.6, 0.4 });
            Console.Write("Setosa:\t\t");
            foreach(double d in result)
            {
                Console.Write("{0:F3}\t", d);
            }
            Console.WriteLine();
            result = network.FeedForward(new double[] { 6.0, 2.2, 4.0, 1.0 });
            Console.Write("Versicolor:\t");
            foreach (double d in result)
            {
                Console.Write("{0:F3}\t", d);
            }
            Console.WriteLine();
            result = network.FeedForward(new double[] { 6.1, 3.0, 4.9, 1.8 });
            Console.Write("Virginica:\t");
            foreach (double d in result)
            {
                Console.Write("{0:F3}\t", d);
            }
            Console.WriteLine("\n");
            //Console.ReadKey();
        }

        static double[] MagicSwitch(string s)
        {
            if (s == "Iris-setosa")
            {
                return new double[] { 1.0, 0.0, 0.0 };
            }
            else if (s == "Iris-versicolor")
            {
                return new double[] { 0.0, 1.0, 0.0 };
            }
            else if (s == "Iris-virginica")
            {
                return new double[] { 0.0, 0.0, 1.0 };
            }
            else
                return null;
        }

        static void Test_XOR()
        {
            NeuralNetwork network = new NeuralNetwork(new int[] { 3, 25, 25, 1 });

            for (int i = 0; i < 5000; i++)
            {
                network.FeedForward(new double[] { 0, 0, 0 });
                network.BackProp(new double[] { 0 });

                network.FeedForward(new double[] { 0, 0, 1 });
                network.BackProp(new double[] { 1 });

                network.FeedForward(new double[] { 0, 1, 0 });
                network.BackProp(new double[] { 1 });

                network.FeedForward(new double[] { 0, 1, 1 });
                network.BackProp(new double[] { 0 });

                network.FeedForward(new double[] { 1, 0, 0 });
                network.BackProp(new double[] { 1 });

                network.FeedForward(new double[] { 1, 0, 1 });
                network.BackProp(new double[] { 0 });

                network.FeedForward(new double[] { 1, 1, 0 });
                network.BackProp(new double[] { 0 });

                network.FeedForward(new double[] { 1, 1, 1 });
                network.BackProp(new double[] { 1 });
            }

            Console.WriteLine(String.Format("0 0 0 |  {0:F3}", network.FeedForward(new double[] { 0, 0, 0 })[0]));
            Console.WriteLine(String.Format("0 0 1 |  {0:F3}", network.FeedForward(new double[] { 0, 0, 1 })[0]));
            Console.WriteLine(String.Format("0 1 0 |  {0:F3}", network.FeedForward(new double[] { 0, 1, 0 })[0]));
            Console.WriteLine(String.Format("0 1 1 |  {0:F3}", network.FeedForward(new double[] { 0, 1, 1 })[0]));
            Console.WriteLine(String.Format("1 0 0 |  {0:F3}", network.FeedForward(new double[] { 1, 0, 0 })[0]));
            Console.WriteLine(String.Format("1 0 1 |  {0:F3}", network.FeedForward(new double[] { 1, 0, 1 })[0]));
            Console.WriteLine(String.Format("1 1 0 |  {0:F3}", network.FeedForward(new double[] { 1, 1, 0 })[0]));
            Console.WriteLine(String.Format("1 1 1 |  {0:F3}", network.FeedForward(new double[] { 1, 1, 1 })[0]));
            Console.WriteLine("\n\n");
            //Console.ReadKey();
        }
    }
}
