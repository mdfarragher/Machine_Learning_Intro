using Accord.Controls;
using Accord.Math;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Statistics.Visualizations;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ml_csharp_lesson9
{
    /// <summary>
    /// The main application class.
    /// </summary>
    public class Program
    {
        // validate a partially-trained network
        private static double Validate(Frame<int, string> validation, ActivationNetwork network)
        {
            int mistakes = 0;
            foreach (var key in validation.RowKeys)
            {
                var record = validation.Rows[key];
                var digit = (int)record.Values.First();
                var input = record.Values.Skip(1).Take(784).Cast<double>();
                var predictions = network.Compute(input.ToArray());
                var best = Enumerable.Range(0, 10)
                            .Select(v => new { Digit = v, Prediction = predictions[v] })
                            .OrderByDescending(v => v.Prediction)
                            .First();
                if (best.Digit != digit)
                    mistakes++;
            }
            return 1.0 * mistakes / validation.Rows.KeyCount;
        }

        /// <summary>
        /// The main application entry point.
        /// </summary>
        /// <param name="args">Command line arguments.</param>
        public static void Main(string[] args)
        {
            // read data
            Console.WriteLine("Loading data....");
            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..");
            path = Path.Combine(path, "..");
            path = Path.Combine(path, "handwritten_digits_medium.csv");
            var digits = Frame.ReadCsv(path, separators: ",", hasHeaders: false);
            Console.WriteLine($"    {digits.RowCount} rows loaded");

            // normalize pixel values to 0..1
            for (var i = 2; i <= 785; i++)
                digits[$"Column{i}"] /= 255.0;

            // grab a random digit
            var rnd = new Random();
            var row = rnd.Next(1, digits.RowCount);
            var label = digits.Rows[row]["Column1"].ToString();

            // plot the digit
            var x = Enumerable.Range(0, 784).Select(v => (double)(v % 28));
            var y = Enumerable.Range(0, 784).Select(v => (double)(-v / 28));
            var z = from i in Enumerable.Range(2, 784)
                    let v = (double)digits.Rows[row][$"Column{i}"]
                    select v > 0.5 ? 1 : 0;
            Scatterplot plot = new Scatterplot($"Digit {label}", "x", "y");
            plot.Compute(x.ToArray(), y.ToArray(), z.ToArray());
            ScatterplotBox.Show(plot);

            // create one-hot label columns
            for (var i = 0; i < 10; i++)
                digits.AddColumn($"Label{i}", from v in digits["Column1"].Values select (int)v == i ? 1.0 : 0.0);

            // print label columns
            digits.Columns[new string[] { "Column1", "Label0", "Label1", "Label2", "Label3", "Label4",
                "Label5", "Label6", "Label7", "Label8", "Label9" }].Print();

            // create training and validation partitions
            var numRows = digits.RowKeys.Count();
            var pivot = (int)(numRows * 0.8);
            var training = digits.Rows[Enumerable.Range(0, pivot)];
            var validation = digits.Rows[Enumerable.Range(pivot, numRows - pivot)];

            // set up feature and label column names
            var featureColumns = Enumerable.Range(2, 784).Select(v => $"Column{v}").ToArray();
            var labelColumns = Enumerable.Range(0, 10).Select(v => $"Label{v}").ToArray();

            // set up feature and label arrays
            var features = training.Columns[featureColumns].ToArray2D<double>().ToJagged();
            var labels = training.Columns[labelColumns].ToArray2D<double>().ToJagged();

            // build a neural network
            var network = new ActivationNetwork(
                new SigmoidFunction(),
                784,
                100,
                100,
                10);

            // randomize network weights
            new GaussianWeights(network, 0.1).Randomize();

            // set up a backpropagation learner
            var learner = new BackPropagationLearning(network)
            {
                LearningRate = 0.05
            };

            // train the network and validate it in each epoch
            Console.WriteLine("Training neural network....");
            var errors = new List<double>();
            var validationErrors = new List<double>();
            for (var epoch = 0; epoch < 50; epoch++)
            {
                var error = learner.RunEpoch(features, labels) / labels.GetLength(0);
                var validationError = Validate(validation, network);
                errors.Add(error);
                validationErrors.Add(validationError);
                Console.WriteLine($"Epoch: {epoch}, Training error: {error}, Validation error: {validationError}");
            }

            // test the network on the validation data
            Console.WriteLine($"Validating neural network on {validation.RowCount} records....");
            int mistakes = 0;
            foreach (var key in validation.RowKeys)
            {
                var record = validation.Rows[key];
                var digit = (int)record.Values.First();
                var input = record.Values.Skip(1).Take(784).Cast<double>();
                var predictions = network.Compute(input.ToArray());
                // Console.Write($"    {digit}: {predictions.ToString("0.00")} ");

                // calculate best prediction
                var best = Enumerable.Range(0, 10)
                    .Select(v => new { Digit = v, Prediction = predictions[v] })
                    .OrderByDescending(v => v.Prediction)
                    .First();
                //Console.Write($" -> {digit} = {best.Digit} ({100 * best.Prediction:0.00}%) ");

                // count incorrect predictions
                if (best.Digit != digit)
                {
                    Console.Write($"    {digit}: {predictions.ToString("0.00")} ");
                    Console.WriteLine($" -> {digit} = {best.Digit} ({100 * best.Prediction:0.00}%) WRONG");
                    //Console.Write("WRONG");
                    mistakes++;
                }
                //Console.WriteLine();
            }

            // report total mistakes
            var accuracy = 100.0 * (validation.Rows.KeyCount - mistakes) / validation.Rows.KeyCount;
            Console.WriteLine($"Total mistakes: {mistakes}, Accuracy: {accuracy:0.00}%");

            // plot the training and validation curves
            var tmp = Enumerable.Range(1, 50).Select(v => (double)v);
            x = tmp.Concat(tmp);
            y = errors.Concat(validationErrors);
            z = Enumerable.Repeat(1, 50).Concat(Enumerable.Repeat(2, 50));
            plot = new Scatterplot("Training & validation curves", "epochs", "training error");
            plot.Compute(x.ToArray(), y.ToArray(), z.ToArray());
            ScatterplotBox.Show(plot);

            Console.ReadLine();
        }
    }
}
