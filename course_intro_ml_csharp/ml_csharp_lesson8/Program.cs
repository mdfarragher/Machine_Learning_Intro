using Accord.Controls;
using Accord.Math;
using Accord.Math.Optimization.Losses;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Statistics;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Statistics.Visualizations;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ml_csharp_lesson8
{
    /// <summary>
    /// The main application class.
    /// </summary>
    public class Program
    {

        /// <summary>
        /// The main application entry point.
        /// </summary>
        /// <param name="args">Command line arguments.</param>
        public static void Main(string[] args)
        {
            // get data
            Console.WriteLine("Loading data....");
            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..");
            path = Path.Combine(path, "..");
            path = Path.Combine(path, "california_housing.csv");
            var housing = Frame.ReadCsv(path, separators: ",");
            housing = housing.Where(kv => ((decimal)kv.Value["median_house_value"]) < 500000);

            // shuffle the frame
            var rnd = new Random();
            var indices = Enumerable.Range(0, housing.Rows.KeyCount).OrderBy(v => rnd.NextDouble());
            housing = housing.IndexRowsWith(indices).SortRowsByKey();

            // convert the house value range to thousands
            housing["median_house_value"] /= 1000;

            // create training, validation, and test frames
            var training = housing.Rows[Enumerable.Range(0, 12000)];
            var validation = housing.Rows[Enumerable.Range(12000, 2500)];
            var test = housing.Rows[Enumerable.Range(14500, 2500)];

            // set up model columns
            var columns = new string[] {
                "latitude",
                "longitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income" };

            // build a neural network
            var network = new ActivationNetwork(
                new RectifiedLinearFunction(),  // the activation function
                8,                              // number of input features
                8,                              // hidden layer with 8 nodes
                1);                             // output layer with 1 node

            // set up a backpropagation learner
            var learner = new ParallelResilientBackpropagationLearning(network);

            // prep training feature and label arrays
            var features = training.Columns[columns].ToArray2D<double>().ToJagged();
            var labels = (from l in training["median_house_value"].Values
                          select new double[] { l }).ToArray();

            // prep validation feature and label arrays
            var features_v = validation.Columns[columns].ToArray2D<double>().ToJagged();
            var labels_v = (from l in validation["median_house_value"].Values
                            select new double[] { l }).ToArray();

            // warm up the network
            network.Randomize();
            for (var i = 0; i < 15; i++)
                learner.RunEpoch(features, labels);

            // train the neural network
            var errors = new List<double>();
            var errors_v = new List<double>();
            for (var epoch = 0; epoch < 100; epoch++)
            {
                learner.RunEpoch(features, labels);
                var rmse = Math.Sqrt(2 * learner.ComputeError(features, labels) / labels.GetLength(0));
                var rmse_v = Math.Sqrt(2 * learner.ComputeError(features_v, labels_v) / labels_v.GetLength(0));
                errors.Add(rmse);
                errors_v.Add(rmse_v);
                Console.WriteLine($"Epoch: {epoch}, Training RMSE: {rmse}, Validation RMSE: {rmse_v}");
            }

            // plot the training curve
            var x = Enumerable.Range(0, 100).Concat(Enumerable.Range(0, 100)).Select(v => (double)v).ToArray();
            var y = errors.Concat(errors_v).ToArray();
            var sets = Enumerable.Repeat(1, 100).Concat(Enumerable.Repeat(2, 100)).ToArray();
            var plot = new Scatterplot("", "Epoch", "RMSE");
            plot.Compute(x, y, sets);
            ScatterplotBox.Show(plot);

            Console.ReadLine();
        }
    }
}
