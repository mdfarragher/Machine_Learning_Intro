using Accord.Controls;
using Accord.Math;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Statistics.Visualizations;
using Deedle;
using System;
using System.IO;
using System.Linq;

namespace ml_csharp_lesson2
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
            var path = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\california_housing.csv"));
            var housing = Frame.ReadCsv(path, separators: ",");
            housing = housing.Where(kv => ((decimal)kv.Value["median_house_value"]) < 500000);

            // convert the house value range to thousands
            housing["median_house_value"] /= 1000;

            // shuffle row indices
            //var rnd = new Random();
            //var indices = Enumerable.Range(0, housing.Rows.KeyCount).OrderBy(v => rnd.NextDouble());

            // shuffle the frame using the indices
            //housing = housing.IndexRowsWith(indices).SortRowsByKey();

            // create training, validation, and test frames
            var training = housing.Rows[Enumerable.Range(0, 14000)];
            var test = housing.Rows[Enumerable.Range(14000, 3000)];
            //var training = housing.Rows[Enumerable.Range(0, 12000)];
            //var validation = housing.Rows[Enumerable.Range(12000, 2500)];
            //var test = housing.Rows[Enumerable.Range(14500, 2500)];

            // plot the training data
            var x = training["longitude"].Values.ToArray();
            var y = training["latitude"].Values.ToArray();
            var plot = new Scatterplot("Training", "longitude", "latitude");
            plot.Compute(x, y);
            ScatterplotBox.Show(plot);

            // plot the test data
            var x2 = test["longitude"].Values.ToArray();
            var y2 = test["latitude"].Values.ToArray();
            var plot2 = new Scatterplot("Test", "longitude", "latitude");
            plot2.Compute(x2, y2);
            ScatterplotBox.Show(plot2);

            // set up training features and labels
            //var training_features = training["median_income"].Values.ToArray();
            //var training_labels = training["median_house_value"].Values.ToArray();

            // train the model
            //Console.WriteLine("Training model....");
            //var learner = new OrdinaryLeastSquares();
            //var model = learner.Learn(training_features, training_labels);

            // show results
            //Console.WriteLine("TRAINING RESULTS");
            //Console.WriteLine($"Slope:       {model.Slope}");
            //Console.WriteLine($"Intercept:   {model.Intercept}");
            //Console.WriteLine();

            // get training predictions
            //var training_predictions = model.Transform(training_features);

            // set up validation features and labels
            //var validation_features = validation["median_income"].Values.ToArray();
            //var validation_labels = validation["median_house_value"].Values.ToArray();

            // validate the model
            //var validation_predictions = model.Transform(validation_features);
            //var validation_rmse = Math.Sqrt(new SquareLoss(validation_labels).Loss(validation_predictions));

            // show validation results
            //var validation_range = Math.Abs(validation_labels.Max() - validation_labels.Min());
            //Console.WriteLine("VALIDATION RESULTS");
            //Console.WriteLine($"Label range: {validation_range}");
            //Console.WriteLine($"RMSE:        {validation_rmse} {validation_rmse / validation_range * 100:0.00}%");
            //Console.WriteLine();

            // set up test features and labels
            //var test_features = test["median_income"].Values.ToArray();
            //var test_labels = test["median_house_value"].Values.ToArray();

            // validate the model
            //var test_predictions = model.Transform(test_features);
            //var test_rmse = Math.Sqrt(new SquareLoss(test_labels).Loss(test_predictions));

            // show validation results
            //var test_range = Math.Abs(test_labels.Max() - test_labels.Min());
            //Console.WriteLine("TEST RESULTS");
            //Console.WriteLine($"Label range: {test_range}");
            //Console.WriteLine($"RMSE:        {test_rmse} {test_rmse / test_range * 100:0.00}%");
            //Console.WriteLine();

            // show training plot
            //x = training_features.Concat(training_features).ToArray();
            //y = training_predictions.Concat(training_labels).ToArray();
            //var colors1 = Enumerable.Repeat(1, training_labels.Length).ToArray();
            //var colors2 = Enumerable.Repeat(2, training_labels.Length).ToArray();
            //var c = colors1.Concat(colors2).ToArray();
            //plot = new Scatterplot("Training", "feature", "label");
            //plot.Compute(x, y, c);
            //ScatterplotBox.Show(plot);

            // show validation plot
            //x = validation_features.Concat(validation_features).ToArray();
            //y = validation_predictions.Concat(validation_labels).ToArray();
            //colors1 = Enumerable.Repeat(1, validation_labels.Length).ToArray();
            //colors2 = Enumerable.Repeat(2, validation_labels.Length).ToArray();
            //c = colors1.Concat(colors2).ToArray();
            //plot = new Scatterplot("Validation", "feature", "label");
            //plot.Compute(x, y, c);
            //ScatterplotBox.Show(plot);

            // show test plot
            //x = test_features.Concat(test_features).ToArray();
            //y = test_predictions.Concat(test_labels).ToArray();
            //colors1 = Enumerable.Repeat(1, test_labels.Length).ToArray();
            //colors2 = Enumerable.Repeat(2, test_labels.Length).ToArray();
            //c = colors1.Concat(colors2).ToArray();
            //plot = new Scatterplot("Test", "feature", "label");
            //plot.Compute(x, y, c);
            //ScatterplotBox.Show(plot);

            Console.ReadLine();
        }
    }
}
