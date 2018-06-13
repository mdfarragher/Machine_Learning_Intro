using Accord.Controls;
using Accord.Math;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Statistics.Visualizations;
using Deedle;
using System;
using System.IO;
using System.Linq;

namespace ml_csharp_lesson1
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
            // housing = housing.Where(kv => ((decimal)kv.Value["median_house_value"]) < 500000);

            // set up a few series
            var total_rooms = housing["total_rooms"];
            var median_house_value = housing["median_house_value"];
            var median_income = housing["median_income"];

            // convert the house value range to thousands
            median_house_value /= 1000;

            // set up feature and label
            var feature = total_rooms.Values.ToArray();
            // var feature = median_income.Values.ToArray();
            var labels = median_house_value.Values.ToArray();

            // train the model
            Console.WriteLine("Training model....");
            var learner = new OrdinaryLeastSquares();
            var model = learner.Learn(feature, labels);

            // show results
            Console.WriteLine($"Slope:       {model.Slope}");
            Console.WriteLine($"Intercept:   {model.Intercept}");

            // validate the model
            var predictions = model.Transform(feature);
            var rmse = Math.Sqrt(new SquareLoss(labels).Loss(predictions));

            var range = Math.Abs(labels.Max() - labels.Min());
            Console.WriteLine($"Label range: {range}");
            Console.WriteLine($"RMSE:        {rmse} {rmse / range * 100:0.00}%");

            // generate plot arrays
            var x = feature.Concat(feature).ToArray();
            var y = predictions.Concat(labels).ToArray();

            // set up color array 
            var colors1 = Enumerable.Repeat(1, labels.Length).ToArray();
            var colors2 = Enumerable.Repeat(2, labels.Length).ToArray();
            var c = colors1.Concat(colors2).ToArray();

            // plot the data 
            var plot = new Scatterplot("Training", "feature", "label");
            plot.Compute(x, y, c);
            ScatterplotBox.Show(plot);

            Console.ReadLine();
        }
    }
}
