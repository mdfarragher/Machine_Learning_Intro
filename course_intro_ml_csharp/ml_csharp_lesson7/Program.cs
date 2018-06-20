using Accord.Controls;
using Accord.Math;
using Accord.Math.Optimization.Losses;
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

namespace ml_csharp_lesson7
{
    /// <summary>
    /// The main application class.
    /// </summary>
    public class Program
    {
        // helper method to generate ranges
        private static IEnumerable<(int Min, int Max)> RangeFromTo(int from, int to, int step = 1)
        {
            for (int i = from; i <= to; i += step)
                yield return (Min: i, Max: i + step);
        }

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

            // create the median_high_house_value feature
            housing.AddColumn("median_high_house_value",
                housing["median_house_value"].Select(v => v.Value >= 265000 ? 1.0 : 0.0));

            // create one-hot vectors for longitude and latitude
            var vectors_long =
                from l in housing["longitude"].Values
                select Vector.Create<double>(
                    1,
                    (from b in RangeFromTo(-125, -114, 1)
                     select l >= b.Min && l < b.Max).ToArray());
            var vectors_lat =
                from l in housing["latitude"].Values
                select Vector.Create<double>(
                    1,
                    (from b in RangeFromTo(32, 43, 1)
                     select l >= b.Min && l < b.Max).ToArray());

            // multiply vectors and create columns
            var vectors_cross =
                vectors_long.Zip(vectors_lat, (lng, lat) => lng.Outer(lat));
            for (var i = 0; i < 12; i++)
                for (var j = 0; j < 12; j++)
                    housing.AddColumn($"location {i},{j}", from v in vectors_cross select v[i, j]);

            // set up model columns
            var columns = (from i in Enumerable.Range(0, 12)
                           from j in Enumerable.Range(0, 12)
                           select $"location {i},{j}").ToList();
            columns.Add("housing_median_age");
            columns.Add("total_rooms");
            columns.Add("total_bedrooms");
            columns.Add("population");
            columns.Add("households");
            columns.Add("median_income");

            // create training, validation, and test partitions
            var training = housing.Rows[Enumerable.Range(0, 12000)];
            var validation = housing.Rows[Enumerable.Range(12000, 2500)];
            var test = housing.Rows[Enumerable.Range(14500, 2500)];

            ////////////////////////////////////////////////////////////////////////
            // Without regularization
            ////////////////////////////////////////////////////////////////////////

            // train the model
            var learner = new IterativeReweightedLeastSquares<LogisticRegression>()
            {
                MaxIterations = 50,
                Regularization = 0 
            };
            var regression = learner.Learn(
                training.Columns[columns].ToArray2D<double>().ToJagged(),
                training["median_high_house_value"].Values.ToArray());

            // display training results
            Console.WriteLine("TRAINING WITHOUT REGULARIZATION");
            Console.WriteLine($"Weights:     {regression.Weights.ToString<double>("0.00")}");
            Console.WriteLine($"Intercept:   {regression.Intercept}");
            Console.WriteLine();

            // plot a histogram of the nonzero weights
            var histogram = new Histogram();
            histogram.Compute(regression.Weights, 0.1); // set to 1.0 when regularization is disabled

            // draw the histogram
            var x = new List<double>();
            var y = new List<double>();
            for (int i = 0; i < histogram.Values.Length; i++)
            {
                var xcor = histogram.Bins[i].Range.Min;
                x.AddRange(from n in Enumerable.Range(0, histogram.Values[i]) select xcor);
                y.AddRange(from n in Enumerable.Range(0, histogram.Values[i]) select n * 1.0);
            }
            var plot = new Scatterplot("Without Regularization", "prediction", "count");
            plot.Compute(x.ToArray(), y.ToArray());
            ScatterplotBox.Show(plot);

            ////////////////////////////////////////////////////////////////////////
            // With regularization
            ////////////////////////////////////////////////////////////////////////

            // train the model
            learner = new IterativeReweightedLeastSquares<LogisticRegression>()
            {
                MaxIterations = 50,
                Regularization = 50
            };
            regression = learner.Learn(
                training.Columns[columns].ToArray2D<double>().ToJagged(),
                training["median_high_house_value"].Values.ToArray());

            // display training results
            Console.WriteLine("TRAINING WITH REGULARIZATION");
            Console.WriteLine($"Weights:     {regression.Weights.ToString<double>("0.00")}");
            Console.WriteLine($"Intercept:   {regression.Intercept}");
            Console.WriteLine();

            // plot a histogram of the nonzero weights
            histogram = new Histogram();
            histogram.Compute(regression.Weights, 0.1); // set to 1.0 when regularization is disabled

            // draw the histogram
            x = new List<double>();
            y = new List<double>();
            for (int i = 0; i < histogram.Values.Length; i++)
            {
                var xcor = histogram.Bins[i].Range.Min;
                x.AddRange(from n in Enumerable.Range(0, histogram.Values[i]) select xcor);
                y.AddRange(from n in Enumerable.Range(0, histogram.Values[i]) select n * 1.0);
            }
            plot = new Scatterplot("With Regularization", "prediction", "count");
            plot.Compute(x.ToArray(), y.ToArray());
            ScatterplotBox.Show(plot);

            Console.ReadLine();
        }
    }
}
