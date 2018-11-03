using Accord.Controls;
using Accord.Math;
using Accord.Math.Optimization.Losses;
using Accord.Statistics;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Statistics.Visualizations;
using Deedle;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ml_csharp_lesson4
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
            ////////////////////////////////////////////////////////////////////////////////
            // Code from lesson3, with correlation matrix and histogram commented out
            ////////////////////////////////////////////////////////////////////////////////

            // get data
            Console.WriteLine("Loading data....");
            var path = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\california_housing.csv"));
            var housing = Frame.ReadCsv(path, separators: ",");
            housing = housing.Where(kv => ((decimal)kv.Value["median_house_value"]) < 500000);

            // convert the house value range to thousands
            housing["median_house_value"] /= 1000;

            // shuffle row indices
            var rnd = new Random();
            var indices = Enumerable.Range(0, housing.Rows.KeyCount).OrderBy(v => rnd.NextDouble());

            // shuffle the frame using the indices
            housing = housing.IndexRowsWith(indices).SortRowsByKey();

            // create the rooms_per_person feature
            housing.AddColumn("rooms_per_person",
               (housing["total_rooms"] / housing["population"]).Select(v => v.Value <= 4.0 ? v.Value : 4.0));

            //// calculate the correlation matrix
            //var correlation = Measures.Correlation(housing.ToArray2D<double>());

            //// show the correlation matrix
            //Console.WriteLine(housing.ColumnKeys.ToArray().ToString<string>());
            //Console.WriteLine(correlation.ToString<double>("0.0"));

            // calculate binned latitudes
            var bins = from b in Enumerable.Range(32, 10) select (Min: b, Max: b + 1);
            var binned_latitude =
                from l in housing["latitude"].Values
                let bin = (from b in bins where l >= b.Min && l < b.Max select b)
                select bin.First().Min;

            // add one-hot encoding columns
            foreach (var i in Enumerable.Range(32, 10))
            {
                housing.AddColumn($"latitude {i}-{i + 1}",
                    from l in binned_latitude
                    select l == i ? 1 : 0);
            }

            // drop the latitude column
            housing.DropColumn("latitude");

            //// show the data frame on the console
            //housing.Print();

            //// calculate rooms_per_person histogram
            //var histogram = new Histogram();
            //histogram.Compute(housing["rooms_per_person"].Values.ToArray(), 0.1); // use 1.0 without clipping

            //// draw the histogram
            //var x = new List<double>();
            //var y = new List<double>();
            //for (int i = 0; i < histogram.Values.Length; i++)
            //{
            //    var xcor = histogram.Bins[i].Range.Min;
            //    x.AddRange(from n in Enumerable.Range(0, histogram.Values[i]) select xcor);
            //    y.AddRange(from n in Enumerable.Range(0, histogram.Values[i]) select n * 1.0);
            //}
            //var plot = new Scatterplot("", "rooms per person", "count");
            //plot.Compute(x.ToArray(), y.ToArray());
            //ScatterplotBox.Show(plot);

            ////////////////////////////////////////////////////////////////////////////////
            // Lesson4 code starts here
            ////////////////////////////////////////////////////////////////////////////////

            // create training, validation, and test frames
            var training = housing.Rows[Enumerable.Range(0, 12000)];
            var validation = housing.Rows[Enumerable.Range(12000, 2500)];
            var test = housing.Rows[Enumerable.Range(14500, 2500)];

            // set up model columns
            var columns = (from i in Enumerable.Range(32, 10)
                           select $"latitude {i}-{i + 1}").ToList();
            columns.Add("median_income");
            columns.Add("rooms_per_person");

            // train the model
            var learner = new OrdinaryLeastSquares();
            var regression = learner.Learn(
                training.Columns[columns].ToArray2D<double>().ToJagged(),  // features
                training["median_house_value"].Values.ToArray());          // labels

            // display training results
            Console.WriteLine("TRAINING RESULTS");
            Console.WriteLine($"Weights:     {regression.Weights.ToString<double>("0.00")}");
            Console.WriteLine($"Intercept:   {regression.Intercept}");
            Console.WriteLine();

            // validate the model
            var predictions = regression.Transform(
                validation.Columns[columns].ToArray2D<double>().ToJagged());

            // display validation results
            var labels = validation["median_house_value"].Values.ToArray();
            var rmse = Math.Sqrt(new SquareLoss(labels).Loss(predictions));
            Console.WriteLine("VALIDATION RESULTS");
            Console.WriteLine($"RMSE:        {rmse:0.00}");
            Console.WriteLine();

            // test the model
            var predictions_test = regression.Transform(
                test.Columns[columns].ToArray2D<double>().ToJagged());

            // display test results
            var labels_test = test["median_house_value"].Values.ToArray();
            rmse = Math.Sqrt(new SquareLoss(labels_test).Loss(predictions_test));
            Console.WriteLine("TEST RESULTS");
            Console.WriteLine($"RMSE:        {rmse:0.00}");
            Console.WriteLine();

            Console.ReadLine();
        }
    }
}
