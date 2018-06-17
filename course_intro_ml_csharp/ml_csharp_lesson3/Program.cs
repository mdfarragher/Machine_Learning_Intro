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

namespace ml_csharp_lesson3
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

            // convert the house value range to thousands
            housing["median_house_value"] /= 1000;

            // shuffle row indices
            var rnd = new Random();
            var indices = Enumerable.Range(0, housing.Rows.KeyCount).OrderBy(v => rnd.NextDouble());

            // shuffle the frame using the indices
            housing = housing.IndexRowsWith(indices).SortRowsByKey();

            // create the rooms_per_person feature
            //housing.AddColumn("rooms_per_person", housing["total_rooms"] / housing["population"]);
            housing.AddColumn("rooms_per_person",
               (housing["total_rooms"] / housing["population"]).Select(v => v.Value <= 4.0 ? v.Value : 4.0));

            // calculate the correlation matrix
            var correlation = Measures.Correlation(housing.ToArray2D<double>());

            // show the correlation matrix
            Console.WriteLine(housing.ColumnKeys.ToArray().ToString<string>());
            Console.WriteLine(correlation.ToString<double>("0.0"));

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

            // show the data frame on the console
            housing.Print();

            // calculate rooms_per_person histogram
            var histogram = new Histogram();
            histogram.Compute(housing["rooms_per_person"].Values.ToArray(), 0.1); // use 1.0 without clipping

            // draw the histogram
            var x = new List<double>();
            var y = new List<double>();
            for (int i = 0; i < histogram.Values.Length; i++)
            {
                var xcor = histogram.Bins[i].Range.Min;
                x.AddRange(from n in Enumerable.Range(0, histogram.Values[i]) select xcor);
                y.AddRange(from n in Enumerable.Range(0, histogram.Values[i]) select n * 1.0);
            }
            var plot = new Scatterplot("", "rooms per person", "count");
            plot.Compute(x.ToArray(), y.ToArray());
            ScatterplotBox.Show(plot);

            Console.ReadLine();
        }
    }
}
