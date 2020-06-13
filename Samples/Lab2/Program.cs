using System;
using System.Collections.Generic;
using NNBasicsUtilities.Core;
using NNBasicsUtilities.Extensions;
using static NNBasicsUtilities.Core.Utilities.UtilitiesFunctions;

namespace Lab2
{
	class Program
	{
		static void Main(string[] args)
		{
			#region Task1

			var networkTask1a =
				NeuralNetwork.Builder.AttachPredictionLayer(new List<double>(new[] {0.5}).ToMatrix()).WithAlpha(0.1)
				   .ApplyTheNameOfYourNetwork("Lab2_Task1a").BuildNetwork();
			networkTask1a.Train(new List<double>(new[] {0.8}).ToMatrix(),
				new List<double>(new[] {2.0}).ToMatrix(), 100);

			var networkTask1b =
				NeuralNetwork.Builder.AttachPredictionLayer(new List<double>(new[] {0.5}).ToMatrix()).WithAlpha(1)
				   .ApplyTheNameOfYourNetwork("Lab2_Task1b").BuildNetwork();
			networkTask1b.Train(new List<double>(new[] {0.8}).ToMatrix(),
				new List<double>(new[] {2.0}).ToMatrix(), 100);

			var networkTask1c =
				NeuralNetwork.Builder.AttachPredictionLayer(new List<double>(new[] {0.5}).ToMatrix()).WithAlpha(1)
				   .ApplyTheNameOfYourNetwork("Lab2_Task1c").BuildNetwork();
			networkTask1c.Train(new List<double>(new[] {0.8}).ToMatrix(),
				new List<double>(new[] {0.1}).ToMatrix(), 5000);

			#endregion

			#region Task2

			var expected = new List<List<double>>
			{
				new List<double>(new[] {1.0}),
				new List<double>(new[] {1.0}),
				new List<double>(new[] {0.0}),
				new List<double>(new[] {1.0})
			};

			var series = new List<List<double>>
			{
				new List<double> {8.5, 0.65, 1.2},
				new List<double> {9.5, 0.8, 1.3},
				new List<double> {9.9, 0.8, 0.5},
				new List<double> {9.0, 0.9, 1.0},
			};

			var networkTask2 =
				NeuralNetwork.Builder.AttachPredictionLayer(new List<double>(new[] {0.1, 0.2, -0.1}).ToMatrix())
				   .WithAlpha(0.01).ApplyTheNameOfYourNetwork("Lab2_Task2").BuildNetwork();
			networkTask2.Train(expected.ToMatrix(), series.ToMatrix(), 1000);

			#endregion

			#region Task3

			expected = new List<List<double>>
			{
				new List<double>(new[] {0.1, 1, 0.1}),
				new List<double>(new[] {0.0, 1, 0.0}),
				new List<double>(new[] {0.0, 0.0, 0.1}),
				new List<double>(new[] {0.1, 1, 0.2})
			};

			series = new List<List<double>>
			{
				new List<double> {8.5, 0.65, 1.2},
				new List<double> {9.5, 0.8, 1.3},
				new List<double> {9.9, 0.8, 0.5},
				new List<double> {9.0, 0.9, 1.0},
			};

			var mat = new List<List<double>>
			{
				new List<double>(new[] {0.1, 0.1, -0.3}),
				new List<double> {0.1, 0.2, 0.0},
				new List<double> {0.0, 1.3, 0.1}
			}.ToMatrix();

			var networkTask3 =
				NeuralNetwork.Builder.AttachPredictionLayer(mat).WithAlpha(0.01).ApplyTheNameOfYourNetwork("Lab2_Task3")
				   .BuildNetwork();
			networkTask3.Train(expected.ToMatrix(), series.ToMatrix(), 1000);

			#endregion

			#region Task4

			var (matSeries, matExpected) =
				ParseSeriesAndExpectedAnswersFromWeb(@"http://pduch.iis.p.lodz.pl/PSI/training_colors.txt");

			var (testSeries, testExpected) =
				ParseSeriesAndExpectedAnswersFromWeb(@"http://pduch.iis.p.lodz.pl/PSI/test_colors.txt");

			var networkTask4 =
				NeuralNetwork.Builder.AttachPredictionLayer(4, 3, 0.2, -0.2).WithAlpha(0.01)
				   .ApplyTheNameOfYourNetwork("Lab2_Task4_Colors").BuildNetwork();

			networkTask4.Train(matExpected, matSeries, 100);
			networkTask4.Test(testExpected, testSeries);

			#endregion

			Console.ReadKey();
		}
	}
}