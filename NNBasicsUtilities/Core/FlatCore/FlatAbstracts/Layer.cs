using System;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;
namespace NNBasicsUtilities.Core.FlatCore.FlatAbstracts
{
	public abstract class Layer
	{
		protected FlatMatrix Ins;
		protected readonly FlatMatrix Ons;
		protected FlatMatrix LatestAnswer;
		protected FlatMatrix LatestDeltas;

		private double _alpha;

		public FlatMatrix Weights => Ons;

		public double Alpha
		{
			get => _alpha;
			set
			{
				const double min = 0.0;
				const double max = 1;
				if (!value.Between(min, max))
				{
					throw new ArgumentException($"Provided value: {value} is out of range of <{min}, {max}>");
				}

				_alpha = value;
			}
		}

		protected FlatMatrix Proceed(FlatMatrix input)
		{
			//var time = Stopwatch.GetTimestamp();
			Ins = input;
			//time = Stopwatch.GetTimestamp() - time;
			//Console.WriteLine($"Layer Ins assignment time: {time}");
			//time = Stopwatch.GetTimestamp();
			var ans = FlatNN.NeuralEngine.Proceed(input,  Ons);
			//time = Stopwatch.GetTimestamp() - time;
			//Console.WriteLine($"Layer proceed time: {time}");
			//time = Stopwatch.GetTimestamp();
			LatestAnswer = ans;
			//time = Stopwatch.GetTimestamp() - time;
			//Console.WriteLine($"Layer LatestAnswer assignment time: {time}");

			return FlatMatrix.Of(ans);
		}

		protected Layer(FlatMatrix ons)
		{
			Ons = ons;
		}

		protected void UpdateWeights(FlatMatrix deltas)
		{
			var mat = deltas.T() * Ins;
			mat.ApplyFunction(d => d * Alpha);
			Ons.SubtractMatrix(mat);
		}
	}
}