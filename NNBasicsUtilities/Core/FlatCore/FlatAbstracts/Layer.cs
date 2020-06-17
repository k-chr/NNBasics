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
		protected FlatMatrix LayerWeightDelta;
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

		protected void Proceed(FlatMatrix input)
		{
			//var time = Stopwatch.GetTimestamp();
			Ins.Assign(input);
			//time = Stopwatch.GetTimestamp() - time;
			//Console.WriteLine($"Layer Ins assignment time: {time}");
			//time = Stopwatch.GetTimestamp();
			FlatNN.NeuralEngine.Proceed(input,  Ons, LatestAnswer);
			//time = Stopwatch.GetTimestamp() - time;
			//Console.WriteLine($"Layer proceed time: {time}");
			//time = Stopwatch.GetTimestamp();
			
			//time = Stopwatch.GetTimestamp() - time;
			//Console.WriteLine($"Layer LatestAnswer assignment time: {time}");
		}

		protected Layer(FlatMatrix ons, int inputRows)
		{
			Ons = ons;
			Ins = FlatMatrix.Of(inputRows, ons.Cols);
			LatestAnswer = FlatMatrix.Of(inputRows, ons.Rows);
			LayerWeightDelta = FlatMatrix.Of(ons.Rows, ons.Cols);
			LatestDeltas = FlatMatrix.Of(inputRows, ons.Rows);
		}

		protected void UpdateWeights()
		{
			FlatMatrix.Multiply(LatestDeltas.T(), Ins, LayerWeightDelta);
			LayerWeightDelta.ApplyFunction(d => d * Alpha);
			Ons.SubtractMatrix(LayerWeightDelta);
		}
	}
}