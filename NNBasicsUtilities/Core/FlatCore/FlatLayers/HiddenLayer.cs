using System;
using NNBasicsUtilities.Core.FlatCore.FlatAbstracts;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core.FlatCore.FlatLayers
{
	public class HiddenLayer : Layer
	{
		public delegate double ActivationFunction(double x);

		public delegate double ActivationFunctionDerivative(double x);

		private readonly ActivationFunction _activationFunction;
		private readonly ActivationFunctionDerivative _activationFunctionDerivative;
		private readonly double _dropoutRate;
		private readonly bool _applyDropout;
		protected FlatMatrix Dropout;
		protected double[] DropoutVec;

		public HiddenLayer(FlatMatrix ons, int inputRows, Func<double, double> fx = null,
			Func<double, double> dfx = null,
			bool dropout = false, double dropoutRate = 0) : base(ons, inputRows)
		{
			_activationFunctionDerivative += d => dfx?.Invoke(d) ?? 1;
			_applyDropout = dropout;

			_activationFunction += d => fx?.Invoke(d) ?? d;

			if (_applyDropout)
			{
				var len = ons.Cols;
				var fill = (int) (len * dropoutRate);
				var trueDropout = fill / (double) len;
				_dropoutRate = trueDropout;
				GenerateDropout();
			}
		}

		private void GenerateDropout()
		{
			Dropout = FlatMatrix.Of(Ins.Rows, Ons.Rows);
			DropoutVec = new double[Ons.Rows];

			var count = Ons.Cols;
			var fill = _dropoutRate * count;

			for (var i = 0; i < count; ++i)
			{
				DropoutVec[i] = i < fill ? 1 : 0;
			}
		}

		private void ShuffleDropout()
		{
			for (var i = 0; i < Ins.Rows; ++i)
			{
				DropoutVec.Shuffle();
				Dropout[i] = DropoutVec;
			}
		}

		public (FlatMatrix, FlatMatrix) BackPropagate(FlatMatrix deltas, FlatMatrix ons)
		{
			FlatMatrix.Multiply(deltas, ons, LatestDeltas);
			LatestAnswer.ApplyFunction(d => _activationFunctionDerivative(d));
			LatestDeltas.HadamardProduct(LatestAnswer);

			if (_applyDropout)
			{
				LatestDeltas.HadamardProduct(Dropout);
			}

			return (LatestDeltas, Ons);
		}

		public void Update()
		{
			UpdateWeights();
		}

		public new void Proceed(FlatMatrix ins)
		{
			//var time = Stopwatch.GetTimestamp();
			//var subTime = time;
			base.Proceed(ins);
			//subTime = Stopwatch.GetTimestamp() - subTime;
			//Console.WriteLine($"Proceed time in hidden layer calling base method: {subTime}");
			//subTime = Stopwatch.GetTimestamp();
			LatestAnswer.ApplyFunction(d => _activationFunction(d));
			//subTime = Stopwatch.GetTimestamp() - subTime;
			//Console.WriteLine($"Proceed time in hidden layer applying activation function: {subTime}");
			if (_applyDropout)
			{
				ShuffleDropout();

				LatestAnswer.HadamardProduct(Dropout);
				LatestAnswer.MultiplyByAlpha(1 / _dropoutRate);
			}

			//time = Stopwatch.GetTimestamp() - time;
			//Console.WriteLine($"Proceed time in hidden layer in total: {time}");
		}

		public override string ToString()
		{
			return Ons.ToString();
		}
	}
}