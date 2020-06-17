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
		private FlatMatrix _dropout;
		private double[] _dropoutVec;

		public HiddenLayer(FlatMatrix ons, int inputRows, Func<double, double> fx = null,
			Func<double, double> dfx = null,
			bool dropout = false, double dropoutRate = 0) : base(ons, inputRows)
		{
			_activationFunctionDerivative += d => dfx?.Invoke(d) ?? 1;
			_applyDropout = dropout;

			_activationFunction += d => fx?.Invoke(d) ?? d;

			if (_applyDropout && !TestPending)
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
			_dropout = FlatMatrix.Of(Ins.Rows, Ons.Rows);
			_dropoutVec = new double[Ons.Rows];

			var count = Ons.Rows;
			var fill = _dropoutRate * count;

			for (var i = 0; i < count; ++i)
			{
				_dropoutVec[i] = i < fill ? 1 : 0;
			}
		}

		private void ShuffleDropout()
		{
			for (var i = 0; i < Ins.Rows; ++i)
			{
				_dropoutVec.Shuffle();
				_dropout[i] = _dropoutVec;
			}
		}

		public (FlatMatrix, FlatMatrix) BackPropagate(FlatMatrix deltas, FlatMatrix ons)
		{
			if (!TestPending)
			{
				FlatMatrix.Multiply(deltas, ons, LatestDeltas);
				LatestAnswer.ApplyFunction(d => _activationFunctionDerivative(d));
				LatestDeltas.HadamardProduct(LatestAnswer);

				if (_applyDropout)
				{
					LatestDeltas.HadamardProduct(_dropout);
				}
			}
			else
			{
				FlatMatrix.Multiply(deltas, ons, TestDeltas);
				TestAnswer.ApplyFunction(d => _activationFunctionDerivative(d));
				TestDeltas.HadamardProduct(TestAnswer);
			}

			return (TestPending ? TestDeltas : LatestDeltas, Ons);
		}

		public void Update()
		{
			UpdateWeights();
		}

		public new void Proceed(FlatMatrix ins)
		{
			base.Proceed(ins);
			if (TestPending)
			{
				TestAnswer.ApplyFunction(d => _activationFunction(d));
			}
			else
			{
				LatestAnswer.ApplyFunction(d => _activationFunction(d));
			}

			if (_applyDropout && !TestPending)
			{
				ShuffleDropout();

				LatestAnswer.HadamardProduct(_dropout);
				LatestAnswer.MultiplyByAlpha(1 / _dropoutRate);
			}

		}

		public override string ToString()
		{
			return Ons.ToString();
		}
	}
}