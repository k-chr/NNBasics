using System;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core.FlatCore.FlatAbstracts
{
	public abstract class Layer
	{
		public FlatMatrix Answer => TestPending ? TestAnswer : LatestAnswer;
		protected FlatMatrix Ins;
		protected FlatMatrix TestIns;
		protected readonly FlatMatrix Ons;
		protected readonly FlatMatrix LatestAnswer;
		protected readonly FlatMatrix TestAnswer;
		protected readonly FlatMatrix TestDeltas;
		protected readonly FlatMatrix LatestDeltas;
		private readonly FlatMatrix _layerWeightDelta;
		private double _alpha;
		protected bool TestPending;

		public FlatMatrix Weights => Ons;

		public void SetTestSession(bool value)
		{
			TestPending = value;
		}

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
			if (TestPending)
			{
				TestIns.Assign(input);
			}
			else
			{
				Ins.Assign(input);
			}

			FlatNN.NeuralEngine.Proceed(input, Ons, TestPending ? TestAnswer : LatestAnswer);

		}

		protected Layer(FlatMatrix ons, int inputRows)
		{
			Ons = ons;
			Ins = FlatMatrix.Of(inputRows, ons.Cols);
			LatestAnswer = FlatMatrix.Of(inputRows, ons.Rows);
			_layerWeightDelta = FlatMatrix.Of(ons.Rows, ons.Cols);
			LatestDeltas = FlatMatrix.Of(inputRows, ons.Rows);
			TestIns = FlatMatrix.Of(1, ons.Cols);
			TestAnswer = FlatMatrix.Of(1, ons.Rows);
			TestDeltas = FlatMatrix.Of(1, ons.Rows);
		}

		protected void UpdateWeights()
		{
			if (TestPending)
			{
				FlatMatrix.Multiply(TestDeltas.T(), TestIns, _layerWeightDelta);
			}
			else
			{
				FlatMatrix.Multiply(LatestDeltas.T(), Ins, _layerWeightDelta);
			}

			_layerWeightDelta.ApplyFunction(d => d * Alpha);
			Ons.SubtractMatrix(_layerWeightDelta);
		}
	}
}