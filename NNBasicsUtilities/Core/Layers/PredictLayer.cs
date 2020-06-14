using System;
using System.Diagnostics;
using NNBasicsUtilities.ActivationFunctions;
using NNBasicsUtilities.Core.Abstracts;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;

namespace NNBasicsUtilities.Core.Layers
{
	public class PredictLayer : Layer
	{
		private readonly bool _useSoftmax;

		public PredictLayer(Matrix ons, bool useSoftmax = false) : base(ons)
		{
			_useSoftmax = useSoftmax;
		}

		public (Matrix, Matrix) GetDeltas(Matrix expectedAnswer)
		{
			var thisLayerResponse = LatestAnswer;
			var deltas = thisLayerResponse - expectedAnswer;
			var ans = deltas;
			LatestDeltas = ans;
			return (ans, Ons);
		}

		public void Update()
		{
			UpdateWeights(LatestDeltas);
		}

		public new Matrix Proceed(Matrix input)
		{
			//var time = Stopwatch.GetTimestamp();

			var ans = base.Proceed(input);
			if (_useSoftmax)
			{
				ans = ans.Softmax();
			}

			//time = Stopwatch.GetTimestamp() - time;
			//Console.WriteLine($"Proceed time in predict layer: {time}");
			return ans;
		}

		public override string ToString()
		{
			return Ons.ToString();
		}
	}
}