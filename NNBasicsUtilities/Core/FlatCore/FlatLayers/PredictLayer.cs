using NNBasicsUtilities.ActivationFunctions;
using NNBasicsUtilities.Core.FlatCore.FlatAbstracts;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;

namespace NNBasicsUtilities.Core.FlatCore.FlatLayers
{
	public class PredictLayer : Layer
	{
		private readonly bool _useSoftmax;

		public PredictLayer(FlatMatrix ons, int inputRows, bool useSoftmax = false) : base(ons, inputRows)
		{
			_useSoftmax = useSoftmax;
		}

		public (FlatMatrix, FlatMatrix) GetDeltas(FlatMatrix expectedAnswer)
		{
			FlatMatrix.SubtractMatrix(LatestAnswer, expectedAnswer, LatestDeltas);
			return (LatestDeltas, Ons);
		}

		public void Update()
		{
			UpdateWeights();
		}

		public new void Proceed(FlatMatrix input)
		{
			//var time = Stopwatch.GetTimestamp();

			base.Proceed(input);
			if (_useSoftmax)
			{
				LatestAnswer.Softmax();
			}

			//time = Stopwatch.GetTimestamp() - time;
			//Console.WriteLine($"Proceed time in predict layer: {time}");
		}

		public override string ToString()
		{
			return Ons.ToString();
		}
	}
}