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
			if (TestPending)
			{
				FlatMatrix.SubtractMatrix(TestAnswer, expectedAnswer, TestDeltas);
			}
			else
			{
				FlatMatrix.SubtractMatrix(LatestAnswer, expectedAnswer, LatestDeltas);
			}


			return (TestPending ? TestDeltas : LatestDeltas, Ons);
		}

		public void Update()
		{
			UpdateWeights();
		}

		public new void Proceed(FlatMatrix input)
		{
			base.Proceed(input);
			if (_useSoftmax && !TestPending)
			{
				LatestAnswer.Softmax();
			}
		}

		public override string ToString()
		{
			return Ons.ToString();
		}
	}
}