using NNBasicsUtilities.ActivationFunctions;
using NNBasicsUtilities.Core.FlatCore.FlatAbstracts;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;

namespace NNBasicsUtilities.Core.FlatCore.FlatLayers
{
   public class PredictLayer : Layer
   {
	   private readonly bool _useSoftmax;

	   public PredictLayer(FlatMatrix ons, bool useSoftmax = false) : base(ons)
	   {
		   _useSoftmax = useSoftmax;
	   }

	   public (FlatMatrix, FlatMatrix) GetDeltas(FlatMatrix expectedAnswer)
	   {
		   var thisLayerResponse = LatestAnswer;
		   LatestDeltas = thisLayerResponse - expectedAnswer;
		   
		   return (LatestDeltas, Ons);
	   }

	   public void Update()
	   {
		   UpdateWeights(LatestDeltas);
	   }

	   public new FlatMatrix Proceed(FlatMatrix input)
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
