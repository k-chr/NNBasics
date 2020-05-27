using NNBasicsUtilities.ActivationFunctions;
using NNBasicsUtilities.Core.Abstracts;
using NNBasicsUtilities.Core.Models;
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

      public FeedbackAnswer GetDeltas(EngineAnswer expectedAnswer)
      {
         var thisLayerResponse = LatestAnswer;
         var deltas = thisLayerResponse.Data - expectedAnswer.Data;
         var ans = new EngineAnswer() { Data = deltas };
         LatestDeltas = ans;
         return new FeedbackAnswer(){Deltas = ans, Ons = Ons};
      }

      public void Update()
      {
         UpdateWeights(new GdEngineAnswer(LatestAnswer, LatestDeltas));
      }

      public new EngineAnswer Proceed(Matrix input)
      {
         var ans = base.Proceed(input);
         if (_useSoftmax)
         {
            ans.Data = ans.Data.Softmax();
         }
         return ans;
      }

      public override string ToString()
      {
	      return Ons.ToString();
      }
   }
}