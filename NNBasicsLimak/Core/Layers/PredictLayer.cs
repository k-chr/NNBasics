using System.Collections.Generic;
using System.Linq;
using System.Text;
using NNBasics.NNBasicsLimak.ActivationFunctions;
using NNBasics.NNBasicsLimak.Core.Abstracts;
using NNBasics.NNBasicsLimak.Core.Models;
using NNBasics.NNBasicsLimak.Core.Neurons;

namespace NNBasics.NNBasicsLimak.Core.Layers
{
   public class PredictLayer : Layer
   {
      private bool _useSoftmax;

      public PredictLayer(List<OutputNeuron> ons, bool useSoftmax = false) : base(ons)
      {
         _useSoftmax = useSoftmax;
      }

      public FeedbackAnswer GetDeltas(EngineAnswer expectedAnswer)
      {
         var thisLayerResponse = LatestAnswer;
         var deltas = thisLayerResponse.Data.Zip(expectedAnswer.Data, (prediction, goal) => prediction - goal).ToList();
         var ans = new EngineAnswer() { Data = deltas };
         UpdateWeights(new GdEngineAnswer(thisLayerResponse, ans));
         return new FeedbackAnswer(){Deltas = ans, Ons = Ons};
      }

      public new EngineAnswer Proceed(List<InputNeuron> input)
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
         var builder = new StringBuilder();
         foreach (var outputNeuron in Ons)
         {
            builder.Append(new EngineAnswer(){Data = outputNeuron.Weights.Select(d=>d).ToList()}+"\n");
         }

         return builder.ToString();
      }
   }
}