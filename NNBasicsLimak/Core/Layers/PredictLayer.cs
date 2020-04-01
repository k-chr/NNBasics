using System.Collections.Generic;
using System.Linq;
using NNBasics.NNBasicsLimak.Core.Abstracts;
using NNBasics.NNBasicsLimak.Core.Models;
using NNBasics.NNBasicsLimak.Core.Neurons;

namespace NNBasics.NNBasicsLimak.Core.Layers
{
   public class PredictLayer : Layer
   {
      public PredictLayer(List<OutputNeuron> ons) : base(ons)
      {
      }

      public FeedbackAnswer GetDeltas(EngineAnswer expectedAnswer)
      {
         var thisLayerResponse = latestAnswer;
         var deltas = thisLayerResponse.Data.Zip(expectedAnswer.Data, (prediction, goal) => prediction - goal).ToList();
         var ans = new EngineAnswer() { Data = deltas };
         UpdateWeights(new GdEngineAnswer(thisLayerResponse, ans));
         return new FeedbackAnswer(){Deltas = ans, Ons = Ons};
      }
   }
}