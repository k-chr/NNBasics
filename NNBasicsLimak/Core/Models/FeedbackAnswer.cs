using System.Collections.Generic;
using NNBasics.NNBasicsLimak.Core.Neurons;

namespace NNBasics.NNBasicsLimak.Core.Models
{
   public class FeedbackAnswer
   {
      public List<OutputNeuron> Ons { get; set; }
      public EngineAnswer Deltas { get; set; }
   }
}
