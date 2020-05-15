using System;
using System.Collections.Generic;
using NNBasicsUtilities.Core.Neurons;

namespace NNBasicsUtilities.Core.Models
{
   public class FeedbackAnswer : IDisposable
   {
      public List<OutputNeuron> Ons { get; set; }
      public EngineAnswer Deltas { get; set; }
      public void Dispose()
      {
         GC.SuppressFinalize(this);
      }
   }
}
