using NNBasicsUtilities.Core.Utilities.UtilityTypes;

namespace NNBasicsUtilities.Core.Models
{
   public class FeedbackAnswer
   {
      public Matrix Ons { get; set; }
      public EngineAnswer Deltas { get; set; }
   }
}
