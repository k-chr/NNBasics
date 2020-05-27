using System.Text;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;

namespace NNBasicsUtilities.Core.Models
{
   public class EngineAnswer
   {
      public Matrix Data { get; set; }

      public override string ToString()
      {
         StringBuilder stringBuilder = new StringBuilder("");
         stringBuilder.Append("[ ");
         foreach (var elem in Data)
         {
            stringBuilder.Append(elem).Append(' ');
         }

         stringBuilder.Append("]\n");

         return stringBuilder.ToString();
      }
   }
}
