using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNBasics.NNBasicsLimak.Core.Models
{
   public class EngineAnswer : IDisposable
   {
      public List<double> Data { get; set; }

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

      public void Dispose()
      {
         GC.SuppressFinalize(this);
      }
   }
}
