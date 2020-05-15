using System;
using System.Text;

namespace NNBasicsUtilities.Core.Models
{
   public class GdEngineAnswer : IDisposable
   {
      public EngineAnswer Output { get; }
      public EngineAnswer Deltas { get; }

      public GdEngineAnswer(EngineAnswer output, EngineAnswer deltas)
      {
         Output = output;
         Deltas = deltas;
      }

      public override string ToString()
      {
         var stringBuilder = new StringBuilder();
         stringBuilder.Append("---------------------------------------------")
            .Append(Output ?? new EngineAnswer())
            .Append("---------------------------------------------")
            .Append(Deltas ?? new EngineAnswer())
            .Append("---------------------------------------------");

         return stringBuilder.ToString();
      }

      public void Dispose()
      {
         GC.SuppressFinalize(this);
      }
   }
}