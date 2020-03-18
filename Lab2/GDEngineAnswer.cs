using System.Text;
using NNBasics.Lab1;

namespace NNBasics.Lab2
{
   public class GDEngineAnswer
   {
      public EngineAnswer Output { get; }
      public EngineAnswer Deltas { get; }

      public GDEngineAnswer(EngineAnswer output, EngineAnswer deltas)
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
   }
}