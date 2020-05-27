using System;

namespace NNBasicsUtilities.ActivationFunctions
{
   public static class ReluFunctions
   {
      public static double Relu(double value) => Math.Max(0, value);
      public static double ReluDerivative(double v) => v > 0 ? 1 : 0;
   }
}
