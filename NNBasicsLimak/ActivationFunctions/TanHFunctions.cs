using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNBasics.NNBasicsLimak.ActivationFunctions
{
   public static class TanHFunctions
   {
      public static double TanH(double input) => Math.Tanh(input);
      public static double TanHDerivative(double input) => 1 - TanH(input) * TanH(input);
   }
}
