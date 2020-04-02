using System;
using System.Collections.Generic;
using System.Linq;
using NNBasics.NNBasicsLimak.Extensions;

namespace NNBasics.NNBasicsLimak.ActivationFunctions
{
   public static class SoftmaxFunction
   {
      public static List<double> Softmax(this List<double> input) => input.Select(Math.Exp).ToList().Normalize();
   }
}
