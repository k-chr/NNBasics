namespace NNBasics.NNBasicsLimak.Extensions
{
   public static class DoubleExtensions
   {
      public static void Swap(this ref double i, ref double value)
      {
         var tmp = i;
         i = value;
         value = tmp;
      }

      public static bool Between(this ref double i, double min, double max) => i >= min && i <= max;
   }
}
