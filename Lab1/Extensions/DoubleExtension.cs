namespace NNBasics.Lab1.Extensions
{
   public static class DoubleExtension
   {
      public static void Swap(this ref double i, ref double value)
      {
         var tmp = i;
         i = value;
         value = tmp;
      }
   }
}
