namespace NNBasics.Lab1
{
   public class InputNeuron:NeuronBase
   {
      public static implicit operator InputNeuron(double val)
      {
         return new InputNeuron(){Value = val};
      }

      public double Value { get; set; }
   }
}
