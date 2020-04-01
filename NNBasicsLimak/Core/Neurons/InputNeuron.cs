using NNBasics.NNBasicsLimak.Core.Abstracts;

namespace NNBasics.NNBasicsLimak.Core.Neurons
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
