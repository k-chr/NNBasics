using NNBasicsUtilities.Core.Abstracts;

namespace NNBasicsUtilities.Core.Neurons
{
   public class InputNeuron : NeuronBase
   {
      public static implicit operator InputNeuron(double val) => new InputNeuron(){Value = val};
      public double Value { get; set; }
   }
}
