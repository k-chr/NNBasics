using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNBasics.Lab1;

namespace NNBasics.Lab2
{
   //TODO Implement Neural network behaviour and add Layers members and backward propagation with activation functions
   public class NeuralNetwork
   {
      private int _currentIteration = 0;
      private double _alpha;
      private int _iterations = 0;
      private List<InputNeuron> _ins;

      public IList<double> ExpectedOutputDoubles { get; set; }
      public double Alpha 
      {
         get => _alpha;
         set => _alpha = value > 1 || value < 0 ? throw new ArgumentException("Wrong alpha parameter") : (value);
      }

      public int Iterations
      {
         get => _iterations;
         set
         {
            _iterations = value;
            _currentIteration = 0;
         }
      }

      public List<InputNeuron> InputNeurons { get; set; }
      public List<OutputNeuron> OutputNeurons { get; set; }

      public int Learn()
      {

         return _iterations;
      }

      private void UpdateWeights(GDEngineAnswer answer)
      {
         var deltas = answer.Deltas;
         var i = 0;
         foreach (var outputNeuron in OutputNeurons)
         {
            var collection = _ins.Select(inputNeuron => inputNeuron.Value * deltas.Data[i]).ToList();
            outputNeuron.Weights = outputNeuron.Weights.Join(
                                       collection, 
                                       weight => weight,
                                       weightDelta => weightDelta,
                                       (weight, weightDelta) => weight - weightDelta * Alpha
                                  ).ToList();
            ++i;
         }

      }

      public int Learn(int iterations)
      {

         return iterations;
      }
   }
}
