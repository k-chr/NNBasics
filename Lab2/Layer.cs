using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NNBasics.Lab1;
namespace NNBasics.Lab2
{
   //TODO implement layer behaviour and its updates - code flexibility.
   public class Layer : NeuralEngine
   {
      public GDEngineAnswer ProceedAndGetErrors(List<double> expectedAnswer, double alpha)
      {
         return null;
      }

      private IList<InputNeuron> _ins;
      private IList<OutputNeuron> _ons;

      public delegate double ActivationFunction(double x);
      public delegate double ActivationFunctionDerivative(double x);

      private ActivationFunction _activationFunction;
      private ActivationFunctionDerivative _activationFunctionDerivative;

      public Layer(IList<InputNeuron> ins, IList<OutputNeuron> ons, ActivationFunction fx, ActivationFunctionDerivative dfx)
      {
         _ins = ins;
         _ons = ons;
         _activationFunctionDerivative += dfx;
         _activationFunction += fx;
      }
   }
}
