﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using NNBasicsUtilities.Core.Models;
using NNBasicsUtilities.Core.Neurons;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core
{
   public class NeuralEngine
   {
      public static EngineAnswer Proceed(List<InputNeuron> iNs, List<OutputNeuron> oNs)
      {
         var rV = new List<double>();
         
         foreach (var outputNeuron in oNs)
         {
            double tmp = 0;
            var i = 0;
            foreach (var inputNeuron in iNs)
            {
               tmp += (inputNeuron.Value * outputNeuron.Weights[i++]);
            }
            rV.Add(tmp);
         }

         var ans = new EngineAnswer() { Data = rV };

         return ans;
      }

      public static EngineAnswer Proceed(List<InputNeuron> ins, List<List<double>> weights)
      {
         var outputNeurons = weights.Select(doubles => new OutputNeuron() { Weights = doubles }).ToList();
         return Proceed(ins, outputNeurons);
      }

      public static EngineAnswer Proceed(List<InputNeuron> ins, Tuple<List<List<double>>, List<List<double>>> weights)
      {
         var hiddenOutput = Proceed(ins, weights.Item1);
         List<InputNeuron> iNs = hiddenOutput.Data.Select(data => new InputNeuron() { Value = data }).ToList();
         return Proceed(iNs, weights.Item2);
      }

      public static EngineAnswer Proceed(List<InputNeuron> iNs, List<List<double>> outputWeights = null, bool randomizeLayers = false, double min = 0, double max = 0, List<int> layersSizes = null)
      {
         if (!(outputWeights is null) && outputWeights.Any(weights => weights.Count != outputWeights[0].Count))
         {
            throw new ArgumentException("Provided arguments are invalid");
         }

         if (randomizeLayers)
         {
            var layers = GenerateRandomLayers(iNs.Count, min, max, layersSizes);
            EngineAnswer output = null;
            foreach (var layer in layers)
            {
               output = Proceed(iNs, layer);
               iNs = output.Data.Select(data => new InputNeuron() { Value = data }).ToList();
            }

            return output;
         }

         return Proceed(iNs, outputWeights);
      }

      public static Matrix GenerateRandomLayer(int cols, int rows, double min, double max)
      {
         var collection = new List<List<double>>();
         using var provider = new RNGCryptoServiceProvider();

         for (var i = 0; i < rows; ++i)
         {
            var row = new List<double>();
            for (var idx = 0; idx < cols; ++idx)
            {
               var bytes = new byte[2];
               double res;
               do
               {
                  provider.GetBytes(bytes);
                  res = BitConverter.ToInt16(bytes, 0) / 30000.0;
               } while (res < min || res > max || Math.Abs(res) < max / 100.0 || double.IsNaN(res));

               row.Add(res);

            }

            collection.Add(row);
         }

         return collection.ToMatrix();
      }

      private static List<List<List<double>>> GenerateRandomLayers(int countOfInputNeurons, double min, double max, List<int> sizes)
      {
         if (min > max)
         {
            min.Swap(ref max);
         }

         if (sizes is null || sizes.Any(i => i < 0))
         {
            throw new ArgumentException("Provided data are not valid to start generate hidden layers");
         }
         var layers = new List<List<List<double>>>();

         using var provider = new RNGCryptoServiceProvider();
         var k = sizes.Count;
         for (var i = 0; i < k; ++i)
         {
            var size = sizes[i];
            var mat = new List<List<double>>();
            for (var j = 0; j < size; ++j)
            {
               var row = new List<double>();
               for (var idx = 0; idx < countOfInputNeurons; ++idx)
               {
                  var bytes = new byte[8];
                  double res;
                  do
                  {
                     provider.GetBytes(bytes);
                     res = BitConverter.ToDouble(bytes, 0);
                  } while (res < min || res > max || Math.Abs(res) < max / 10 || double.IsNaN(res));

                  row.Add(res);

               }

               mat.Add(row);
            }

            layers.Add(mat);
            countOfInputNeurons = size;
         }

         return layers;
      }
   }
}