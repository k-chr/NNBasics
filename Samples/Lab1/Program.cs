using System;
using System.Collections.Generic;
using NNBasicsUtilities.Core;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;

namespace Lab1
{
   class Program
   {
      private static void Main()
      {
         //Task1
         Console.BackgroundColor = ConsoleColor.DarkBlue;
         Console.ForegroundColor = ConsoleColor.DarkGreen;
         Console.Beep();
         Console.WriteLine("---------------------------------------------------");
         Console.WriteLine("-----------------------Task 1----------------------");
         Console.WriteLine("---------------------------------------------------");


         var oN = new Matrix((1, 1).ToTuple()) { [0, 0] =  0.6 };
         var iN = new Matrix(new Tuple<int, int>(1, 1)) {[0, 0] = 34.0};

         Console.BackgroundColor = ConsoleColor.DarkGray;
         Console.ForegroundColor = ConsoleColor.White;
         Console.WriteLine(NeuralEngine.Proceed( iN ,  oN.Transpose()).Data);
         Console.ReadKey();

         //Task2
         Console.BackgroundColor = ConsoleColor.DarkBlue;
         Console.ForegroundColor = ConsoleColor.DarkGreen;
         Console.Beep();
         Console.WriteLine("---------------------------------------------------");
         Console.WriteLine("-----------------------Task 2----------------------");
         Console.WriteLine("---------------------------------------------------");

         var input = new Matrix((1, 3).ToTuple()) {[0, 0] = 3, [0, 1] = 4, [0, 2] = 5};

         oN = new Matrix((3, 1).ToTuple())  {[0, 0] = 0.3, [1, 0] = 0.6, [1, 0] = 0.7 };

         Console.BackgroundColor = ConsoleColor.DarkGray;
         Console.ForegroundColor = ConsoleColor.White;
         Console.WriteLine(NeuralEngine.Proceed(input,  oN).Data);
         Console.ReadKey();

         //Task3
         Console.BackgroundColor = ConsoleColor.DarkBlue;
         Console.ForegroundColor = ConsoleColor.DarkGreen;
         Console.Beep();
         Console.WriteLine("---------------------------------------------------");
         Console.WriteLine("-----------------------Task 3----------------------");
         Console.WriteLine("---------------------------------------------------");

         input[0, 0] = 8.5;
         input[0, 1] = 0.65;
         input[0, 2] = 1.2;

         List<double> l1 = new List<double>(new[] { 0.1, 0.1, -0.3 });
         List<double> l2 = new List<double>(new[] { 0.1, 0.2, 0.0 });
         List<double> l3 = new List<double>(new[] { 0.0, 1.3, 0.1 });

         Console.BackgroundColor = ConsoleColor.DarkGray;
         Console.ForegroundColor = ConsoleColor.White;
         Console.WriteLine(NeuralEngine.Proceed(input, new List<List<double>>(new[] { l1, l2, l3 }).ToMatrix().Transpose()).Data);
         Console.ReadKey();

         //Task4
         Console.BackgroundColor = ConsoleColor.DarkBlue;
         Console.ForegroundColor = ConsoleColor.DarkGreen;
         Console.Beep();
         Console.WriteLine("---------------------------------------------------");
         Console.WriteLine("-----------------------Task 4----------------------");
         Console.WriteLine("---------------------------------------------------");

         l1 = new List<double>(new[] { 0.1, 0.2, -0.1 });
         l2 = new List<double>(new[] { -0.1, 0.1, 0.9 });
         l3 = new List<double>(new[] { 0.1, 0.4, 0.1 });

         List<double> l4 = new List<double>(new[] { 0.3, 1.1, -0.3 });
         List<double> l5 = new List<double>(new[] { 0.1, 0.2, 0.0 });
         List<double> l6 = new List<double>(new[] { 0.0, 1.3, 0.1 });

         var output = new List<List<double>>(new[] {l4, l5, l6}).ToMatrix();
         var hidden = new List<List<double>>(new[] {l1, l2, l3}).ToMatrix();
         Console.BackgroundColor = ConsoleColor.DarkGray;
         Console.ForegroundColor = ConsoleColor.White;
         var inputData = NeuralEngine.Proceed(input, hidden.Transpose()).Data;
         Console.WriteLine(inputData);
         Console.WriteLine(NeuralEngine.Proceed(inputData, output.Transpose()).Data);
         Console.ReadKey();

         //Task5
         Console.BackgroundColor = ConsoleColor.DarkBlue;
         Console.ForegroundColor = ConsoleColor.DarkGreen;
         Console.Beep();
         Console.WriteLine("---------------------------------------------------");
         Console.WriteLine("-----------------------Task 5----------------------");
         Console.WriteLine("---------------------------------------------------");

         Console.BackgroundColor = ConsoleColor.DarkGray;
         Console.ForegroundColor = ConsoleColor.White;
         var h2 = NeuralEngine.GenerateRandomLayer(2, 9, -1.3, 1.3);
         var h1 = NeuralEngine.GenerateRandomLayer(3, 2, -0.2, 0.3);
         var h3 = NeuralEngine.GenerateRandomLayer(9, 5, -0.2, 0.3);
         var h4 = NeuralEngine.GenerateRandomLayer(5, 6, -0.32, 0.23);
         var h5 = NeuralEngine.GenerateRandomLayer(6, 7, -0.32, 0.23);
         var o1 = NeuralEngine.GenerateRandomLayer(7, 3, -0.32, 0.23);
         var fP1 = NeuralEngine.Proceed(input, h1.Transpose()).Data;
         Console.WriteLine(fP1);
         var fP2 = NeuralEngine.Proceed(fP1, h2.Transpose()).Data;
	      Console.WriteLine(fP2);
         var fP3 = NeuralEngine.Proceed(fP2, h3.Transpose()).Data;
         Console.WriteLine(fP3);
         var fP4 = NeuralEngine.Proceed(fP3, h4.Transpose()).Data;
         Console.WriteLine(fP4);
         var fP5 = NeuralEngine.Proceed(fP4, h5.Transpose()).Data;
         Console.WriteLine(fP5);
         var outAnswer = NeuralEngine.Proceed(fP5, o1.Transpose()).Data;
         Console.WriteLine(outAnswer);
         Console.ReadKey();
      }
   }
}
