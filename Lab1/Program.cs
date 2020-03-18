using System;
using System.Collections.Generic;

namespace NNBasics.Lab1
{
   class Program
   {
      private static void Main(string[] args)
      {
         //Task1
         Console.BackgroundColor = ConsoleColor.DarkBlue;
         Console.ForegroundColor = ConsoleColor.DarkGreen;
         Console.Beep();
         Console.WriteLine("---------------------------------------------------");
         Console.WriteLine("-----------------------Task 1----------------------");
         Console.WriteLine("---------------------------------------------------");

         InputNeuron iN = new InputNeuron {Value = 34.0};
         OutputNeuron oN = new OutputNeuron {Weights = new List<double>(new[] {0.6})};

         Console.BackgroundColor = ConsoleColor.DarkGray;
         Console.ForegroundColor = ConsoleColor.White;
         Console.WriteLine(NeuralEngine.Proceed(new List<InputNeuron>(new []{iN}), new List<OutputNeuron>(new []{oN})).ToString());
         Console.ReadKey();

         //Task2
         Console.BackgroundColor = ConsoleColor.DarkBlue;
         Console.ForegroundColor = ConsoleColor.DarkGreen;
         Console.Beep();
         Console.WriteLine("---------------------------------------------------");
         Console.WriteLine("-----------------------Task 2----------------------");
         Console.WriteLine("---------------------------------------------------");

         InputNeuron iN1 = new InputNeuron(){Value = 3}, iN2 = new InputNeuron(){Value = 4}, iN3= new InputNeuron(){Value = 5};
         List<InputNeuron> input = new List<InputNeuron>(new []
         {
            iN1, iN2, iN3
         });

         oN.Weights = new List<double>(new[]{0.3, 0.6, 0.7});

         Console.BackgroundColor = ConsoleColor.DarkGray;
         Console.ForegroundColor = ConsoleColor.White;
         Console.WriteLine(NeuralEngine.Proceed(input, new List<OutputNeuron>(new []{oN})).ToString());
         Console.ReadKey();

         //Task3
         Console.BackgroundColor = ConsoleColor.DarkBlue;
         Console.ForegroundColor = ConsoleColor.DarkGreen;
         Console.Beep();
         Console.WriteLine("---------------------------------------------------");
         Console.WriteLine("-----------------------Task 3----------------------");
         Console.WriteLine("---------------------------------------------------");

         iN1.Value = 8.5;
         iN2.Value = 0.65;
         iN3.Value = 1.2;

         input = new List<InputNeuron>(new[]
         {
            iN1, iN2, iN3
         });

         List<double> l1 = new List<double>(new[] {0.1,0.1,-0.3 });
         List<double> l2 = new List<double>(new[] {0.1, 0.2, 0.0 });
         List<double> l3 = new List<double>(new[] {0.0, 1.3, 0.1 });

         Console.BackgroundColor = ConsoleColor.DarkGray;
         Console.ForegroundColor = ConsoleColor.White;
         Console.WriteLine(NeuralEngine.Proceed(input, new List<List<double>>(new[] {l1,l2,l3 })).ToString());
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

         Console.BackgroundColor = ConsoleColor.DarkGray;
         Console.ForegroundColor = ConsoleColor.White;
         Console.WriteLine(NeuralEngine.Proceed(input, Tuple.Create(
            new List<List<double>>(new[] { l1, l2, l3 }),
            new List<List<double>>(new[] { l4, l5, l6 }))
         ).ToString());
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
         Console.WriteLine(NeuralEngine.Proceed(input, null, true, -1.3, 1.3, new List<int>(new int[]{2,9,5,6,7,3}) ).ToString());
         Console.ReadKey();
      }
   }
}
