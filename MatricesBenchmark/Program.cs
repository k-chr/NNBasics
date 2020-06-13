using System;
using BenchmarkDotNet.Running;
namespace MatricesBenchmark
{
   class Program
   {
	  static void Main(string[] args)
	  {
		 Console.WriteLine("Hello World!");
		 BenchmarkRunner.Run<Benchmarking>();
	  }
   }
}
