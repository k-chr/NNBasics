using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using NNBasicsUtilities.Core.Utilities.UtilityTypes;
using NNBasicsUtilities.Extensions;

namespace NNBasicsUtilities.Core.Utilities
{
	public static class UtilitiesFunctions
	{
		public static (Matrix, Matrix) ParseSeriesAndExpectedAnswersFromWeb(string webUrl)
		{
			var predefinedDictionary = new Dictionary<int, List<double>>()
			{
				{1, new List<double> {1, 0, 0, 0}},
				{2, new List<double> {0, 1, 0, 0}},
				{3, new List<double> {0, 0, 1, 0}},
				{4, new List<double> {0, 0, 0, 1}},
			};

			var client = new WebClient();
			client.DownloadFile(webUrl, "data.txt");
			var lines = File.ReadAllLines("data.txt");

			if (!lines.Any()) throw new ArgumentException("Cannot download data");

			var mat = new List<List<double>>();
			var expected = new List<List<double>>();

			foreach (var line in lines)
			{
				var values = line.Split(' ').Select(s => s.Replace('.', ',')).ToList();
				var rowValues = values.Take(3).ToList();
				var key = values.Skip(3).Take(1).First();
				var row = rowValues.Select(double.Parse).ToList();
				mat.Add(row);
				expected.Add(predefinedDictionary[int.Parse(key)]);
			}

			return (mat.ToMatrix(), expected.ToMatrix());
		}

		private static Matrix LoadMnistImagesFromFileName(string name)
		{
			var mat = new List<List<double>>();
			var runningPath = AppDomain.CurrentDomain.BaseDirectory;
			using var fs =
				File.OpenRead($"{Path.GetFullPath(Path.Combine(runningPath, @"..\..\..\..\..\"))}Resources\\{name}");

			Console.WriteLine($"Parsing {fs.Name}");
			var bytes = new byte[4];
			fs.Read(bytes, 0, 4);
			var res = BitConverter.ToUInt32(bytes.Reverse().ToArray(), 0);
			Console.WriteLine($"Successfully obtained Magic Number: {res}");
			fs.Read(bytes, 0, 4);
			res = BitConverter.ToUInt32(bytes.Reverse().ToArray(), 0);
			Console.WriteLine($"Number of images: {res}");
			fs.Read(bytes, 0, 4);
			var rows = BitConverter.ToUInt32(bytes.Reverse().ToArray(), 0);
			fs.Read(bytes, 0, 4);
			var cols = BitConverter.ToUInt32(bytes.Reverse().ToArray(), 0);

			for (var idx = 0; idx < res; ++idx)
			{
				var row = new List<double>();

				for (var k = 0; k < rows * cols; ++k)
				{
					row.Add(fs.ReadByte() / 255.0);
				}

				mat.Add(row);
			}

			return mat.ToMatrix();
		}

		private static Matrix LoadMnistLabelsFromFileName(string name)
		{
			var predefinedDictionary = new Dictionary<int, List<double>>()
			{
				{0, new List<double> {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
				{1, new List<double> {0, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
				{2, new List<double> {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}},
				{3, new List<double> {0, 0, 0, 1, 0, 0, 0, 0, 0, 0}},
				{4, new List<double> {0, 0, 0, 0, 1, 0, 0, 0, 0, 0}},
				{5, new List<double> {0, 0, 0, 0, 0, 1, 0, 0, 0, 0}},
				{6, new List<double> {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}},
				{7, new List<double> {0, 0, 0, 0, 0, 0, 0, 1, 0, 0}},
				{8, new List<double> {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}},
				{9, new List<double> {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}},
			};

			var labels = new List<List<double>>();

			var runningPath = AppDomain.CurrentDomain.BaseDirectory;
			using var fs =
				File.OpenRead($"{Path.GetFullPath(Path.Combine(runningPath, @"..\..\..\..\..\"))}Resources\\{name}");

			Console.WriteLine($"Parsing {fs.Name}");
			var bytes = new byte[4];
			fs.Read(bytes, 0, 4);
			var res = BitConverter.ToUInt32(bytes.Reverse().ToArray(), 0);
			Console.WriteLine($"Successfully obtained Magic Number: {res}");
			fs.Read(bytes, 0, 4);
			res = BitConverter.ToUInt32(bytes.Reverse().ToArray(), 0);
			Console.WriteLine($"Number of labels: {res}");

			for (var idx = 0; idx < res; ++idx)
			{
				var val = fs.ReadByte();
				labels.Add(predefinedDictionary[val]);
			}

			return labels.ToMatrix();
		}

		public static (Matrix, Matrix, Matrix, Matrix) LoadMnistDataBase()
		{
			return (LoadMnistImagesFromFileName("train-images-idx3-ubyte"),
				LoadMnistLabelsFromFileName("train-labels-idx1-ubyte"),
				LoadMnistImagesFromFileName("t10k-images-idx3-ubyte"),
				LoadMnistLabelsFromFileName("t10k-labels-idx1-ubyte"));
		}
	}
}