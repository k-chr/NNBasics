using System;
using System.Text;

namespace NNBasicsUtilities.Core.Utilities.UtilityTypes
{
	public struct FlatMatrix
	{
		private readonly double[] _data;
		public readonly int Rows;
		public readonly int Cols;

		public override string ToString()
		{
			var builder = new StringBuilder();
			for (var i = 0; i < _data.Length; ++i)
			{
				if (i % (Cols) == 0)
				{
					builder.Append("| ");
				}
				builder.Append(_data[i]).Append(' ');
				if ((i + 1) % (Cols) == 0)
				{
					builder.Append("|\n");
				}
			}

			return builder.ToString();
		}



	}
}
