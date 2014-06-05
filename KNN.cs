using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Phisten
{
    class Classifier
    {
        public static int KNN(int K, SortedList<double,int> distanceList)
        {

            int length = distanceList.Count;
            for (int i = 0; i < K; i++)
            {
                double curMinDistance = double.MaxValue;
                int curMinDistanceIndex = 0;
                for (int j = 0; j < length - K; j++)
                {
                    if (distanceList[j] < curMinDistance)
                    {
                        curMinDistance = distanceList[j];
                        curMinDistanceIndex = j;
                    }
                }
                distanceList.RemoveAt(curMinDistanceIndex);
            }


            return -1;
        }
    }
}
