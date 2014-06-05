using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Drawing;


namespace MarkRecognition
{
    class Mark
    {
        public Image<Gray, byte> iptImg;

        //特徵
        public double[,] Mean;
        public double[,] StdDiv;
        public double FrameMean = 0d;
        public double StdDivSum = 0d;
        public int MarkIndex;
        /// <summary>與其他mark的距離  由外部設定</summary>
        public double MarkIndexDistance = 0d;
        public Rectangle MarkRectangle;

        Size MarkSizeByCell;
        Size CellSize;
        int imgWidth;
        int imgHeight;

        

        public Mark(string sampleImagePath, Size MarkSizeByCell, Size CellSize)
            : this(new Image<Gray, byte>(sampleImagePath),MarkSizeByCell, CellSize)
        {

        }

        public Mark(Image<Gray, byte> iptImg, Size MarkSizeByCell, Size CellSize)
            : this(MarkSizeByCell, CellSize)
        {
            this.iptImg = iptImg;
            Normalize();
            imgWidth = iptImg.Width;
            imgHeight = iptImg.Height;
        }

        public Mark(Size MarkSizeByCell, Size CellSize)
        {
            this.MarkSizeByCell = MarkSizeByCell;
            this.CellSize = CellSize;
            Mean = new double[MarkSizeByCell.Width, MarkSizeByCell.Height];
            StdDiv = new double[MarkSizeByCell.Width, MarkSizeByCell.Height];
        }

        /// <summary>影像處理類正規化</summary>
        private void Normalize()
        {
            this.iptImg = this.iptImg.SmoothGaussian(3);
        }

        public void GetFeatures()
        {
            //Mark類別載入影像
            Image<Gray, byte> srcImg = iptImg;
            int tarImgWidth = srcImg.Width;
            int tarImgHeight = srcImg.Height;

            //Mark類別正規化(影像轉正)




            //擷取特徵
            for (int i = 0; i < MarkSizeByCell.Width; i++)
            {
                int startX = i * CellSize.Width;
                Rectangle tmpROI = new Rectangle(startX + 2, 0, CellSize.Width - 2, CellSize.Height - 2);

                for (int j = 0; j < MarkSizeByCell.Height; j++)
                {
                    int startY = j * CellSize.Height;
                    tmpROI.Y = startY + 2;
                    srcImg.ROI = tmpROI;

                    //計算所有子區塊平均值與標準差
                    Gray curMean;
                    MCvScalar curStdDiv;
                    srcImg.AvgSdv(out curMean, out curStdDiv);
                    Mean[j, i] = curMean.MCvScalar.v0;
                    StdDiv[j, i] = curStdDiv.v0;

                    //srcImg.ROI.Offset(0, tarImgHeight);
                }


            }

            iptImg.ROI = Rectangle.Empty;

            FrameMean = 0d;
            // 加總外框平均值
            for (int i = 0; i < MarkSizeByCell.Width; i++)
            {
                if (Mean[0, i] > 255 || Mean[MarkSizeByCell.Height - 1, i] > 255)
                {
                    throw new ApplicationException("平均值異常");
                }
                FrameMean += Mean[0, i];
                FrameMean += Mean[MarkSizeByCell.Height - 1, i];
            }
            for (int i = 1; i < MarkSizeByCell.Height - 1; i++)
            {
                if (Mean[i, 0] > 255 || Mean[i, MarkSizeByCell.Width - 1] > 255)
                {
                    throw new ApplicationException("平均值異常");
                }
                FrameMean += Mean[i, 0];
                FrameMean += Mean[i, MarkSizeByCell.Width - 1];
            }
            double FrameCellCount = ((MarkSizeByCell.Width + MarkSizeByCell.Height) * 2d - 4d);
            FrameMean = FrameMean / FrameCellCount;


            StdDivSum = 0d;
            // 計算25格的標準差總和
            for (int i = 0; i < MarkSizeByCell.Height; i++)
            {
                for (int j = 0; j < MarkSizeByCell.Width; j++)
                {
                    StdDivSum += StdDiv[i, j];
                }
            }
        }

        public bool FrameFilter(int threshold)
        {
            if (this.FrameMean > 255d)
            {
                throw new ApplicationException("外框平均值異常");
            }
            if (this.FrameMean > threshold)
                return false;
            return true;
        }
        public bool StdDivSumFilter(double threshold)
        {
            if (this.StdDivSum > threshold)
                return false;
            return true;
        }

        public double Distance(Mark tarMark)
        {
            double retDistance = 0d;

            for (int i = 0; i < MarkSizeByCell.Width; i++)
            {
                for (int j = 0; j < MarkSizeByCell.Height; j++)
                {
                    double disTmp = this.Mean[j, i] - tarMark.Mean[j, i];
                    retDistance += disTmp * disTmp;
                }
            }

            return Math.Sqrt(retDistance);
        }

    }
}
