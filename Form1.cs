using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using System.IO;
using System.Diagnostics;

namespace MarkRecognition
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            int AllMarkCount = 6;

            //原始影像=照片, 目標影像=Mark輸入影像
            //原始影像長寬
            int orgImgWidth = 640;
            int orgImgHeight = 480;
            //滑動窗掃描間隔
            int slidingWindowStepX = 3;
            int slidingWindowStepY = 3;
            //滑動窗長寬(目標影像長寬)
            int tarImgWidth = 50;
            int tarImgHeight = 50;

            Size MarkSizeByCell = new Size(5, 5); //以子區塊為單位的mark長寬
            Size CellSize = new Size(tarImgWidth / MarkSizeByCell.Width, tarImgHeight / MarkSizeByCell.Height);

            // ------------------


            //KNN分類器訓練
            // Mark類別 載入標籤影像
            List<string> sampleImgPath = Directory.GetFiles(@"Image\").ToList();
            List<Mark> sampleMark = new List<Mark>();


            // 擷取特徵
            // KNN分類器 監督學習
            for (int i = 0; i < sampleImgPath.Count; i++)
            {
                FileInfo curFile = new FileInfo(sampleImgPath[i]);
                string exName = curFile.Extension;
                if (exName == ".png")
                {
                    Image<Gray, byte> imageTmp = new Image<Gray, byte>(sampleImgPath[i]);

                    //threshold ----------------------
                    Emgu.CV.CvInvoke.cvThreshold(imageTmp.Ptr, imageTmp.Ptr, -1, 255d, Emgu.CV.CvEnum.THRESH.CV_THRESH_BINARY | Emgu.CV.CvEnum.THRESH.CV_THRESH_OTSU);
                    
                    Mark markTmp = new Mark(imageTmp, MarkSizeByCell, CellSize);
                    sampleMark.Add(markTmp);
                    sampleMark[i].GetFeatures();

                    int length = AllMarkCount;
                    for (int j = 0; j < length; j++)
                    {
                        if (curFile.Name.Contains("training0" + (j + 1).ToString()))
                        {
                            sampleMark[i].MarkIndex = j + 1;
                            break;
                        }
                    }
                    new ImageViewer(sampleMark[i].iptImg, "[" + sampleMark[i].MarkIndex.ToString() + "]FrameMean" + sampleMark[i].FrameMean.ToString() + ",StdDivSum" + sampleMark[i].StdDivSum.ToString()).Show();

                }
            }

            // KNN分類器


            // ----------------

            //攝影機載入影像
            List<string> orgImgPath = Directory.GetFiles(@"D:\Phisten\GoogleCloud\圖訊識別\圖訊testdata\").ToList();
            int imgCoung = orgImgPath.Count;
            //imgCoung = imgCoung > 5 ? 5 : imgCoung;
            //imgCoung = 1;
            for (int imgIdx = 0; imgIdx < imgCoung; imgIdx++)
            {

                string imgPath = orgImgPath[imgIdx];

                Image<Rgb, byte> orgImg = new Image<Rgb, byte>(imgPath);

                //正規化
                Image<Gray, byte> norImg;
                norImg = orgImg.Convert<Gray, byte>();
                //norImg = norImg.ConvertScale<byte>(3d, -100d);
                // 影像長寬
                // 亮度

                //SlidingWindow擷取輸入影像
                List<IImage> iptImgList = new List<IImage>();
                List<Rectangle> iptImgRectList = new List<Rectangle>();
                Rectangle tmpRect = new Rectangle(0, 0, 50, 50);
                norImg.ROI = tmpRect;
                int StepWidthLimit = orgImgWidth - tarImgWidth - (orgImgWidth - tarImgWidth) % slidingWindowStepX;
                int StepHeightLimit = orgImgHeight - tarImgHeight - (orgImgHeight - tarImgHeight) % slidingWindowStepY;
                for (int i = 0; i < StepWidthLimit; i += slidingWindowStepX)
                {
                    tmpRect.Y = 0;
                    for (int j = 0; j < StepHeightLimit; j += slidingWindowStepY)
                    {
                        Image<Gray, byte> curMarkImg = norImg.CopyBlank();
                        //threshold ----------------------
                        int greyThreshValue = (int)Emgu.CV.CvInvoke.cvThreshold(norImg.Ptr, curMarkImg.Ptr, -1, 255d, Emgu.CV.CvEnum.THRESH.CV_THRESH_BINARY | Emgu.CV.CvEnum.THRESH.CV_THRESH_OTSU);
                        iptImgList.Add(curMarkImg);
                        //iptImgList.Add(norImg.Copy());
                        iptImgRectList.Add(norImg.ROI);
                        tmpRect.Offset(0, slidingWindowStepY);
                        norImg.ROI = tmpRect;
                    }
                    tmpRect.Offset(slidingWindowStepX, 0);
                    norImg.ROI = tmpRect;
                }

                List<Mark> markList = new List<Mark>();
                for (int imgIndex = 0; imgIndex < iptImgList.Count; imgIndex++)
                {
                    //擷取特徵
                    Mark curMark = new Mark(iptImgList[imgIndex] as Image<Gray, byte>, MarkSizeByCell, CellSize);
                    
                    curMark.GetFeatures();

                    //特徵匹配
                    double KNNdistanceThreshold = 256d;
                    // 過濾外框平均值過高
                    // 過濾標準差總和過高
                    if (curMark.FrameFilter(96) && curMark.StdDivSumFilter(1024d))
                    {
                        // 最近鄰分類
                        SortedList<double, int> distanceSList = new SortedList<double, int>();
                        for (int i = 0,length = sampleMark.Count; i < length; i++)
                        {
                            double curDis = sampleMark[i].Distance(curMark);
                            if (curDis < KNNdistanceThreshold)
                            {
                                distanceSList.Add(curDis, sampleMark[i].MarkIndex);
                            }
                        }


                        //合格的Mark
                        //int markIndex = Phisten.Classifier.KNN(1, distanceSList);.
                        if (distanceSList.Count > 0)// && distanceSList.Keys[0] < KNNdistanceThreshold)
                        {
                            curMark.MarkIndex = distanceSList.Values[0];
                            curMark.MarkIndexDistance = distanceSList.Keys[0];

                            Rectangle markRect = iptImgRectList[imgIndex];
                            curMark.MarkRectangle = markRect;

                            bool IsNewMark = true;
                            //重疊過濾
                            for (int i = 0, length = markList.Count; i < length; i++)
                            {
                                if (markRect.IntersectsWith(markList[i].MarkRectangle)) //若區域重疊則不新增mark
                                {
                                    if (distanceSList.Keys[0] < markList[i].MarkIndexDistance) //且若curMark距離較近
                                    {
                                        //則替換Mark
                                        markList[i] = curMark;
                                    }
                                    IsNewMark = false;
                                    break;
                                }
                            }
                            if (IsNewMark)
                            {
                                //否則新增Mark
                                markList.Add(curMark);
                            }
                        }

                    }

                }

                Image<Rgb, byte> outputImg = orgImg.Convert<Gray, byte>().Convert<Rgb, byte>();
                outputImg = outputImg.SmoothGaussian(3);
                MCvFont pen1 = new MCvFont(Emgu.CV.CvEnum.FONT.CV_FONT_HERSHEY_SIMPLEX, 0.5d, 0.5d);
                //繪製分類結果
                for (int i = 0; i < markList.Count; i++)
                {
                    string fileName1 = i.ToString() + ".jpg";
                    //markList[i].iptImg.Save(@"opt\" + fileName1);
                    outputImg.Draw(markList[i].MarkRectangle, new Rgb(255, 0, 0), 1);
                    outputImg.Draw("[" + markList[i].MarkIndex + "]" + Math.Round(markList[i].MarkIndexDistance), ref pen1, markList[i].MarkRectangle.Location, new Rgb(50, 0, 200));
                }

                //輸出影像
                this.Width = 0;
                this.Height = 0;
                ImageViewer imgViewer = new Emgu.CV.UI.ImageViewer(outputImg);
                imgViewer.Show();


            }
        }

    }
}
