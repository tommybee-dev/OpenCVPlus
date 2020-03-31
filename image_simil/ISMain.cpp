#include <iostream>

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

#include "core.h"
#include "iqi.h"
#include "mse.h"
#include "ssim.h"
#include "msssim.h"
#include "psnr.h"

cv::String keys =
        "{@image1    |<none>| input image 1 name}"
        "{@image2    |<none>| input image 2 name}"
        "{algorithm  |all| mse - Mean Square Error \r\n"
        "               psnr - Peak Signal to Noise Ratio \r\n"
        "               ssim - Structural Similarity Index Metric \r\n"
        "               msssim - Multi-scale Structural Similarity Index Metric \r\n"
        "               iqi - Image Quality Index \r\n"
        "               all - All of the above metrics  | algorithm to use}"
        "{colorspace  |0| 0 - GRAYSCALE \r\n"
        "                1 - RGB \r\n"
        "                2 - YCbCr  | colorspace}"
        "{help       |      | show help message}";      // optional, show help optional


static void printCvScalar(CvScalar value, const char *comment)
{
    cout<<comment<<" : "<<value.val[0]<<" , "<<value.val[1]<<" , "<<value.val[2]<<"\n";
}


void showTwoImagesComb(string title, const Mat& img1, const Mat& img2, const string &label)
{
    namedWindow(title, WINDOW_AUTOSIZE);

    Mat matDst(Size(img1.cols*2,img1.rows),img1.type(),Scalar::all(0));
    Mat matRoi = matDst(Rect(0,0,img1.cols,img1.rows));
    img1.copyTo(matRoi);
    matRoi = matDst(Rect(img1.cols,0,img1.cols,img1.rows));
    img2.copyTo(matRoi);
    putText(matDst, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 212, 255));
    imshow(title,matDst);

    waitKey(0);
}

int showImgSimil(const string file1,const string file2, CvScalar result, const char *algorithm, const char *comment)
{
    Mat mtx1;
	Mat mtx2;

    mtx1 = imread(file1, IMREAD_UNCHANGED);
    if (mtx1.empty()) return 1;
	mtx2 = imread(file2, IMREAD_UNCHANGED);
	if (mtx2.empty()) return 1;


    std::ostringstream sOsStream;

    printCvScalar(result, comment);


    sOsStream <<algorithm<<" : "<<result.val[0]<<" , "<<result.val[1]<<" , "<<result.val[2];

    showTwoImagesComb(comment, mtx1, mtx2, sOsStream.str());

    sOsStream.clear();
    sOsStream.seekp(0);

    mtx1.release();
    mtx2.release();


    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("usage: --image1 lena.bmp --image2 lena_blur.bmp --algorithm mse ");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    bool hasAlgorithm = parser.has("algorithm");
    try
    {

        if (!hasAlgorithm) {
            parser.printMessage();
            return 0;
        }
    }
    catch(std::exception& e)
    {
        std::cerr << "exception caught: " << e.what() << '\n';
    }

    bool hasColorspace = parser.has("colorspace");
    if (!hasColorspace) {
        parser.printMessage();
        return 0;
    }

    cv::String image1 = parser.get<cv::String>(0);
    cv::String image2 = parser.get<cv::String>(1);

    cv::String algorithm = parser.get<cv::String>("algorithm");
    int colorspace = parser.get<int>("colorspace");

    if (!parser.check()) {
        std::cerr << "printErrors: " << '\n';
        parser.printErrors();
        return -1;
    }

    std::cout << "image1:(" << image1 << ")" << '\n';
    std::cout << "image2:(" << image2 << ")" << '\n';
    std::cout << "algorithm:(" << algorithm << ")" << '\n';
    std::cout << "colorspace:(" << colorspace << ")" << '\n' << std::endl;
    Colorspace cspace;
    switch(colorspace)
    {

    case 0:
        cspace = GRAYSCALE;
        break;
    case 1:
        cspace = RGB;
        break;
    case 2:
        cspace = YCbCr;
        break;
    }

    // Reading Images
    IplImage *src1;
    IplImage *src2;

    src1 = cvLoadImage(image1.c_str());
    src2 = cvLoadImage(image2.c_str());

    if(algorithm.compare("iqi") == 0)
    {
        calcQualityIndex qi;
        CvScalar res;


        res = qi.compare(src1, src2, cspace);

        showImgSimil(image1, image2, res, algorithm.c_str(), "[QualityIndex]");

    } else if(algorithm.compare("mse") == 0)
    {
        calcMSE mse;
        CvScalar res;

        res = mse.compare(src1, src2, cspace);

        showImgSimil(image1, image2, res,algorithm.c_str(), "[MSE]");

    } else if(algorithm.compare("ssim") == 0)
    {
        calcSSIM ssim;
        CvScalar res;

        res = ssim.compare(src1, src2, cspace);

        showImgSimil(image1, image2, res,algorithm.c_str(), "[SSIM]");

    } else if(algorithm.compare("mssim") == 0)
    {
        calcMSSSIM mssim;
        CvScalar res;

        res = mssim.compare(src1, src2, cspace);

        showImgSimil(image1, image2, res,algorithm.c_str(), "[MSSSIM]");

    } else if(algorithm.compare("psnr") == 0)
    {
        calcPSNR psnr;
        CvScalar res;

        res = psnr.compare(src1, src2, cspace);

        showImgSimil(image1, image2, res,algorithm.c_str(), "[PSNR]");

    } else if(algorithm.compare("all") == 0)
    {
        calcQualityIndex qi;
        calcMSE mse;
        calcSSIM ssim;
        calcMSSSIM mssim;
        calcPSNR psnr;
        CvScalar res1, res2, res3, res4, res5;

        res1 = qi.compare(src1, src2, cspace);
        res2 = mse.compare(src1, src2, cspace);
        res3 = ssim.compare(src1, src2, cspace);
        res4 = mssim.compare(src1, src2, cspace);
        res5 = psnr.compare(src1, src2, cspace);

        showImgSimil(image1, image2, res1,"iqi", "[QualityIndex]");
        showImgSimil(image1, image2, res2,"mse", "[MSE]");
        showImgSimil(image1, image2, res3,"ssim", "[SSIM]");
        showImgSimil(image1, image2, res4,"mssim", "[MSSSIM]");
        showImgSimil(image1, image2, res5,"psnr", "[PSNR]");

    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    cvReleaseImage(&src1);
    cvReleaseImage(&src2);


    return EXIT_SUCCESS;

}
