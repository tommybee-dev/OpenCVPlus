#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

//https://answers.opencv.org/question/27695/puttext-with-black-background/
void setLabel(cv::Mat& im, const std::string label, const cv::Point& pt)
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, cv::Point(0 + pt.x, baseline + pt.y), cv::Point(text.width + pt.x, -text.height + pt.y), CV_RGB(0,0,0), CV_FILLED);
    cv::putText(im, label, pt, fontface, scale, CV_RGB(255,255,255), thickness, 8);
}


//https://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/
char showTwoImagesEach(string title, const Mat& img1, const Mat& img2, const string &label)
{
    char k=0;
    namedWindow(title);
    Mat img_result;
    hconcat(img1, img2, img_result); //두 이미지의 높이(row)가 같아야함.
    //putText(img_result, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255));
    setLabel(img_result, label, Point(0, 15));

	namedWindow(title, WINDOW_AUTOSIZE);
	imshow(title, img_result);

    k=waitKey(10);
    return k;
}

char showTwoImagesComb(string title, const Mat& img1, const Mat& img2, const string &label)
{
    char k=0;
    namedWindow(title, WINDOW_AUTOSIZE);

    Mat matDst(Size(img1.cols*2,img1.rows),img1.type(),Scalar::all(0));
    Mat matRoi = matDst(Rect(0,0,img1.cols,img1.rows));
    img1.copyTo(matRoi);
    matRoi = matDst(Rect(img1.cols,0,img1.cols,img1.rows));
    img2.copyTo(matRoi);
    //putText(matDst, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255));
    setLabel(matDst, label, Point(0, 15));
    imshow(title,matDst);

    k=waitKey(10);
    return k;
}

cv::Mat createOne(std::vector<cv::Mat> & images, int cols, int min_gap_size)
{
    // let's first find out the maximum dimensions
    int max_width = 0;
    int max_height = 0;

    for ( unsigned i = 0; i < images.size(); i++) {
        // check if type is correct
        // you could actually remove that check and convert the image
        // in question to a specific type
        if ( i > 0 && images[i].type() != images[i-1].type() ) {
            std::cerr << "WARNING:createOne failed, different types of images";
            return cv::Mat();
        }
        max_height = std::max(max_height, images[i].rows);
        max_width = std::max(max_width, images[i].cols);
    }
    // number of images in y direction
    int rows = std::ceil(images.size() / cols);

    // create our result-matrix
    cv::Mat result = cv::Mat::zeros(rows*max_height + (rows-1)*min_gap_size,
                                    cols*max_width + (cols-1)*min_gap_size, images[0].type());
    size_t i = 0;
    int current_height = 0;
    int current_width = 0;
    for ( int y = 0; y < rows; y++ ) {
        for ( int x = 0; x < cols; x++ ) {
            if ( i >= images.size() ) // shouldn't happen, but let's be safe
                return result;
            // get the ROI in our result-image
            cv::Mat to(result,
                       cv::Range(current_height, current_height + images[i].rows),
                       cv::Range(current_width, current_width + images[i].cols));
            // copy the current image to the ROI
            images[i++].copyTo(to);
            current_width += max_width + min_gap_size;
        }
        // next line - reset width and update height
        current_width = 0;
        current_height += max_height + min_gap_size;
    }
    return result;
}

