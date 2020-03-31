//#include <Windows.h>
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\video\background_segm.hpp"
#include "opencv2\video\tracking.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <unistd.h>

using namespace cv;
using namespace std;

static void reSize(Mat& image, int width=0, int height=0, int inter=CV_INTER_AREA)
{
    // initialize the dimensions of the image to be resized and
    // grab the image size
    Size dim;

    if(image.empty()) return;

    int h = image.rows;
    int w = image.cols;
    float r = 0.0f;
    //(h, w) = image.shape[:2]

    // if both the width and height are None, then return the
    // original image
    if( width == 0 && height==0) return;

    // check to see if the width is None
    if( width == 0)
    {
        // calculate the ratio of the height and construct the
        // dimensions
        r = height / float(h);
        Size sz((int)(w * r), height);
        dim = sz;
    }
    // otherwise, the height is None
    else
    {
        // calculate the ratio of the width and construct the
        // dimensions
        r = width / float(w);
        Size sz(width, int(h * r));
        dim = sz;
    }


    // resize the image
    resize(image, image, dim, inter);

    // return the resized image
    return;
}


static void showImage(string name, const Mat image)
{
    imshow(name, image);
    waitKey(1000);
}

static int pyramid(Mat& image, void (*slidingWindowFcn)(Mat& image, int stepSize, Size windowSize), float scale=1.5, int stepSize=32, Size minSize=Size(30, 30), Size windowSize=Size(128,128))
{
    int ret = 0;
    if(image.empty()) return ret;

	while(true)
    {
        // compute the new dimensions of the image and resize it
		int w = int(image.cols / scale);

		reSize(image, w);

		if(slidingWindowFcn) slidingWindowFcn(image, stepSize, windowSize);

		cout<< "cols: " <<image.cols << " rows: " << image.rows<<endl;

        if (image.rows < minSize.height || image.cols < minSize.width)
		{
            ret = 1;
			break;
		}
    }

    return ret;
}

static int pyramid(Mat& image, void (*showImageFcn)(string name, const Mat image), float scale=1.5, Size minSize=Size(30, 30))
{
    int ret = 0;

    if(image.empty()) return ret;

    string name = "pyramid image";

    // yield the original image
	//yield image
	// keep looping over the pyramid
	while(true)
    {
        char tmp[10] = {0,};

        // compute the new dimensions of the image and resize it
		int w = int(image.cols / scale);
		itoa(w,tmp,10);    // 정수형 -> 문자열 변환
        name.append("_").append(tmp);
		reSize(image, w);

		if(showImageFcn) showImageFcn(name, image);

		cout<< "cols: " <<image.cols << " rows: " << image.rows<<endl;

		// if the resized image does not meet the supplied minimum
		// size, then stop constructing the pyramid
		if (image.rows < minSize.height || image.cols < minSize.width)
		{
            ret = 1;
			break;
		}
        // yield the next image in the pyramid
		//yield image
    }

    return ret;
}


int image_pyramid(int argc, const char** argv)
{
    cv::String keys =
        "{@image |<none>           | input image path}"         // input image is the first argument (positional)
        "{@scale  |/path/to/file.xml| scale factor for an image}"        // optional, face cascade is the second argument (positional)
        "{eyes   |                 | eye cascade path}"         // optional, default value ""
        "{nose   |      | nose cascade path}"       // optional, default value ""
        "{mouth  |      | mouth cascade path}"      // optional, default value ""
        "{help   |      | show help message}";      // optional, show help optional


    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("usage: --image <path to image> --scale 1.5");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    cv::String input_image_path = parser.get<cv::String>(0); // read @image (mandatory, error if not present)
    cv::String scale_factor = parser.get<cv::String>(1); // read @face (use default value if not in cmd)

    bool hasEyeCascade = parser.has("eyes");
    bool hasNoseCascade = parser.has("nose");
    bool hasMouthCascade = parser.has("mouth");
    cv::String eye_cascade_path = parser.get<cv::String>("eyes");
    cv::String nose_cascade_path = parser.get<cv::String>("nose");
    cv::String mouth_cascade_path = parser.get<cv::String>("mouth");


    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }

    std::cout<< "image path : "  << input_image_path << std::endl;

    Mat image;
    image = imread(input_image_path, IMREAD_COLOR);

    void (*showImageFcn)(string name, const Mat image);

    showImageFcn = showImage;
    namedWindow("pyramid image", WINDOW_AUTOSIZE);

    float scale = atof(scale_factor.c_str());

    if(scale != 1.5)
    {
        pyramid(image, showImageFcn, scale);
    }
    else
    {
        pyramid(image, showImageFcn);
    }


    //imshow("pyramid image", image);
    waitKey(1000);

    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}


static void sliding_window(Mat& image, int stepSize, Size windowSize)
{
    int y = 0, x = 0;
    Mat clone;

    // THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
    // MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
    // WINDOW
    // since we do not have a classifier, we'll just draw the window

    for(y = 0 ; y < image.rows; y += stepSize)
    {
        for(x = 0; x < image.cols; x += stepSize)
        {
            Rect rect(Point(x,y), Point(x + windowSize.width, y + windowSize.height));

            //https://stackoverflow.com/questions/29120231/how-to-verify-if-rect-is-inside-cvmat-in-opencv
            bool is_inside = (rect & cv::Rect(0, 0, image.cols, image.rows)) == rect;

            if(!is_inside) continue;

            if (windowSize.height != rect.height || windowSize.width != rect.width)
                continue;

            // yield the current window
            //yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            image.copyTo(clone);
            rectangle(clone, Point(x,y), Point(x + windowSize.width, y + windowSize.height), Scalar(0,255,128),2,8,0);

            imshow("Window", clone);
            waitKey(1);
            std::this_thread::sleep_for (std::chrono::milliseconds(25));//0.025
        }
    }
}


int main_silding_window(int argc, const char** argv)
{
    cv::String keys =
        "{@image |<none>           | input image path}"         // input image is the first argument (positional)
        "{help   |      | show help message}";      // optional, show help optional

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("usage: --image <path to image> ");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    cv::String input_image_path = parser.get<cv::String>(0); // read @image (mandatory, error if not present)

    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }

    std::cout<< "image path : "  << input_image_path << std::endl;

    Mat image;
    int winW =128, winH = 128;

    image = imread(input_image_path, IMREAD_COLOR);

    int resized = 0;
    int stepSize=32;
    Size windowSize(winW, winH);

    void (*slidingWindowFcn)(Mat& image, int stepSize, Size windowSize);

    slidingWindowFcn = sliding_window;

    while(!(resized = pyramid(image, slidingWindowFcn)));

    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}

int main(int argc, const char** argv)
{
    main_silding_window(argc, argv);
    return EXIT_SUCCESS;
}

//https://funvision.blogspot.com/2015/12/opencv-tutorial-sliding-window.html
/*
//  Load the image from file
// 파일에서 이미지를 로딩한다.
Mat LoadedImage;
// Just loaded image Lenna.png from project dir to LoadedImage Mat
// LoadedImage Mat에 프로젝트 디렉토리의 Lenna.png를 로딩 합니다.
LoadedImage = imread("Lenna.png", IMREAD_COLOR);
//I would like to visualize Mat step by step to see the result immediately.
//이미지를 바로 볼 수 있도록 단계별로 Mat을 표시하고 싶습니다
// Show what is in the Mat after load
// 로딩 후의 Mat의 내용을 보여 줍니다.
namedWindow("Step 1 image loaded", WINDOW_AUTOSIZE);
imshow("Step 1 image loaded", LoadedImage);
waitKey(1000);
// Same the result from LoadedImage to Step1.JPG
// LoadedImage의 결과를 Step1.JPG 에 저장합니다.
imwrite("Step1.JPG", LoadedImage);

// You can load colored image directly as gray scale
// 컬러 이미지를 바로 그레이 스케일 이미지로 바로 로딩 할 수 있습니다.
LoadedImage = imread("Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);
// Show what is in the Mat after load
// 로딩 후의 Mat의 내용을 보여 줍니다.
namedWindow("Step 2 gray image loaded", WINDOW_AUTOSIZE);
imshow("Step 2 gray image loaded", LoadedImage);
// Show the result for the longer time.
// If you want to see video frames in high rates in the loop jist put here waitKey(20).
//이미지를 일정 시간 동안 좀 더 보여 줍니다.
// 만약 높은 비율의 비디오 프레임으로 보고 싶다면 waitKey(20).
waitKey(1000);

// Same the result from LoadedImage to Step2.JPG
imwrite("Step2.JPG", LoadedImage);
//  Basic resize and rescale
//
// Resize LoadedImage and save the result to same Mat loaded Image.
// You can also resize( loadedImage, Result, ..... )

// Load again source images
LoadedImage = imread("Lenna.png", IMREAD_COLOR);
//You can resize to any size you want Size(width,heigth)
resize(LoadedImage, LoadedImage, Size(100, 100));

// Vizualization
namedWindow("Step 3 image resize", WINDOW_AUTOSIZE);
imshow("Step 3 image resize", LoadedImage);
waitKey(1000);

//Save above image to Step3.jpg
 imwrite("Step3.JPG", LoadedImage);
 LoadedImage = imread("Lenna.png", IMREAD_COLOR);

// Better is resize based on ratio of width and heigth
// Width and heigth are 2 times smaller than original source image
// result will be saved into same mat. If you are confused by this.
// You can try to modify the code and add MAT outputImage and dysplay it.
//!! cols number of collumn of the image mat. and rows are rows
// cols and rows are same ase width and heigth
resize(LoadedImage, LoadedImage, Size(LoadedImage.cols/2, LoadedImage.rows/2));

// Vizualization
namedWindow("Step 4 image resize better", WINDOW_AUTOSIZE);
imshow("Step 4 image resize better", LoadedImage);
waitKey(1000);

// Save
imwrite("Step4.JPG", LoadedImage);
//All the steps are saved in Step1 Step
*/
