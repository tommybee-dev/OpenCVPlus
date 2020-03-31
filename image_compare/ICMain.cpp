#include "SSIM.h"
#include "PSNR.h"

extern double mse(Mat & m0, Mat & m1, bool grayscale = false, bool rooted = false);
extern double rmse(Mat & m0, Mat & m1);
extern double psnr(Mat & m0, Mat & m1, int block_size);
extern double ssim(Mat & m0, Mat & m1, int block_size);

//https://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/
char showTwoImagesEach(string title, const Mat& img1, const Mat& img2, const string &label)
{
    char k=0;
    namedWindow(title);
    Mat img_result;
    hconcat(img1, img2, img_result); //두 이미지의 높이(row)가 같아야함.
    putText(img_result, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 212, 255));

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
    putText(matDst, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 212, 255));
    imshow(title,matDst);

    k=waitKey(10);
    return k;
}

cv::Mat createOne(std::vector<cv::Mat> & images, int cols, int min_gap_size)
{
    // let's first find out the maximum dimensions
    int max_width = 0;
    int max_height = 0;
    for ( int i = 0; i < images.size(); i++) {
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

int main(int argc, char* argv[])
{
    const char* file1 ="jp_gates_original.png";
    const char* file2 = "jp_gates_contrast.png";
    const char* file3 = "jp_gates_photoshopped.png";

    Mat mtx1;
	Mat mtx2;
	Mat mtx3;


    mtx1 = imread(file1, IMREAD_UNCHANGED);
    if (mtx1.empty()) return 1;
	mtx2 = imread(file2, IMREAD_UNCHANGED);
	if (mtx2.empty()) return 1;
	mtx3 = imread(file3, IMREAD_UNCHANGED);
	if (mtx3.empty()) return 1;

    REPORT_DATA stats;

    Scalar ssimScalar;
    double mseVal;
    double psnrVal;
    double rmseVal;
    double ssimVal;

    CMP_Feedback_Proc pFeedbackProc = NULL;

    std::ostringstream sOsStream;
    std::vector<cv::Mat> images;
    images.push_back(mtx1);
    images.push_back(mtx2);
    images.push_back(mtx3);
    int cols = 3;
    int min_gap_size = 5;

    Mat ret = createOne(images, cols, min_gap_size);
    imshow("Three images", ret);
    cv::waitKey(0);

    getMSE_PSNR( mtx1, mtx1, mseVal, psnrVal);
    ssimScalar = getSSIM( mtx1, mtx1, pFeedbackProc);
    CalcPSNR(mtx1, mtx1, &stats);

    sOsStream << "MSE :(" << mseVal << "/" << stats.MSE << ") PSNR:(" << psnrVal << "/" << stats.PSNR << ") SSIM:" << ssimScalar[0];
    showTwoImagesEach("Original vs Original", mtx1, mtx1, sOsStream.str());
    cv::waitKey(0);
    sOsStream.clear();
    sOsStream.seekp(0);

    getMSE_PSNR( mtx1, mtx2, mseVal, psnrVal);
    ssimScalar = getSSIM( mtx1, mtx2, pFeedbackProc);
    CalcPSNR(mtx1, mtx2, &stats);

    sOsStream << "MSE :(" << mseVal << "/" << stats.MSE << ") PSNR:(" << psnrVal << "/" << stats.PSNR << ") SSIM:" << ssimScalar[0];
    showTwoImagesEach("Original vs Contrast", mtx1, mtx2, sOsStream.str());
    cv::waitKey(0);
    sOsStream.clear();
    sOsStream.seekp(0);

    getMSE_PSNR( mtx1, mtx3, mseVal, psnrVal);
    ssimScalar = getSSIM( mtx1, mtx3, pFeedbackProc);
    CalcPSNR(mtx1, mtx3, &stats);

    sOsStream << "MSE :(" << mseVal << "/" << stats.MSE << ") PSNR:(" << psnrVal << "/" << stats.PSNR << ") SSIM:" << ssimScalar[0];
    showTwoImagesComb("Original vs Photoshopped", mtx1, mtx3, sOsStream.str());
    cv::waitKey(0);
    sOsStream.clear();
    sOsStream.seekp(0);

    cv::destroyAllWindows();


    return EXIT_SUCCESS;
}

int main_org(int argc, char* argv[])
{
    const char* file1 ="jp_gates_original.png";
    const char* file2 = "jp_gates_contrast.png";
    const char* file3 = "jp_gates_photoshopped.png";

    Mat mtx1;
	Mat mtx2;
	Mat mtx3;


    mtx1 = imread(file1, IMREAD_UNCHANGED);
    if (mtx1.empty()) return 1;
	mtx2 = imread(file2, IMREAD_UNCHANGED);
	if (mtx2.empty()) return 1;
	mtx3 = imread(file3, IMREAD_UNCHANGED);
	if (mtx3.empty()) return 1;


    REPORT_DATA stats;
    Scalar ssimScalar;
    double mseVal;
    double psnrVal;
    double rmseVal;
    double ssimVal;

    CMP_Feedback_Proc pFeedbackProc = NULL;


    std::ostringstream sOsStream;

    getMSE_PSNR( mtx1, mtx1, mseVal, psnrVal);
    ssimScalar = getSSIM( mtx1, mtx1, pFeedbackProc);

    sOsStream << "MSE : " << mseVal << " PSNR: " << psnrVal << " SSIM: " << ssimScalar[0];
    showTwoImagesEach("Original vs Original", mtx1, mtx1, sOsStream.str());
    cv::waitKey(0);
    sOsStream.clear();
    sOsStream.seekp(0);

    getMSE_PSNR( mtx1, mtx2, mseVal, psnrVal);
    ssimScalar = getSSIM( mtx1, mtx2, pFeedbackProc);

    sOsStream << "MSE : " << mseVal << " PSNR: " << psnrVal << " SSIM: " << ssimScalar[0];
    showTwoImagesEach("Original vs Contrast", mtx1, mtx2, sOsStream.str());
    cv::waitKey(0);
    sOsStream.clear();
    sOsStream.seekp(0);

    getMSE_PSNR( mtx1, mtx3, mseVal, psnrVal);
    ssimScalar = getSSIM( mtx1, mtx3, pFeedbackProc);

    sOsStream << "MSE : " << mseVal << " PSNR: " << psnrVal << " SSIM: " << ssimScalar[0];
    showTwoImagesComb("Original vs Photoshopped", mtx1, mtx3, sOsStream.str());
    cv::waitKey(0);
    sOsStream.clear();
    sOsStream.seekp(0);

    mseVal = mse(mtx1, mtx1);
    psnrVal = psnr(mtx1, mtx1, 10);
    rmseVal = rmse(mtx1, mtx1);
    ssimVal = ssim(mtx1, mtx1, 10);
    sOsStream << "MSE : " << mseVal << " RMSE : " << rmseVal << " PSNR: " << psnrVal << " SSIM: " << ssimVal;
    showTwoImagesEach("Another Original vs Original", mtx1, mtx1, sOsStream.str());
    cv::waitKey(0);
    sOsStream.clear();
    sOsStream.seekp(0);


    //https://answers.opencv.org/question/13876/read-multiple-images-from-folder-and-concat-display-images-in-single-window-opencv-c-visual-studio-2010/
    std::vector<cv::Mat> images;
    images.push_back(mtx1);
    images.push_back(mtx2);
    images.push_back(mtx3);
    int cols = 3;
    int min_gap_size = 5;
    Mat ret = createOne(images, cols, min_gap_size);
    imshow("multiple images in one", ret);
    cv::waitKey(0);

    cv::destroyAllWindows();


    return EXIT_SUCCESS;
}

int main_combine_images(int argc, char* argv[])
{
    const char* file1 ="jp_gates_original.png";
    const char* file2 = "jp_gates_contrast.png";
    const char* file3 = "jp_gates_photoshopped.png";

    Mat mtx1;
	Mat mtx2;
	Mat mtx3;


    mtx1 = imread(file1, IMREAD_UNCHANGED);
    if (mtx1.empty()) return 1;
	mtx2 = imread(file2, IMREAD_UNCHANGED);
	if (mtx2.empty()) return 1;
	mtx3 = imread(file3, IMREAD_UNCHANGED);
	if (mtx3.empty()) return 1;


    showTwoImagesEach("첫번째 방법", mtx1, mtx1, "Method #1");
    cv::waitKey(0);
    showTwoImagesComb("두번째 방법", mtx1, mtx3, "Method #2");
    cv::waitKey(0);

    std::vector<cv::Mat> images;
    images.push_back(mtx1);
    images.push_back(mtx2);
    images.push_back(mtx3);
    int cols = 3;
    int min_gap_size = 5;

    Mat ret = createOne(images, cols, min_gap_size);
    imshow("여러 이미지 합치", ret);
    cv::waitKey(0);

    cv::destroyAllWindows();


    return EXIT_SUCCESS;
}
