#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

#include <limits>

typedef std::numeric_limits< double > dbl;

using namespace std;
using namespace cv;
double getPSNR ( const Mat& I1, const Mat& I2, double &mse);
Scalar getMSSIM( const Mat& i1, const Mat& i2, Mat &ssim_map);
static Mat getMatC3U8C(const cv::Mat& ssim_map, const Scalar similarity);
static Mat getMatC4U8C(const cv::Mat& ssim_map, const Scalar similarity);

extern char showTwoImagesEach(string title, const Mat& img1, const Mat& img2, const string &label);
extern char showTwoImagesComb(string title, const Mat& img1, const Mat& img2, const string &label);
extern cv::Mat createOne(std::vector<cv::Mat> & images, int cols, int min_gap_size);


#define SAVE_MAT 0
#define PRINT_MAT 1
#define PRINT_COUNT 100

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

//https://stackoverflow.com/questions/554063/how-do-i-print-a-double-value-with-full-precision-using-cout
static void showMatC4Image(const cv::Mat& ssim_map, const Scalar similarity)
{
    //float similarity = 0.9f;
    int width = ssim_map.rows;
    int height = ssim_map.cols;

    Mat ssim_map_uchar(width, height, CV_8UC4);

#if PRINT_MAT
    int count = PRINT_COUNT;
#endif

    for (int r = 0; r < width; ++r)
    {
        for (int c = 0; c < height; ++c)
        {
            if (ssim_map.at<cv::Vec4f>(r, c)[0] >= similarity[0]
                || ssim_map.at<cv::Vec4f>(r, c)[1] >= similarity[1]
                || ssim_map.at<cv::Vec4f>(r, c)[2] >= similarity[2]
            ){
                continue;
            }

            ssim_map_uchar.at<cv::Vec4b>(r, c)[0] = 255;//();
            ssim_map_uchar.at<cv::Vec4b>(r, c)[1] = 255;//();
            ssim_map_uchar.at<cv::Vec4b>(r, c)[2] = 255;//();
            ssim_map_uchar.at<cv::Vec4b>(r, c)[3] = 0;//();
#if PRINT_MAT
            if(count > 0)
            {
                double blue = ssim_map.at<cv::Vec4f>(r, c)[0];
                double green = ssim_map.at<cv::Vec4f>(r, c)[1];
                double red = ssim_map.at<cv::Vec4f>(r, c)[2];
                double alpha = ssim_map.at<cv::Vec4f>(r, c)[3];

                cout.precision(dbl::max_digits10);

                cout << blue;
                cout << ":" << green;
                cout << ":" << red;
                cout << ":" << alpha << endl;
                count--;
            }
#endif // PRINT_MAT
        }
    }
#if (SAVE_MAT)
    FileStorage fs("ssim_map_uchar.txt",FileStorage::WRITE);
    fs << "mat2" << merged_isomap_uchar;
    //FileStorage fs("myfile.txt",FileStorage::READ);
    //fs["mat1"] >> m;
#endif

    imshow("1.ssim_map", ssim_map);
    imshow("2. ssim_map_uchar", ssim_map_uchar);

    return;
};

static void showMatC3Image(const cv::Mat& ssim_map, Scalar similarity)
{
    //float similarity = 0.9f;
    int width = ssim_map.rows;
    int height = ssim_map.cols;

    Mat ssim_map_uchar(width, height, CV_8UC3);

#if PRINT_MAT
    int count = PRINT_COUNT;
#endif

    for (int r = 0; r < width; ++r)
    {
        for (int c = 0; c < height; ++c)
        {
            if (ssim_map.at<cv::Vec3f>(r, c)[0] >= similarity[0]
                || ssim_map.at<cv::Vec3f>(r, c)[1] >= similarity[1]
                || ssim_map.at<cv::Vec3f>(r, c)[2] >= similarity[2]
            ){
                continue;
            }

            ssim_map_uchar.at<cv::Vec3b>(r, c)[0] = 255;//();
            ssim_map_uchar.at<cv::Vec3b>(r, c)[1] = 255;//();
            ssim_map_uchar.at<cv::Vec3b>(r, c)[2] = 255;//();
            //ssim_map_uchar.at<cv::Vec4b>(r, c)[3] = 0;//();
#if PRINT_MAT
            if(count > 0)
            {
                double blue = ssim_map.at<cv::Vec3f>(r, c)[0];
                double green = ssim_map.at<cv::Vec3f>(r, c)[1];
                double red = ssim_map.at<cv::Vec3f>(r, c)[2];
                double alpha = ssim_map.at<cv::Vec3f>(r, c)[3];

                cout.precision(dbl::max_digits10);

                cout << blue;
                cout << ":" << green;
                cout << ":" << red;
                cout << ":" << alpha << endl;
                count--;
            }
#endif // PRINT_MAT
        }
    }
#if (SAVE_MAT)
    FileStorage fs("ssim_map_uchar.txt",FileStorage::WRITE);
    fs << "mat2" << merged_isomap_uchar;
    //FileStorage fs("myfile.txt",FileStorage::READ);
    //fs["mat1"] >> m;
#endif

    imshow("1.ssim_map", ssim_map);
    imshow("2. ssim_map_uchar", ssim_map_uchar);

    return;
};

const cv::String keys =
        "{@first |<none>           | original image path}"
        "{@second |<none>           | similar image path}"
        "{help   |      | show help message}";

int main(int argc, const char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("usage: --first <path to original image> --second <path to similar image>");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    cv::String image_org = parser.get<cv::String>(0);
    cv::String image_sim = parser.get<cv::String>(1);

    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }

    Mat mtx1;
	Mat mtx2;
    Mat ssim_map;

    mtx1 = imread(image_org, IMREAD_UNCHANGED);
    if (mtx1.empty()) return 1;
	mtx2 = imread(image_sim, IMREAD_UNCHANGED);
	if (mtx2.empty()) return 1;

    std::ostringstream sOsStream;

    double psnrV, mse;
    Scalar mssimV;

    psnrV = getPSNR(mtx1,mtx2, mse);

    mssimV = getMSSIM(mtx1, mtx2, ssim_map);
    sOsStream << "type: " << type2str(ssim_map.type()) << setiosflags(ios::fixed) << setprecision(3) << " MSE: " << mse << " PSNR: " << psnrV << "dB" ;

    sOsStream << " MSSIM: "
        << " R " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[2] * 100 << "%"
        << " G " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[1] * 100 << "%"
        << " B " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[0] * 100 << "%";

    showTwoImagesEach("Original vs Similar", mtx1, mtx2, sOsStream.str());

    Mat ssim_map_uc8;

    if(ssim_map.channels() == 3)
    {
        //showMatC3Image(ssim_map, similarity);
        ssim_map_uc8 = getMatC3U8C(ssim_map, mssimV);
    }
    else if(ssim_map.channels() == 4)
    {
        //showMatC4Image(ssim_map, similarity);
        ssim_map_uc8 = getMatC4U8C(ssim_map, mssimV);
    }
    else return 10001;

    cvtColor(ssim_map_uc8, ssim_map_uc8, CV_RGBA2GRAY );

    Mat thresh;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    threshold( ssim_map_uc8, thresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU );

    imshow("Thresh", thresh);

    findContours(thresh.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );

    Mat drawing = Mat::zeros(mtx1.size(), CV_8UC3);
    mtx1.copyTo(drawing);

    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( 0, 0, 255 );

        Rect boundRect = boundingRect( contours[i] );

        if(boundRect.width >= drawing.rows) continue;

        rectangle( drawing, boundRect.tl(), boundRect.br(), color, 2 );

    }
    std::vector<cv::Mat> images;

    images.push_back(drawing);
    images.push_back(mtx2);

    int cols = 2;
    int min_gap_size = 2;

    Mat ret = createOne(images, cols, min_gap_size);
    imshow("Three images", ret);

    cv::waitKey(0);
    sOsStream.clear();
    sOsStream.seekp(0);
    ssim_map.release();

    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}

double getPSNR(const Mat& I1, const Mat& I2, double &mse)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    Scalar s = sum(s1);        // sum elements per channel
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    if( sse <= 1e-10) // for small values return zero
    {
        mse = 0;
        return 0;
    }
    else
    {
        mse  = sse / (double)(I1.channels() * I1.total());
        if( mse <= 1e-10) // for small values return zero
            mse = 0;
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

static Mat getMatC3U8C(const cv::Mat& ssim_map, const Scalar similarity)
{
    //float similarity = 0.9f;
    int width = ssim_map.rows;
    int height = ssim_map.cols;

    Mat ssim_map_uchar(width, height, CV_8UC3);

    for (int r = 0; r < width; ++r)
    {
        for (int c = 0; c < height; ++c)
        {
            if (ssim_map.at<cv::Vec3f>(r, c)[0] >= similarity[0]
                || ssim_map.at<cv::Vec3f>(r, c)[1] >= similarity[1]
                || ssim_map.at<cv::Vec3f>(r, c)[2] >= similarity[2]
            ){
                continue;
            }

            ssim_map_uchar.at<cv::Vec3b>(r, c)[0] = 255;
            ssim_map_uchar.at<cv::Vec3b>(r, c)[1] = 255;
            ssim_map_uchar.at<cv::Vec3b>(r, c)[2] = 255;
            //ssim_map_uchar.at<cv::Vec3f>(r, c)[3] = 1;
        }
    }


    return ssim_map_uchar;
};

static Mat getMatC4U8C(const cv::Mat& ssim_map, const Scalar similarity)
{
    //float similarity = 0.9f;
    int width = ssim_map.rows;
    int height = ssim_map.cols;

    Mat ssim_map_uchar(width, height, CV_8UC4);

    for (int r = 0; r < width; ++r)
    {
        for (int c = 0; c < height; ++c)
        {
            if (ssim_map.at<cv::Vec4f>(r, c)[0] >= similarity[0]
                || ssim_map.at<cv::Vec4f>(r, c)[1] >= similarity[1]
                || ssim_map.at<cv::Vec4f>(r, c)[2] >= similarity[2]
            ){
                continue;
            }

            ssim_map_uchar.at<cv::Vec4b>(r, c)[0] = 255;
            ssim_map_uchar.at<cv::Vec4b>(r, c)[1] = 255;
            ssim_map_uchar.at<cv::Vec4b>(r, c)[2] = 255;
            ssim_map_uchar.at<cv::Vec4b>(r, c)[3] = 1;
        }
    }


    return ssim_map_uchar;
};

Scalar getMSSIM( const Mat& i1, const Mat& i2, Mat& ssim_map)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;

    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}
int main4(int argc, char *argv[])
{
    const char* file1 ="jp_gates_original.png";
    const char* file2 = "jp_gates_contrast.png";
    const char* file3 = "jp_gates_photoshopped.png";

    Mat mtx1;
	Mat mtx2;
	Mat mtx3;
	Mat ssim_map;


    mtx1 = imread(file1, IMREAD_UNCHANGED);
    if (mtx1.empty()) return 1;
	mtx2 = imread(file2, IMREAD_UNCHANGED);
	if (mtx2.empty()) return 1;
	mtx3 = imread(file3, IMREAD_UNCHANGED);
	if (mtx3.empty()) return 1;

    std::ostringstream sOsStream;
    std::vector<cv::Mat> images;
    double psnrV, mse;
    Scalar mssimV;

    images.push_back(mtx1);
    images.push_back(mtx2);
    images.push_back(mtx3);
    int cols = 3;
    int min_gap_size = 5;

    Mat ret = createOne(images, cols, min_gap_size);
    imshow("Three images", ret);
    cv::waitKey(0);


    psnrV = getPSNR(mtx1,mtx1, mse);
    sOsStream << setiosflags(ios::fixed) << setprecision(3) << " MSE: " << mse << " PSNR: " << setiosflags(ios::fixed) << setprecision(3) << psnrV << "dB" ;
    mssimV = getMSSIM(mtx1, mtx1, ssim_map);

    sOsStream << " MSSIM: "
        << " R " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[2] * 100 << "%"
        << " G " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[1] * 100 << "%"
        << " B " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[0] * 100 << "%";

    showTwoImagesEach("Original vs Original", mtx1, mtx1, sOsStream.str());
    cv::waitKey(0);
    sOsStream.clear();
    sOsStream.seekp(0);
    ssim_map.release();
    //-----------------//
    psnrV = getPSNR(mtx1,mtx2, mse);
    sOsStream << setiosflags(ios::fixed) << setprecision(3) << " MSE: " << mse << " PSNR: " << setiosflags(ios::fixed) << setprecision(3) << psnrV << "dB" ;
    mssimV = getMSSIM(mtx1, mtx2, ssim_map);

    sOsStream << " MSSIM: "
        << " R " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[2] * 100 << "%"
        << " G " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[1] * 100 << "%"
        << " B " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[0] * 100 << "%";

    showTwoImagesEach("Original vs Contrast", mtx1, mtx2, sOsStream.str());
    cv::waitKey(0);
    sOsStream.clear();
    sOsStream.seekp(0);
    ssim_map.release();

    //-----------------//
    psnrV = getPSNR(mtx1,mtx3, mse);
    sOsStream << setiosflags(ios::fixed) << setprecision(3) << " MSE: " << mse << " PSNR: " << setiosflags(ios::fixed) << setprecision(3) << psnrV << "dB" ;
    mssimV = getMSSIM(mtx1, mtx3,ssim_map);

    sOsStream << " MSSIM: "
        << " R " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[2] * 100 << "%"
        << " G " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[1] * 100 << "%"
        << " B " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[0] * 100 << "%";

    //sOsStream << endl;

    showTwoImagesComb("Original vs Photoshopped", mtx1, mtx3, sOsStream.str());
    cv::waitKey(0);
    sOsStream.clear();
    sOsStream.seekp(0);
    ssim_map.release();
    cv::destroyAllWindows();


    return EXIT_SUCCESS;
}
