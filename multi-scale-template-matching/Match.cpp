#include <cmath>
#include <vector>
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <regex>

#include <opencv2/opencv.hpp>

#include <dirent.h>

using namespace std;
using namespace cv;

static string type2str(int type) {
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

//https://gist.github.com/damithsj/c96a8482b282a3dc89bd
//linspace function in MatLab. hacked the code ;)
// ...\toolbox\matlab\elmat\linspace.m
//generates N points between min and max and return as vector.
vector<double> linspace(double min, double max, int n)
{
	vector<double> result;
	// vector iterator
	int iterator = 0;

	for (int i = 0; i <= n-2; i++)
	{
		double temp = min + i*(max-min)/(floor((double)n) - 1);
		result.insert(result.begin() + iterator, temp);
		iterator += 1;
	}

	//iterator += 1;

	result.insert(result.begin() + iterator, max);
	return result;
}

#define TRIM_SPACE " \t\n\v"

inline std::string trim(std::string& s,const std::string& drop = TRIM_SPACE)
{
    std::string r=s.erase(s.find_last_not_of(drop)+1);
    return r.erase(0,r.find_first_not_of(drop));
}
inline std::string rtrim(std::string s,const std::string& drop = TRIM_SPACE)
{
    return s.erase(s.find_last_not_of(drop)+1);
}
inline std::string ltrim(std::string s,const std::string& drop = TRIM_SPACE)
{
    return s.erase(0,s.find_first_not_of(drop));
}


inline bool exists1 (const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

inline bool exists2 (const std::string& name) {
    return ( access( name.c_str(), F_OK ) != -1 );
}

inline bool exists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

static vector<string> getFileList(const string images_path)
{
    vector<string> filelist;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (images_path.c_str())) != NULL) {

        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            string tmp(ent->d_name);
            string path = images_path + "/" + tmp;
            //tmp = trim(tmp);
            if(exists(trim(path)))
            {
                //cout << ent->d_name  << endl;
                filelist.push_back(path);
            }
            //else cout << "file or directory: " << path  << endl;
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
    }

    return filelist;

}

// Linear interpolation following MATLAB linspace
//https://gist.github.com/mortenpi/f20a93c8ed3ee7785e65
std::vector<double> LinearSpacedArray(double minVal, double maxVal, std::size_t N)
{
    double h = (maxVal - minVal) / static_cast<double>(N-1);
    std::vector<double> xs(N);
    std::vector<double>::iterator x;
    double val;
    for (x = xs.begin(), val = minVal; x != xs.end(); ++x, val += h) {
        *x = val;
    }
    return xs;
}

cv::Mat resizeImage(cv::Mat image, int width=-1, int height=-1, int inter=cv::INTER_AREA)
{
    // initialize the dimensions of the image to be resized and
    // grab the image size
    cv::Size dim;
    cv::Mat resized;

    //(h, w) = image.shape[:2]
    int h = image.rows;
    int w = image.cols;

    // if both the width and height are None, then return the
    // original image
    if(width == -1 && height == -1) return image;

    // check to see if the width is None
    if(width == -1)
    {
        // calculate the ratio of the height and construct the
        // dimensions
        float r = height / float(h);
        dim = cv::Size((int)(w * r), height);
    }
    // otherwise, the height is None
    else
    {
        // calculate the ratio of the width and construct the
        // dimensions
        float r = width / float(w);
        dim = cv::Size(width, (int)(h * r));
    }


    // resize the image
    cv::resize(image, resized, dim, 0, 0, inter); // upscale 2x

    // return the resized image
    return resized;
}

/*
https://docs.opencv.org/2.4/doc/tutorials/introduction/display_image/display_image.html
https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html
https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
CV_LOAD_IMAGE_UNCHANGED (<0) loads the image as is (including the alpha channel if present)
CV_LOAD_IMAGE_GRAYSCALE ( 0) loads the image as an intensity one
CV_LOAD_IMAGE_COLOR (>0) loads the image in the BGR format
*/
int main(int argc, char* argv[])
{
    const cv::String keys =
        "{@template |<none>           | Path to template image}"         // input image is the first argument (positional)
        "{@images   |/path/to/file.xml| Path to images where template will be matched}"        // optional, face cascade is the second argument (positional)
        "{visualize |                 | Flag indicating whether or not to visualize each iteration}"         // optional, default value ""
        "{help      |                 | show help message}";      // optional, show help optional

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("usage: --template <path to template image> --images  <path to images image> [--visualize]");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    Mat templateMat, templateGrayMat;
    //for canny
    const int lowThreshold = 50;
    const int highThreshold = 200;
    bool isVisualized = false;
    bool isFirstRun = false;

    cv::String template_image = parser.get<cv::String>(0); // read @image (mandatory, error if not present)
    cv::String images_path = parser.get<cv::String>(1); // read @face (use default value if not in cmd)

    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }

    if(parser.has("visualize"))
    {
        isVisualized = true;
    }

    templateMat = imread(template_image, CV_LOAD_IMAGE_COLOR);

    if(templateMat.empty())
    {

        cout << "Image is NULL!!!!" << endl;
        return EXIT_SUCCESS;
    }

    cout << "Image is " << type2str(templateMat.type()) << endl;

    cvtColor(templateMat, templateGrayMat, COLOR_BGR2GRAY);
    blur(templateGrayMat, templateGrayMat, Size(3, 3));
	Canny(templateGrayMat, templateGrayMat, lowThreshold, highThreshold);

    int tmpl_height = templateGrayMat.rows;
    int tmpl_width = templateGrayMat.cols;

    vector<string> filelist;
    vector<string>::const_iterator cii;
    const std::regex pic_regex("^.*\\.(png|jpg)$");
    const double minVal = 0.2;
    const double maxVal = 1.0;
    const std::size_t N = 20;

    struct found{
        double maxVal;
        double rad;
        Point maxLoc;
    };

    filelist = getFileList(images_path);
    std::vector<double> range = LinearSpacedArray(minVal, maxVal, N);

    imshow("Template", templateMat);
    isFirstRun = true;

    for(cii=filelist.begin(); cii!=filelist.end(); cii++)
    {
        if (std::regex_match(*cii, pic_regex))
        {
            Mat image = imread(*cii, CV_LOAD_IMAGE_COLOR);
            Mat imageGray;
            float r;
            cvtColor(image, imageGray, COLOR_BGR2GRAY);

            Mat resized, edged, result, templat, clone;
            int img_height = imageGray.rows;
            int img_width = imageGray.cols;
            int width = -1;
            vector<double>::reverse_iterator ciid;
            double minVal; double maxVal; Point minLoc; Point maxLoc;
            struct found* foundobj = (struct found*)malloc(sizeof(struct found));
            foundobj->maxVal = 0;
            foundobj->rad = 0.0f;

            for(ciid=range.rbegin(); ciid!=range.rend(); ++ciid)
            {
                width = img_width * (*ciid);
                //resized = imutils.resize(imageGray, width);
                resized = resizeImage(imageGray, width);

                r = (float)(imageGray.cols / (float)resized.cols);
                // if the resized image is smaller than the template, then break
                // from the loop
                if(resized.rows < tmpl_height || resized.cols < tmpl_width) break;

                // detect edges in the resized, grayscale image and apply template
                // matching to find the template in the image
                cv::Canny(resized, edged, lowThreshold, highThreshold);
                cv::matchTemplate(edged, templateGrayMat, result, cv::TM_CCOEFF);

                cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc);

                // check to see if the iteration should be visualized
                if (isVisualized && isFirstRun)
                {
                    // draw a bounding box around the detected region
                    //clone = np.dstack([edged, edged, edged])
                    //Mat clone = edged.clone();
                    //Mat clone;
                    edged.copyTo(clone);

                    cv::rectangle( clone, maxLoc,
                                  cv::Point( maxLoc.x + tmpl_width , maxLoc.y + tmpl_height ),
                                  cv::Scalar(255, 0, 0), 2 );
                    cv::imshow("Visualize", clone);
                    cv::waitKey(0);
                    //clone.release();
                }

                // if we have found a new maximum correlation value, then update
                // the bookkeeping variable
                if(!foundobj || (maxVal > foundobj->maxVal)) {
                    foundobj->maxVal = maxVal;
                    foundobj->maxLoc = maxLoc;
                    foundobj->rad = r;
                }

            }
            isFirstRun = false;
            maxLoc = foundobj->maxLoc;
            r = (float)foundobj->rad;
            cv::Point startP((int)(maxLoc.x * r), (int)(maxLoc.y * r));
            cv::Point endP((int)((maxLoc.x + tmpl_width) * r), (int)((maxLoc.y + tmpl_height) * r));

            cv::rectangle(image, startP, endP, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Image", image);
            cv::waitKey(0);
            free(foundobj), foundobj = NULL;
        }
    }

    templateMat.release();
    templateGrayMat.release();

    return EXIT_SUCCESS;
}
