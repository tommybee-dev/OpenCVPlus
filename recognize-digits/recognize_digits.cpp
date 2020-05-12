#include <cmath>
#include <vector>
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <regex>

#include <tuple>
#include <iostream>
#include <array>
#include <utility>

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

void displayFoundArea(const Mat image, const vector<Point> displayCnt)
{
    cv::Rect brRect = cv::boundingRect(displayCnt);
    cv::Point TopLeft = brRect.tl();
    cv::Point BottomRight = brRect.br();

    cv::rectangle( image, TopLeft,
                  BottomRight,
                  cv::Scalar(255, 0, 0), 2 );

    imshow("found rect", image);
    waitKey(0);
}

template <typename T> void order_points(const array<Point_<T>, 4> &pts, array<Point_<T>, 4> &ordered)
{
    // initialzie a list of coordinates that will be ordered
    // such that the first entry in the list is the top-left,
    // the second entry is the top-right, the third is the
    // bottom-right, and the fourth is the bottom-left

	// the top-left point will have the smallest sum, whereas
    // the bottom-right point will have the largest sum
    // NOTE: Therefore the "top-left point" is neither necessarily the top-most point,
    // nor necessarily the left-most point.
    array<T, 4> sums = { pts[0].x + pts[0].y, pts[1].x + pts[1].y, pts[2].x + pts[2].y, pts[3].x + pts[3].y };
    // dump_array("sum:", sums);
    ordered[0] = pts[min_element(sums.begin(), sums.end()) - sums.begin()];
    ordered[2] = pts[max_element(sums.begin(), sums.end()) - sums.begin()];

    // now, compute the difference between the points, the
    // top-right point will have the smallest difference,
    // whereas the bottom-left will have the largest difference
    array<T, 4> differences = { pts[0].y - pts[0].x, pts[1].y - pts[1].x, pts[2].y - pts[2].x, pts[3].y - pts[3].x };
    // dump_array("differences:", differences);

    ordered[1] = pts[min_element(differences.begin(), differences.end()) - differences.begin()];
    ordered[3] = pts[max_element(differences.begin(), differences.end()) - differences.begin()];
    // dump_point_array("order_points ordered", ordered);
}

double pythagorean_distance(const Point &p1, const Point &p2)
{
	double xdiff = p2.x - p1.x, ydiff = p2.y - p1.y;
	return sqrt(xdiff * xdiff + ydiff * ydiff);
}

template <typename T> void four_point_transform(const Mat &image, const array<Point_<T>, 4> &pts, Mat &warped)
{
	// dump_point_array("four_point_transform", pts);
    // obtain a consistent order of the points and unpack them
    // individually
	array<Point_<T>, 4> rect;
    order_points(pts, rect);
    // dump_point_array("four_point_transform rect", rect);
    const Point_<T> &tl = rect[0], &tr = rect[1], &br = rect[2], &bl = rect[3];

    // compute the width of the new image, which will be the
    // maximum distance between bottom-right and bottom-left
    // x-coordiates or the top-right and top-left x-coordinates
    T maxWidth = max(pythagorean_distance(br, bl), pythagorean_distance(tr, tl));

    // compute the height of the new image, which will be the
    // maximum distance between the top-right and bottom-right
    // y-coordinates or the top-left and bottom-left y-coordinates
    T maxHeight = max(pythagorean_distance(tr, br), pythagorean_distance(tl, bl));

    // now that we have the dimensions of the new image, construct
    // the set of destination points to obtain a "birds eye view",
    // (i.e. top-down view) of the image, again specifying points
    // in the top-left, top-right, bottom-right, and bottom-left
    // order
    array<Point_<T>, 4> dst = {{{0, 0}, {maxWidth - 1, 0}, {maxWidth - 1, maxHeight - 1}, {0, maxHeight - 1}}};
    // dump_array("dst", dst);

    // compute the perspective transform matrix and then apply it
    Mat M = getPerspectiveTransform(&rect[0], &dst[0]);
    // cerr << "Matrix: " << M << endl;
    // display_image("Perspective input", image);
    warpPerspective(image, warped, M, Size(maxWidth, maxHeight));
    // display_image("Perspective output", warped);
}

static pair<tuple<int, int, int, int, int, int, int>, string> DIGITS_LOOKUP[10];


void makeLookupTable()
{
    DIGITS_LOOKUP[0] = make_pair(make_tuple(1, 1, 1, 0, 1, 1, 1), "0"); //: 0,
	DIGITS_LOOKUP[1] = make_pair(make_tuple(0, 0, 1, 0, 0, 1, 0), "1"); //: 1,
	DIGITS_LOOKUP[2] = make_pair(make_tuple(1, 0, 1, 1, 1, 1, 0), "2"); //: 2,
	DIGITS_LOOKUP[3] = make_pair(make_tuple(1, 0, 1, 1, 0, 1, 1), "3"); //: 3,
	DIGITS_LOOKUP[4] = make_pair(make_tuple(0, 1, 1, 1, 0, 1, 0), "4"); //: 4,
	DIGITS_LOOKUP[5] = make_pair(make_tuple(1, 1, 0, 1, 0, 1, 1), "5"); //: 5,
	DIGITS_LOOKUP[6] = make_pair(make_tuple(1, 1, 0, 1, 1, 1, 1), "6"); //: 6,
	DIGITS_LOOKUP[7] = make_pair(make_tuple(1, 0, 1, 0, 0, 1, 0), "7"); //: 7,
	DIGITS_LOOKUP[8] = make_pair(make_tuple(1, 1, 1, 1, 1, 1, 1), "8"); //: 8,
	DIGITS_LOOKUP[9] = make_pair(make_tuple(1, 1, 1, 1, 0, 1, 1), "9"); //: 9
}

void getNumString(const int on[], string& ret)
{
    int len = sizeof(DIGITS_LOOKUP)/sizeof(*DIGITS_LOOKUP);
    tuple<int, int, int, int, int, int, int> t_on = make_tuple(on[0], on[1], on[2], on[3], on[4], on[5], on[6]);

    for(int i = 0; i< len; i++)
    {

        if(t_on == DIGITS_LOOKUP[i].first)
        {
            //cout << "................" << DIGITS_LOOKUP[i].second;
            ret = DIGITS_LOOKUP[i].second;
            break;
        }
    }
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
    Mat image, gray, edged, blurred;
    Mat warped, output;
    vector<vector<Point> > cnts;
    vector<Vec4i> hierarchy;

    image = imread("images/recognize-digits.jpg", CV_LOAD_IMAGE_COLOR);

    if(image.empty())
    {
        cout << "Image is NULL!!!!" << endl;
        return EXIT_SUCCESS;
    }

    cout << "Image is " << type2str(image.type()) << endl;

    int height=500;
    int lowThreshold =50, highThreshold=200;

    makeLookupTable();

    // pre-process the image by resizing it, converting it to
    // graycale, blurring it, and computing an edge map
    image = resizeImage(image, -1, height);
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur( gray, blurred, Size( 5, 5 ), 0, 0 );
    //edged = cv2.Canny(blurred, 50, 200, 255)
    Canny(blurred, edged, lowThreshold, highThreshold);

    vector<vector<Point>>::reverse_iterator ciid;

    vector<Point> displayCnt, approx;
    double peri;

    //# sort the contours from left-to-right, then initialize the
    //# actual digits themselves
    // sort contours
    struct contour_sorter // 'less' for contours
    {
        bool operator ()( const vector<Point>& a, const vector<Point> & b )
        {
            Rect ra(boundingRect(a));
            Rect rb(boundingRect(b));
            return (ra.x > rb.x);
        }
    };

    // find contours in the edge map, then sort them by their
    // size in descending order
    //cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    findContours(edged.clone(), cnts, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
    //findContours(edged.clone(), cnts, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

    std::sort(cnts.begin(), cnts.end(), contour_sorter());

    // loop over the contours
    for(ciid=cnts.rbegin(); ciid!=cnts.rend(); ++ciid)
    {
        // approximate the contour
        //peri = arcLength(Mat(*cii), true);
        peri = arcLength(*ciid, true);

        approxPolyDP(*ciid, approx, 0.02 * peri, true);

        // if the contour has four vertices, then we have found
        // the thermostat display
        int size = approx.size();

        if(size == 4)
        {
            displayCnt = approx;

            break;
        }

    }

    displayFoundArea(image, displayCnt);

    std::array<Point2f, 4> border_arr;
    std::copy_n(displayCnt.begin(), 4, border_arr.begin());

    four_point_transform(gray, border_arr, warped);
    four_point_transform(image, border_arr, output);


    Mat thresh, kernel;
    // threshold the warped image, then apply a series of morphological
    // operations to cleanup the thresholded image
    threshold( warped, thresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU );
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 5));
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);

    imshow("thresh", thresh);
    waitKey(0);

    cnts.clear();
    hierarchy.clear();

    vector<vector<Point> > digitCnts;
    // find contours in the thresholded image, then initialize the
    // digit contours lists
    findContours(thresh.clone(), cnts, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
    Rect rect;

    //cout<< cnts.size() << endl;
    //# loop over the digit area candidates
    for(ciid=cnts.rbegin(); ciid!=cnts.rend(); ++ciid)
    {
        //# compute the bounding box of the contour
        //displayFoundArea(thresh, *ciid);
        rect = boundingRect( *ciid );
        //# if the contour is sufficiently large, it must be a digit
        if(rect.width >= 15 && (rect.height >= 30 && rect.height <= 40))
        {
            //displayFoundArea(warped, *ciid);
            digitCnts.push_back(*ciid);
        }
    }

    // apply it to the contours:
    std::sort(digitCnts.begin(), digitCnts.end(), contour_sorter());

    int roiH, roiW, dW, dH, dHC;
    vector<vector<Point>>::const_iterator cii;

    // loop over each of the digits
    for(ciid=digitCnts.rbegin(); ciid!=digitCnts.rend(); ++ciid)
    {
        vector<tuple<Point2f, Point2f>> segments;
        int on[7];

        // extract the digit ROI
        //rect = boundingRect( Mat(*ciid) );
        rect = boundingRect( *ciid );
        int x = rect.x;
        int y = rect.y;
        int w = rect.width;
        int h = rect.height;

        Mat roi = thresh(rect).clone();

        // compute the width and height of each of the 7 segments
        // we are going to examine
        roiH=roi.rows;//rect.height;
        roiW=roi.cols;//rect.width;
        dW=(int)(roiW * 0.25);
        dH=(int)(roiH * 0.15);
        dHC=(int)(roiH * 0.05);


        segments.push_back(make_tuple(Point2f(0, 0), Point2f(w, dH)));	// top
        segments.push_back(make_tuple(Point2f(0, 0), Point2f(dW, h / 2)));	// top-left
        segments.push_back(make_tuple(Point2f(w - dW, 0), Point2f(w, (int)h / 2)));	// top-right
        segments.push_back(make_tuple(Point2f(0, (h / 2) - dHC), Point2f(w, (int)(h / 2) + dHC))); // center
        segments.push_back(make_tuple(Point2f(0, h / 2), Point2f(dW, h)));	// bottom-left
        segments.push_back(make_tuple(Point2f(w - dW, (int)h / 2), Point2f(w, h)));	// bottom-right
        segments.push_back(make_tuple(Point2f(0, h - dH), Point2f(w, h)));	// bottom
        memset(on, 0x00, sizeof(on));

        for(unsigned i = 0; i < segments.size(); i++)
        {
            Point2f pA = std::get<0>(segments[i]);
            Point2f pB = std::get<1>(segments[i]);
            float xA, yA, xB, yB;

            xA = pA.x;
            yA = pA.y;
            xB = pB.x;
            yB = pB.y;

            Rect segRect(pA, pB);

            Mat segROI = roi(segRect);

            float total = countNonZero(segROI);
            float area = (xB - xA) * (yB - yA);

            // if the total number of non-zero pixels is greater than
            // 50% of the area, mark the segment as "on"
            if (total / float(area) > 0.5)
                on[i]= 1;
        }

        string digit;
        getNumString(on, digit);

        cv::rectangle( output, rect.tl(),rect.br(),cv::Scalar(0, 255, 0), 1 );
        cv::putText(output, digit, cv::Point(x - 10, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.65, CV_RGB(0,255,0), 2, 8);

        imshow("Window", output);

        waitKey(0);
    }

    warped.release(), output.release();
    image.release();
    gray.release();
    blurred.release();
    edged.release();
    thresh.release();

    return EXIT_SUCCESS;
}
