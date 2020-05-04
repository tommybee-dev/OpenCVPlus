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
//double pythagorean_distance(int x1, int y1, int x2, int y2) { return sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1)); }
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

int main(int argc, char* argv[])
{
    const cv::String keys =
        "{@images   |/path/to/image.png| Path to images where template will be matched}"        // optional, face cascade is the second argument (positional)
        "{help      |                 | show help message}";      // optional, show help optional

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("usage: --template <path to template image> --images  <path to images image> [--visualize]");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    Mat templateMat, templateGrayMat;


    cv::String image_path = parser.get<cv::String>(0); // read @image (mandatory, error if not present)

    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }

    Mat warped;
    Mat image1 = imread("images/example_01.png");
    Mat image2 = imread("images/example_02.png");
    Mat image3 = imread("images/example_03.png");

    std::array<Point2f, 4> pnts1 =
        {cv::Point(73, 239),    cv::Point(356, 117),    cv::Point(475, 265), cv::Point(187, 443)};
    four_point_transform(image1, pnts1, warped);
    imshow("original", image1);
    imshow("warped1", warped);
    waitKey(0);

    pnts1 = {cv::Point(101, 185),    cv::Point(393, 151),    cv::Point(479, 323), cv::Point(187, 441)};
    four_point_transform(image2, pnts1, warped);
    imshow("original", image2);
    imshow("warped2", warped);
    waitKey(0);

    pnts1 = {cv::Point(63, 242),    cv::Point(291, 110),    cv::Point(361, 252), cv::Point(78, 386)};
    four_point_transform(image3, pnts1, warped);
    imshow("original", image3);
    imshow("warped3", warped);
    waitKey(0);

    return EXIT_SUCCESS;
}
