#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/core/cvstd.hpp>

#define SIZE_WIDTH 800

//https://blog.naver.com/tommybee/221424578644
static void compute_skew(const char* filename)

{
    // Load in grayscale.
    cv::Mat imgOrg = cv::imread(filename, 0), img;

    //resize image
    //https://answers.opencv.org/question/12825/how-can-i-resize-an-image-with-opencv/
    //double scale = float(SIZE_WIDTH)/imgOrg.size().width;
    //cv::resize(imgOrg, imgRs, cv::Size(0, 0), scale, scale);
    //std::cout<< "scale : "  << scale <<std::endl;

    //convert to gray scale image
    //cv::Mat imgRs;
    //cv::cvtColor( imgRs, img, CV_RGB2GRAY );

    pyrDown(imgOrg, img);

    // Binarize
    cv::threshold(img, img, 255, 255, cv::THRESH_BINARY);
    medianBlur(img, img, 9);

    // Invert colors
    cv::bitwise_not(img, img);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3));
    cv::erode(img, img, element);

    std::vector<cv::Point> points;

    cv::Mat_<uchar>::iterator it = img.begin<uchar>();
    cv::Mat_<uchar>::iterator end = img.end<uchar>();

    for (; it != end; ++it)
        if (*it)
            points.push_back(it.pos());

    std::cout<< "points : "  << points.size() <<std::endl;
    cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));
    double angle = box.angle;

    if (angle < -45.) angle += 90.;

    cv::Point2f vertices[4];
    box.points(vertices);

    for(int i = 0; i < 4; ++i)
        cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 1, CV_AA);

    std::cout << "File " << filename << ": " << angle << std::endl;

    cv::imshow("Result", img);
    cv::waitKey(0);

}

//https://stackoverflow.com/questions/34415815/detect-text-in-images-with-opencv
static void detect_text(std::string input){
    cv::Mat large = cv::imread(input);

    cv::Mat rgb;
    // downsample and use it for processing
    pyrDown(large, rgb);
    pyrDown(rgb, rgb);
    cv::Mat small;
    cvtColor(rgb, small, CV_BGR2GRAY);
    // morphological gradient
    cv::Mat grad;
    cv::Mat morphKernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    morphologyEx(small, grad, cv::MORPH_GRADIENT, morphKernel);
    // binarize
    cv::Mat bw;
    threshold(grad, bw, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    // connect horizontally oriented regions
    cv::Mat connected;
    morphKernel = getStructuringElement(cv::MORPH_RECT, cv::Size(9, 1));
    morphologyEx(bw, connected, cv::MORPH_CLOSE, morphKernel);
    // find contours
    cv::Mat mask = cv::Mat::zeros(bw.size(), CV_8UC1);
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    // filter contours
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0]){
        cv::Rect rect = boundingRect(contours[idx]);
        cv::Mat maskROI(mask, rect);
        maskROI = cv::Scalar(0, 0, 0);
        // fill the contour
        drawContours(mask, contours, idx, cv::Scalar(255, 255, 255), CV_FILLED);

        cv::RotatedRect rrect = minAreaRect(contours[idx]);
        double r = (double)countNonZero(maskROI) / (rrect.size.width * rrect.size.height);

        cv::Scalar color;
        int thickness = 1;
        // assume at least 25% of the area is filled if it contains text
        if (r > 0.25 &&
        (rrect.size.height > 8 && rrect.size.width > 8) // constraints on region size
        // these two conditions alone are not very robust. better to use something
        //like the number of significant peaks in a horizontal projection as a third condition
        ){
            std::cout << "size.height: " << rrect.size.height << " size.width: " << rrect.size.width << " r: " << r << std::endl;
            //if(rrect.size.height == 144 || rrect.size.height == 358 || rrect.size.height == 113) //image area
            //if(rrect.size.height > 20 && rrect.size.height < 21) //image area 20.7883
            //if(rrect.size.height > 23 && rrect.size.height < 24) //image area 23.5882
            //if(rrect.size.height > 15 && rrect.size.height < 16) //image area 15.6813
            //if(rrect.size.height > 28 && rrect.size.height < 29) //image area 28.2888
            //if(rrect.size.height > 27 && rrect.size.height <28) //image area 28.2888
            //if(rrect.size.height > 43 && rrect.size.height <44) //image area 43.6576
            //if(rrect.size.height > 12 && rrect.size.height <13) //image area 12.8155
            //if(rrect.size.height == 358) //image area
            //if(rrect.size.height == 144) //image area
            //if(rrect.size.height > 28 && rrect.size.height <29) //text area
            //if(rrect.size.height > 196 && rrect.size.height < 197) //text area 196.955
            //if(rrect.size.height > 222 && rrect.size.height < 223) //text area 222.084
            //if(rrect.size.height == 24 || rrect.size.height ==25) //text area
            //if(rrect.size.height == 23) //text area
            //if((rrect.size.height > 196 && rrect.size.height < 197)
            //   ||(rrect.size.height > 222 && rrect.size.height < 223)
            //   ||(rrect.size.height == 24 || rrect.size.height ==25)
            //   ||(rrect.size.height == 23)
            //   ) //text area
            {
                thickness = 2;
                color = cv::Scalar(0, 255, 0);
            }

        }
        else
        {
            thickness = 1;
            color = cv::Scalar(0, 0, 255);
        }

        cv::Point2f pts[4];
        rrect.points(pts);
        for (int i = 0; i < 4; i++)
        {
            line(rgb, cv::Point((int)pts[i].x, (int)pts[i].y),
                 cv::Point((int)pts[(i+1)%4].x, (int)pts[(i+1)%4].y), color, thickness);
        }
    }

    //imwrite("cont.jpg", rgb);
    cv::imshow("Result", rgb);
    cv::waitKey(0);

}

static void detect_mser(const std::string input, bool isDrawRect){
    cv::Mat img = cv::imread(input, 1);
    cv::Mat rgb;
    // downsample and use it for processing
    pyrDown(img, rgb);
    pyrDown(rgb, rgb);

    cv::Mat small;
    cvtColor(rgb, small, CV_BGR2GRAY);

    cv::Ptr<cv::MSER> ms = cv::MSER::create();

    cv::Mat detMat;
    small.copyTo(detMat);
    //cv::Ptr<cv::MSER> ms = cv::MSER::create(21, (int)(0.00002*small.cols*small.rows), (int)(0.05*small.cols*small.rows), 1, 0.7);
    std::vector<std::vector<cv::Point> > regions;
    std::vector<cv::Rect> mser_bbox;
    ms->detectRegions(detMat, regions, mser_bbox);

    cv::RNG rng(12345);

    if(isDrawRect)
        for (int i = 0; i < regions.size(); i++)
        {
            cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
            rectangle(rgb, mser_bbox[i], color);
        }
    else
        for (std::vector<cv::Point> v : regions){
            unsigned bVal = rng.uniform(0,255);
            unsigned gVal = rng.uniform(0,255);
            unsigned rVal = rng.uniform(0,255);

            for (cv::Point p : v){
                rgb.at<cv::Vec3b>(p.y, p.x)[0] = bVal;
                rgb.at<cv::Vec3b>(p.y, p.x)[1] = gVal;
                rgb.at<cv::Vec3b>(p.y, p.x)[2] = rVal;
            }
        }


    imshow("mser", rgb);
    cv::waitKey(0);

}

static void detect_mser2(const std::string input, const bool isDrawRect){
    cv::Mat large = cv::imread(input);

    cv::Mat rgb;
    // downsample and use it for processing
    pyrDown(large, rgb);
    pyrDown(rgb, rgb);
    cv::Mat small;
    cvtColor(rgb, small, CV_BGR2GRAY);
    // morphological gradient
    cv::Mat grad;
    cv::Mat morphKernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    morphologyEx(small, grad, cv::MORPH_GRADIENT, morphKernel);
    // binarize
    cv::Mat bw;
    threshold(grad, bw, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    // connect horizontally oriented regions
    cv::Mat connected;
    morphKernel = getStructuringElement(cv::MORPH_RECT, cv::Size(9, 1));
    morphologyEx(bw, connected, cv::MORPH_CLOSE, morphKernel);

    cv::Mat detMat;
    bw.copyTo(detMat);

    cv::Ptr<cv::MSER> ms = cv::MSER::create();

    std::vector<std::vector<cv::Point> > regions;
    std::vector<cv::Rect> mser_bbox;
    ms->detectRegions(detMat, regions, mser_bbox);

    cv::RNG rng(12345);

    if(isDrawRect)
        for (int i = 0; i < regions.size(); i++)
        {
            cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
            rectangle(rgb, mser_bbox[i], color);
        }
    else
        for (std::vector<cv::Point> v : regions){
            unsigned bVal = rng.uniform(0,255);
            unsigned gVal = rng.uniform(0,255);
            unsigned rVal = rng.uniform(0,255);

            for (cv::Point p : v){
                rgb.at<cv::Vec3b>(p.y, p.x)[0] = bVal;
                rgb.at<cv::Vec3b>(p.y, p.x)[1] = gVal;
                rgb.at<cv::Vec3b>(p.y, p.x)[2] = rVal;
            }
        }

    imshow("mser2", rgb);
    cv::waitKey(0);

}

int	main( void )
{
    const std::string filename = "Handicap_Sign.jpg";
    const bool isDrawRect = false;

    detect_text(filename);
    detect_mser(filename, isDrawRect);
    detect_mser2(filename, isDrawRect);
    //compute_skew(filename.c_str());

    return EXIT_SUCCESS;
}
