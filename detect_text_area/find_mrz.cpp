#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/core/cvstd.hpp>

bool find_mrz(const cv::Mat &original, cv::Mat &mrz)
{
    // initialize a rectangular and square structuring kernel
    cv::Mat rectKernel = getStructuringElement(cv::MORPH_RECT, cv::Size(13, 5));
    cv::Mat sqKernel = getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));

    // resize the image and convert it to grayscale
    cv::Mat image;
    resize(original, image, cv::Size(original.size().width * 600 / original.size().height, 600));
    cv::Mat gray;
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // smooth the image using a 3x3 Gaussian, then apply the blackhat
    // morphological operator to find dark regions on a light background
    GaussianBlur(gray, gray, cv::Size(3, 3), 0);
    cv::Mat blackhat;
    morphologyEx(gray, blackhat, cv::MORPH_BLACKHAT, rectKernel);

#ifdef DISPLAY_INTERMEDIATE_IMAGES
    display_image("Blackhat", blackhat);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */

    // compute the Scharr gradient of the blackhat image and scale the
    // result into the range [0, 255]
    cv::Mat gradX;
    Sobel(blackhat, gradX, CV_32F, 1, 0, -1);
    gradX = abs(gradX);
    double minVal, maxVal;
    minMaxIdx(gradX, &minVal, &maxVal);
    cv::Mat gradXfloat = (255 * ((gradX - minVal) / (maxVal - minVal)));
    gradXfloat.convertTo(gradX, CV_8UC1);

#ifdef DISPLAY_INTERMEDIATE_IMAGES
    display_image("Gx", gradX);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */

    // apply a closing operation using the rectangular kernel to close
    // gaps in between letters -- then apply Otsu's thresholding method
    morphologyEx(gradX, gradX, cv::MORPH_CLOSE, rectKernel);
    cv::Mat thresh;
    threshold(gradX, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

#ifdef DISPLAY_INTERMEDIATE_IMAGES
    display_image("Horizontal closing", thresh);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */

    // perform another closing operation, this time using the square
    // kernel to close gaps between lines of the MRZ, then perform a
    // series of erosions to break apart connected components
    morphologyEx(thresh, thresh, cv::MORPH_CLOSE, sqKernel);
    cv::Mat nullKernel;
    erode(thresh, thresh, nullKernel, cv::Point(-1, -1), 4);

#ifdef DISPLAY_INTERMEDIATE_IMAGES
    display_image("Vertical closing", thresh);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */

    // during thresholding, it's possible that border pixels were
    // included in the thresholding, so let's set 5% of the left and
    // right borders to zero
    double p = image.size().height * 0.05;
    thresh = thresh(cv::Rect(p, p, image.size().width - 2 * p, image.size().height - 2 * p));

#ifdef DISPLAY_INTERMEDIATE_IMAGES
    display_image("Border removal", thresh);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */

    // find contours in the thresholded image and sort them by their
    // size
    std::vector<std::vector<cv::Point> > contours;
    findContours(thresh, contours, cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);
    // Sort the contours in decreasing area
    sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2){
        return contourArea(c1, false) > contourArea(c2, false);
    });

    // Find the first contour with the right aspect ratio and a large width relative to the width of the image
    cv::Rect roiRect(0, 0, 0, 0);
    std::vector<std::vector<cv::Point> >::iterator border_iter
        = find_if(contours.begin(), contours.end(), [&roiRect, gray](std::vector<cv::Point> &contour) {
        // compute the bounding box of the contour and use the contour to
        // compute the aspect ratio and coverage ratio of the bounding box
        // width to the width of the image
        roiRect = boundingRect(contour);
        // dump_rect("Bounding rect", roiRect);
        // pprint([x, y, w, h])
        double aspect = (double) roiRect.size().width / (double) roiRect.size().height;
        double coverageWidth = (double) roiRect.size().width / (double) gray.size().height;
        // cerr << "aspect=" << aspect << "; coverageWidth=" << coverageWidth << endl;
        // check to see if the aspect ratio and coverage width are within
        // acceptable criteria
        if (aspect > 5 and coverageWidth > 0.5) {
            return true;
        }
        return false;
    });

    if (border_iter == contours.end()) {
        return false;
    }

    // Correct ROI for border removal offset
    roiRect += cv::Point(p, p);
    // pad the bounding box since we applied erosions and now need
    // to re-grow it
    int pX = (roiRect.x + roiRect.size().width) * 0.03;
    int pY = (roiRect.y + roiRect.size().height) * 0.03;
    roiRect -= cv::Point(pX, pY);
    roiRect += cv::Size(pX * 2, pY * 2);
    // Ensure ROI is within image
    roiRect &= cv::Rect(0, 0, image.size().width, image.size().height);
    // Make it relative to original image again
    float scale = static_cast<float>(original.size().width) / static_cast<float>(image.size().width);
    roiRect.x *= scale;
    roiRect.y *= scale;
    roiRect.width *= scale;
    roiRect.height *= scale;
    mrz = original(roiRect);

#if 0 || defined(DISPLAY_INTERMEDIATE_IMAGES)
    // Draw a bounding box surrounding the MRZ
    Mat display_roi = original.clone();
    rectangle(display_roi, roiRect, Scalar(0, 255, 0), 2);
    display_image("MRZ detection results", display_roi);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */

    return true;
}

int	main3( int argc, char *argv[] )
{
    cv::CommandLineParser parser( argc, argv, "{@images | | input image}" );

    //https://answers.opencv.org/question/69712/load-multiple-images-from-a-single-folder/
    cv::String path(parser.get<cv::String>( "@images" )); //select only jpg
    std::vector<cv::String> fn;
    std::vector<cv::Mat> data;
    cv::glob(path,fn,true); // recurse

    for (size_t k=0; k<fn.size(); ++k)
    {
        cv::Mat src = cv::imread(fn[k]);

        if( src.empty() )
        {
            std::cout << "Could not open or find the image!\n" << std::endl;
            std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
            return -1;
        }
        // Show source image
        //
        data.push_back(src);
        //cv::waitKey(0);
    }

    // initialize a rectangular and square structuring kernel
    cv::Mat rectKernel = getStructuringElement(cv::MORPH_RECT, cv::Size(13, 5));
    cv::Mat sqKernel = getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));

    std::vector<cv::Mat>::iterator iter;

    for (iter = data.begin(); iter != data.end(); ++iter){
        // resize the image and convert it to grayscale
        cv::Mat original = *iter;
        cv::Mat mrz;

        // resize the image and convert it to grayscale
        cv::Mat image;
        resize(original, image, cv::Size(original.size().width * 600 / original.size().height, 600));
        cv::Mat gray;
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // smooth the image using a 3x3 Gaussian, then apply the blackhat
        // morphological operator to find dark regions on a light background
        GaussianBlur(gray, gray, cv::Size(3, 3), 0);
        cv::Mat blackhat;
        morphologyEx(gray, blackhat, cv::MORPH_BLACKHAT, rectKernel);

    #ifdef DISPLAY_INTERMEDIATE_IMAGES
        display_image("Blackhat", blackhat);
    #endif /* DISPLAY_INTERMEDIATE_IMAGES */

        // compute the Scharr gradient of the blackhat image and scale the
        // result into the range [0, 255]
        cv::Mat gradX;
        Sobel(blackhat, gradX, CV_32F, 1, 0, -1);
        gradX = abs(gradX);
        double minVal, maxVal;
        minMaxIdx(gradX, &minVal, &maxVal);
        cv::Mat gradXfloat = (255 * ((gradX - minVal) / (maxVal - minVal)));
        gradXfloat.convertTo(gradX, CV_8UC1);

    #ifdef DISPLAY_INTERMEDIATE_IMAGES
        display_image("Gx", gradX);
    #endif /* DISPLAY_INTERMEDIATE_IMAGES */

        // apply a closing operation using the rectangular kernel to close
        // gaps in between letters -- then apply Otsu's thresholding method
        morphologyEx(gradX, gradX, cv::MORPH_CLOSE, rectKernel);
        cv::Mat thresh;
        threshold(gradX, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    #ifdef DISPLAY_INTERMEDIATE_IMAGES
        display_image("Horizontal closing", thresh);
    #endif /* DISPLAY_INTERMEDIATE_IMAGES */

        // perform another closing operation, this time using the square
        // kernel to close gaps between lines of the MRZ, then perform a
        // series of erosions to break apart connected components
        morphologyEx(thresh, thresh, cv::MORPH_CLOSE, sqKernel);
        cv::Mat nullKernel;
        erode(thresh, thresh, nullKernel, cv::Point(-1, -1), 4);

    #ifdef DISPLAY_INTERMEDIATE_IMAGES
        display_image("Vertical closing", thresh);
    #endif /* DISPLAY_INTERMEDIATE_IMAGES */

        // during thresholding, it's possible that border pixels were
        // included in the thresholding, so let's set 5% of the left and
        // right borders to zero
        double p = image.size().height * 0.05;
        thresh = thresh(cv::Rect(p, p, image.size().width - 2 * p, image.size().height - 2 * p));

    #ifdef DISPLAY_INTERMEDIATE_IMAGES
        display_image("Border removal", thresh);
    #endif /* DISPLAY_INTERMEDIATE_IMAGES */

        // find contours in the thresholded image and sort them by their
        // size
        std::vector<std::vector<cv::Point> > contours;
        findContours(thresh, contours, cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE);
        // Sort the contours in decreasing area
        sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2){
            return contourArea(c1, false) > contourArea(c2, false);
        });

        // Find the first contour with the right aspect ratio and a large width relative to the width of the image
        cv::Rect roiRect(0, 0, 0, 0);
        std::vector<std::vector<cv::Point> >::iterator border_iter
            = find_if(contours.begin(), contours.end(), [&roiRect, gray](std::vector<cv::Point> &contour) {
            // compute the bounding box of the contour and use the contour to
            // compute the aspect ratio and coverage ratio of the bounding box
            // width to the width of the image
            roiRect = boundingRect(contour);
            // dump_rect("Bounding rect", roiRect);
            // pprint([x, y, w, h])
            double aspect = (double) roiRect.size().width / (double) roiRect.size().height;
            double coverageWidth = (double) roiRect.size().width / (double) gray.size().height;
            // cerr << "aspect=" << aspect << "; coverageWidth=" << coverageWidth << endl;
            // check to see if the aspect ratio and coverage width are within
            // acceptable criteria
            if (aspect > 5 and coverageWidth > 0.5) {
                return true;
            }
            return false;
        });

        if (border_iter == contours.end()) {
            return false;
        }

        // Correct ROI for border removal offset
        roiRect += cv::Point(p, p);
        // pad the bounding box since we applied erosions and now need
        // to re-grow it
        int pX = (roiRect.x + roiRect.size().width) * 0.03;
        int pY = (roiRect.y + roiRect.size().height) * 0.03;
        roiRect -= cv::Point(pX, pY);
        roiRect += cv::Size(pX * 2, pY * 2);
        // Ensure ROI is within image
        roiRect &= cv::Rect(0, 0, image.size().width, image.size().height);
        // Make it relative to original image again
        float scale = static_cast<float>(original.size().width) / static_cast<float>(image.size().width);
        roiRect.x *= scale;
        roiRect.y *= scale;
        roiRect.width *= scale;
        roiRect.height *= scale;
        mrz = original(roiRect);

    #if 0 || defined(DISPLAY_INTERMEDIATE_IMAGES)
        // Draw a bounding box surrounding the MRZ
        Mat display_roi = original.clone();
        rectangle(display_roi, roiRect, Scalar(0, 255, 0), 2);
        display_image("MRZ detection results", display_roi);
    #endif /* DISPLAY_INTERMEDIATE_IMAGES */
        cv::Mat display_roi = original.clone();
        rectangle(display_roi, roiRect, cv::Scalar(0, 255, 0), 2);
        pyrDown( display_roi, display_roi, cv::Size( display_roi.cols/2, display_roi.rows/2 ) );
        imshow("roi", display_roi);
        imshow("output", mrz);

        cv::waitKey(0);
    }

    return EXIT_SUCCESS;
}
