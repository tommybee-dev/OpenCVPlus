#include <iostream>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <log4cpp/Category.hh>
#include <log4cpp/Appender.hh>
#include <log4cpp/FileAppender.hh>
#include <log4cpp/OstreamAppender.hh>
#include <log4cpp/Layout.hh>
#include <log4cpp/PatternLayout.hh>
#include <log4cpp/SimpleLayout.hh>
#include <log4cpp/Priority.hh>

#include "debug.h"
#include "ocr.h"
#include "mrz.h"
#include "RecogniserKNearest.h"
#include "RecogniserAbsDiff.h"
#include "recogniser_tesseract.h"

#include "find_mrz.h"
#include "find_borders.h"
#include "getcwd.h"
//#include "file_capture.h"
#include "sliding_window_capture.h"

using namespace std;
using namespace cv;
using namespace ocr;


//#define USE_TESSERACT

#define CHAR_SIZE_TOLERANCE 0.15
#define MRZ_LINE_SPACING 1.0
//#define TRAINING_DATA_FILENAME "training.data"

//#if 1
//#define DISPLAY_INTERMEDIATE_IMAGES
//#endif

extern void assign_indeterminate(vector<Rect> &indeterminate, vector<vector<Rect> > &lines);

static void calc_char_cell(const Size &mrz_size, Size &char_min, Size &char_max,
		MRZ *mrz = NULL)
{
	unsigned int min_lines, max_lines;
	unsigned int min_chars_per_line, max_chars_per_line;
	if (NULL == mrz) {
		min_chars_per_line = MRZ::getMinCharsPerLine();
		max_chars_per_line = MRZ::getMaxCharsPerLine();
		min_lines = MRZ::getMinLineCount();
		max_lines = MRZ::getMaxLineCount();
	} else {
		min_chars_per_line = mrz->getCharsPerLine();
		max_chars_per_line = min_chars_per_line;
		min_lines = mrz->getLineCount();
		max_lines = min_lines;
	}
	// Account for inter-line spacing
	min_lines *= MRZ_LINE_SPACING + 1;
	min_lines -= 1;
	max_lines *= MRZ_LINE_SPACING + 1;
	max_lines -= 1;
	char_min = Size((double) mrz_size.width / (double) max_chars_per_line,
			(double) mrz_size.height / (double) max_lines);
	char_max = Size((double) mrz_size.width / (double) min_chars_per_line,
			(double) mrz_size.height / (double) min_lines);
	// Add a tolerance
	char_min.width /= (1 + CHAR_SIZE_TOLERANCE);
	char_min.height /= (1 + CHAR_SIZE_TOLERANCE);
	char_max.width *= (1 + CHAR_SIZE_TOLERANCE);
	char_max.height *= (1 + CHAR_SIZE_TOLERANCE);
#if 0
	cerr << "Char min rect: " << char_min << endl;
	cerr << "Char max rect: " << char_max << endl;
#endif
	// Additional tuning for minimum width.
	// Although OCR B is monospaced, some character glyphs are much narrower than others.
	char_min.width *= 0.25;
	// Additional tuning for minimum height.
	// Line spacing varies widely.
	char_min.height *= 0.75;
}

static bool is_character(const Rect boundingRect, const Size &minSize,
		const Size &maxSize)
{
	return boundingRect.width >= minSize.width
			&& boundingRect.height >= minSize.height
			&& boundingRect.width <= maxSize.width
			&& boundingRect.height <= maxSize.height;
}

static void find_character_bboxes(const Mat &image, const Rect &borders, vector<Rect> &char_bboxes,
		MRZ *mrz = NULL)
{
	vector<vector<Point> > contours;
	Mat work = image.clone();
	findContours(work, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, borders.tl());
	Size char_min, char_max;
	calc_char_cell(image.size(), char_min, char_max, mrz);
	for_each(
		contours.begin(), contours.end(),
		[&char_bboxes, char_min, char_max, &work]
		(vector<Point> &contour) {
			Rect br = boundingRect(contour);
			if (is_character(br, char_min, char_max)) {
				// dump_rect("Character", br);
#if 0 && defined(DISPLAY_INTERMEDIATE_IMAGES)
				drawContours(work, contour, -1, Scalar(1, 1, 1));
#endif
				char_bboxes.push_back(br);
			} else {
				// Not the right size
				dump_rect("Rejected char", br);
			}
		}
	);
#if 0 && defined(DISPLAY_INTERMEDIATE_IMAGES)
	display_image("Character contours", work);
#endif
}

static void assign_to_lines(const Size &image_size,
		const vector<Rect> &char_bboxes, vector<vector<Rect> > &lines,
		vector<Rect> &indeterminate)
{
	for_each(char_bboxes.begin(), char_bboxes.end(),
			[image_size, &lines, &indeterminate](const Rect &bbox) {
				unsigned int line_num;
				bool found = false;
				for (line_num = 0; line_num < lines.size(); line_num++) {
					int top = image_size.height * line_num / lines.size();
					int middle = image_size.height * (line_num + 0.5) / lines.size();
					int bottom = image_size.height * (line_num + 1) / lines.size();
					if (
							bbox.y >= top && bbox.y + bbox.height <= bottom &&
							abs((bbox.y + bbox.y + bbox.height) / 2 - middle) < abs(image_size.height / (3 * lines.size()))
					) {
						found = true;
						break;
					}
				}
				if (found) {
					lines[line_num].push_back(bbox);
				} else {
					indeterminate.push_back(bbox);
				}
			});
}

static float confidence_type(const Size &image_size,
		vector<vector<Rect> > &lines, unsigned int chars_per_line,
		const vector<Rect> &char_bboxes, vector<Rect> &indeterminate)
{
	if (char_bboxes.size() < lines.size() * chars_per_line / 2) {
		return 0; // Less than 50% characters recognised
	}
	assign_to_lines(image_size, char_bboxes, lines, indeterminate);
	if (indeterminate.size() > char_bboxes.size() / 5) {
		return 0; // More than 20% of characters not aligned
	}
	unsigned int num_aligned = 0;
	for (unsigned int line_num = 0; line_num < lines.size(); line_num++) {
		if (lines[line_num].size() > chars_per_line) {
			return 0; // Line too long
		}
		num_aligned += lines[line_num].size();
	}
	return static_cast<float>(num_aligned)
			/ static_cast<float>(lines.size() * chars_per_line);
}

static void fixup_missing_chars(const Mat &image, vector<vector<Rect> > &lines,
		MRZ &mrz)
{
	unsigned int num_expected = mrz.getLineCount() * mrz.getCharsPerLine();
	unsigned int num_found = 0;
	for_each(lines.begin(), lines.end(), [&num_found](vector<Rect> &line) {
		// cerr << "Chars this line: " << line.size() << endl;
		num_found += line.size();
	});
	if (num_found < num_expected) {
		cerr << "Only found " << num_found << " of " << num_expected << " chars. Interpolating." << endl;
	}
	unsigned int expected_width = image.cols / mrz.getCharsPerLine();
	unsigned int expected_height = image.rows / mrz.getLineCount();
	unsigned int expected_x = 0;
	MRZ *pmrz = &mrz;
	for_each(lines.begin(), lines.end(), [&num_found, lines, pmrz](vector<Rect> &line) {
		if (lines.size() != pmrz->getCharsPerLine()) {
			// unsigned int expected_y = TODO;
			for (vector<Rect>::size_type i = 0; i < line.size(); i++) {
				;
			}
		}
	});
}

static void sort_lines(vector<vector<Rect> > &lines)
{
	for_each(lines.begin(), lines.end(), [](vector<Rect> &line) {
		sort(line.begin(), line.end(), [](const Rect &r1, const Rect &r2) {
			return r1.x < r2.x;
		});
	});
}

static bool find_chars(
	const Mat &image,
	vector<vector<Rect> > &lines
)
{
	Rect borders = find_borders(image);
	Mat cropped = image(borders);
	cropped = 255 - cropped;

	vector<Rect> bboxes;
	find_character_bboxes(cropped, borders, bboxes);

	MRZ *chosen_mrz = NULL;
	MRZType1 mrz1;
	MRZType2 mrz2;
	MRZType3 mrz3;
	vector<Rect> indeterminate_type_1;
	vector<vector<Rect> > lines_type_1(
			mrz1.getLineCount());
	float conf_type_1 = confidence_type(cropped.size(), lines_type_1, mrz1.getCharsPerLine(), bboxes,
			indeterminate_type_1);
	vector<Rect> indeterminate_type_2;
	vector<vector<Rect> > lines_type_2(
			mrz2.getLineCount());
	float conf_type_2 = confidence_type(cropped.size(), lines_type_2, mrz2.getCharsPerLine(), bboxes,
			indeterminate_type_2);
	vector<Rect> indeterminate_type_3;
	vector<vector<Rect> > lines_type_3(
			mrz3.getLineCount());
	float conf_type_3 = confidence_type(cropped.size(), lines_type_3, mrz3.getCharsPerLine(), bboxes,
			indeterminate_type_3);
	if (conf_type_1 > max({ conf_type_2, conf_type_3, 0.75f })) {
		cerr << "Looks like type 1" << endl;
		chosen_mrz = &mrz1;
		lines = lines_type_1;
		assign_indeterminate(indeterminate_type_1, lines);
	} else if (conf_type_2 > max({ conf_type_1, conf_type_3, 0.75f })) {
		cerr << "Looks like type 2" << endl;
		chosen_mrz = &mrz2;
		lines = lines_type_2;
		assign_indeterminate(indeterminate_type_2, lines);
	} else if (conf_type_3 > max({ conf_type_1, conf_type_2, 0.75f })) {
		cerr << "Looks like type 3" << endl;
		chosen_mrz = &mrz3;
		lines = lines_type_3;
		assign_indeterminate(indeterminate_type_3, lines);
	} else {
		cerr << "Indeterminate type: " << conf_type_1 << " confidence Type 1, "
				<< conf_type_2 << " confidence Type 2, "
				<< conf_type_3 << " confidence Type 3" << endl;
	}

	if (NULL == chosen_mrz) {
		return false;
	}

	sort_lines(lines);
	fixup_missing_chars(cropped, lines, *chosen_mrz);

	return true;
}

static bool find_chars_dbg(
	const Mat &image,
	Mat &draw_image,
	vector<vector<Rect> > &lines
)
{
	assert(image.size() == draw_image.size());
	Rect borders = find_borders(image);
	Mat cropped = image(borders);
	cropped = 255 - cropped;

	display_image("Inverted cropped ROI", cropped);

	vector<Rect> bboxes;
	find_character_bboxes(cropped, borders, bboxes);

	for_each(bboxes.begin(), bboxes.end(), [&draw_image, borders](const Rect &bbox) {
		rectangle(draw_image, bbox, Scalar(0, 0, 255));
	});
	display_image("Char bboxes", draw_image);

	MRZ *chosen_mrz = NULL;
	MRZType1 mrz1;
	MRZType2 mrz2;
	MRZType3 mrz3;
	vector<Rect> indeterminate_type_1;
	vector<vector<Rect> > lines_type_1(
			mrz1.getLineCount());
	float conf_type_1 = confidence_type(cropped.size(), lines_type_1, mrz1.getCharsPerLine(), bboxes,
			indeterminate_type_1);
	vector<Rect> indeterminate_type_2;
	vector<vector<Rect> > lines_type_2(
			mrz2.getLineCount());
	float conf_type_2 = confidence_type(cropped.size(), lines_type_2, mrz2.getCharsPerLine(), bboxes,
			indeterminate_type_2);
	vector<Rect> indeterminate_type_3;
	vector<vector<Rect> > lines_type_3(
			mrz3.getLineCount());
	float conf_type_3 = confidence_type(cropped.size(), lines_type_3, mrz3.getCharsPerLine(), bboxes,
			indeterminate_type_3);
	if (conf_type_1 > max({ conf_type_2, conf_type_3, 0.75f })) {
		cerr << "Looks like type 1" << endl;
		chosen_mrz = &mrz1;
		lines = lines_type_1;
		assign_indeterminate(indeterminate_type_1, lines);
	} else if (conf_type_2 > max({ conf_type_1, conf_type_3, 0.75f })) {
		cerr << "Looks like type 2" << endl;
		chosen_mrz = &mrz2;
		lines = lines_type_2;
		assign_indeterminate(indeterminate_type_2, lines);
	} else if (conf_type_3 > max({ conf_type_1, conf_type_2, 0.75f })) {
		cerr << "Looks like type 3" << endl;
		chosen_mrz = &mrz3;
		lines = lines_type_3;
		assign_indeterminate(indeterminate_type_3, lines);
	} else {
		cerr << "Indeterminate type: " << conf_type_1 << " confidence Type 1, "
				<< conf_type_2 << " confidence Type 2, "
				<< conf_type_3 << " confidence Type 3" << endl;
	}

	if (NULL == chosen_mrz) {
		return false;
	}

	sort_lines(lines);
	fixup_missing_chars(cropped, lines, *chosen_mrz);

	return true;
}

static void process_diff(Mat &original, bool is_dbg)
{
	Mat roiImage;
	if (!find_mrz(original, roiImage)) {
		cerr << "No MRZ found" << endl;
		return;
	}

    if(is_dbg) display_image("ROI", roiImage);

	Mat roi_grey;
	cvtColor(roiImage, roi_grey, COLOR_BGR2GRAY);

	Mat roi_thresh;
	threshold(roi_grey, roi_thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

	if(is_dbg) display_image("ROI threshold", roi_thresh);

	vector<vector<Rect> > lines;
    if(is_dbg)
    {
        Mat drawImage = roiImage.clone();

        if (find_chars_dbg( roi_thresh, drawImage, lines )) {
            string text;
            RecogniserAbsDiff recogniser(MRZ::charset, "OCRB");
            recogniser.recognise(roi_grey, lines, text);
            cerr << "Recognised text: " << text << endl;
        }

    }
    else
    {
         if (find_chars( roi_thresh, lines )) {
            string text;
            RecogniserAbsDiff recogniser(MRZ::charset, "OCRB");
            recogniser.recognise(roi_grey, lines, text);
            cerr << "Recognised text: " << text << endl;
        }
    }

}

static int process_train_k_near(string train_img, string train_data_file)
{
	// cout << getBuildInformation() << endl;
	//Mat img = imread("ocrb.png");
	Mat img = imread(train_img);

	SlidingWindowCapture image_source(img, Size(70, 115), Point(70 + 2));
	RecogniserKNearest::learnOcr(image_source, MRZ::charset,
			train_data_file.c_str());

	return EXIT_SUCCESS;
}

static void process_k_near(Mat &original, string train_data_file, bool is_dbg)
{
	Mat roiImage;
	if (!find_mrz(original, roiImage)) {
		cerr << "No MRZ found" << endl;
		return;
	}

    if(is_dbg) display_image("ROI", roiImage);

	Mat roi_grey;
	cvtColor(roiImage, roi_grey, COLOR_BGR2GRAY);

	Mat roi_thresh;
	threshold(roi_grey, roi_thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

	if(is_dbg) display_image("ROI threshold", roi_thresh);

	vector<vector<Rect> > lines;
	if (find_chars(roi_thresh, lines)) {
		string text;
		RecogniserKNearest recogniser(train_data_file.c_str());
		recogniser.recognise(roiImage, lines, text);
		cerr << "Recognised text: " << text << endl;

		if(is_dbg) display_image("Original", original);
	}
}

static void process_tess(Mat &original, string data_dir, bool is_dbg)
{
	Mat roiImage;
	if (!find_mrz(original, roiImage)) {
		cerr << "No MRZ found" << endl;
		return;
	}

    if(is_dbg) display_image("ROI", roiImage);

	Mat roi_grey;
	cvtColor(roiImage, roi_grey, COLOR_BGR2GRAY);

	Mat roi_thresh;
	threshold(roi_grey, roi_thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
    if(is_dbg) display_image("ROI threshold", roi_thresh);


	vector<uchar> buf;
	imencode(".bmp", roi_thresh, buf);

	cerr << "data dir: " << data_dir << endl;
	RecogniserTesseract tess("eng", &data_dir[0], MRZ::charset.c_str());

	tess.set_image_bmp(&buf[0]);
	tess.ocr();

}

const cv::String keys =
        "{@image    |<none>| input ocr image to detect}"
        "{method    |diff| tess - Use a tesseract \r\n"
        "               k_near - Use a K-Nearest \r\n"
        "               diff - Use a Absolute Difference  | algorithm to use}"
        "{data_dir  | | | data directory for tesseract}"
        "{is_train  | | | training phase}"
        "{train_img | | | an image file name for training data when is_train is on}"
        "{train_file| | | a name of the trained dataset when is_train is on}"
        "{dbg       | | | debug flag}"
        "{help      |      | show help message}";      // optional, show help optional
//--image examples/passport_01.jpg --data_dir=C:\DEV\workspaces_opencv\passport_mrz_detector_cpp\tessdata --method=k_near --is_train --train_img=ocrb.png --train_file=ocr_train
int main(int argc, char *argv[])
{
	log4cpp::Appender *consoleAppender
        = new log4cpp::OstreamAppender("console", &std::cerr);
	log4cpp::Category& root = log4cpp::Category::getRoot();
	root.setPriority(log4cpp::Priority::getPriorityValue("DEBUG"));
	consoleAppender->setLayout(new log4cpp::SimpleLayout());
	root.addAppender(consoleAppender);


    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("usage: --type tess ");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    cv::String image = parser.get<cv::String>(0);
    cv::String method = parser.get<cv::String>("method");
    cv::String data_dir = parser.get<cv::String>("data_dir");

    if (!parser.check()) {
        std::cerr << "printErrors: " << '\n';
        parser.printErrors();
        return -1;
    }
    Mat input = imread(image);
    bool is_debug = parser.has("dbg");

    if(method.compare("diff") == 0)
    {
        process_diff(input, is_debug);
    }
    else if(method.compare("tess") == 0)
    {
        process_tess(input, data_dir, is_debug);
    }else if(method.compare("k_near") == 0)
    {
        bool is_train = parser.has("is_train");
        cv::String train_file = parser.get<cv::String>("train_file");
        if(is_train)
        {
            cv::String train_img = parser.get<cv::String>("train_img");

            process_train_k_near(train_img, train_file);
        }
        else
            process_k_near(input, train_file, is_debug);
    }
    else
    {
        std::cerr << "No method found: " << '\n';
        return EXIT_FAILURE;

    }

   std::cout << "method:(" << method << ")" << '\n';

   return EXIT_SUCCESS;

}
