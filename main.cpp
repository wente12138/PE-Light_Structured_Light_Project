#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

// 法向质心法
vector<Point2f> GetLinePoints_NormalCentroidMethod(Mat& src, int Min, int Max, float grayThresh, float minBrightness, int range = 10, int max_iter = 10, float thre_distance = 0.05) {
    Mat srcGray;
    cvtColor(src, srcGray, COLOR_BGR2GRAY);

    // 初始中心点的垂直投影
    Mat sumCols;
    reduce(srcGray, sumCols, 0, REDUCE_SUM, CV_32F);

    vector<Point2f> pts;
    for (int i = Min; i < Max; i++) {
        // 检查该列是否有足够的亮度
        if (sumCols.at<float>(i) < minBrightness) continue;

        double minValCol, maxValCol; // 用来存储每一列的最大值和最小值
        Point minLocCol, maxLocCol; // 用来存储每一列的最大值和最小值的位置
        minMaxLoc(srcGray.col(i), &minValCol, &maxValCol, &minLocCol, &maxLocCol);
        pts.push_back(Point2f(i, maxLocCol.y)); // 此处注意，虽然minMaxLoc函数返回的maxLocCol是Point类型，但是其中的x是没有意义的，因为输入的是一列矩阵，所以x是0，想要获取最大值点的行号，需要外围的循环变量i
    }

    vector<Point2f> pts_new = pts;
    int iter = 0;
    float distance;

    do {
        distance = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            float sumX = 0, sumY = 0, sum = 0;
            for (int j = -range; j <= range; ++j) {
                int newX = cvRound(pts[i].x);
                int newY = cvRound(pts[i].y + j);
                if (newX >= 0 && newX < srcGray.cols && newY >= 0 && newY < srcGray.rows) {
                    float pixelValue = srcGray.at<uchar>(newY, newX); //注意这里的坐标是反的，因为从图像矩阵中取值是先行后列，而图像的坐标是先列后行
                    if (pixelValue > grayThresh) {
                        sum += pixelValue;
                        sumX += pixelValue * newX;
                        sumY += pixelValue * newY;
                    }
                }
            }
            if (sum > 0) {
                pts_new[i].x = sumX / sum;
                pts_new[i].y = sumY / sum;
            } else {
                pts_new[i] = pts[i]; // 如果亮度不足，保持位置不变
            }
            distance += pow(pts_new[i].x - pts[i].x, 2) + pow(pts_new[i].y - pts[i].y, 2);
        }
        distance = sqrt(distance / pts.size());
        pts = pts_new;
        iter++;
    } while (iter < max_iter && distance > thre_distance);

    // 移除亮度不足的点
    pts.erase(remove_if(pts.begin(), pts.end(), [&](const Point2f& pt) {
        return srcGray.at<uchar>(cvRound(pt.y), cvRound(pt.x)) < minBrightness;
    }), pts.end());

    return pts;
}

// 极值检测法
vector<Point2f> GetLinePoints_PeakDetection(Mat& src, int Min, int Max, int windowSize, int grayThresh) {
    Mat srcGray;
    cvtColor(src, srcGray, COLOR_BGR2GRAY);

    // 对图像进行高斯模糊处理，减少噪声
    GaussianBlur(srcGray, srcGray, Size(5, 5), 1.5);

    vector<Point2f> points_vec;

    for (int col = Min; col < Max; col++) {
        uchar maxVal = 0;
        int maxRow = -1;
        for (int row = windowSize; row < srcGray.rows - windowSize; row++) {
            uchar currentVal = srcGray.at<uchar>(row, col);
            if (currentVal > maxVal && currentVal > grayThresh) {
                bool isPeak = true;
                for (int k = -windowSize; k <= windowSize; k++) {
                    if (currentVal < srcGray.at<uchar>(row + k, col)) {
                        isPeak = false;
                        break;
                    }
                }
                if (isPeak) {
                    maxVal = currentVal;
                    maxRow = row;
                }
            }
        }
        if (maxRow != -1) {
            points_vec.emplace_back(Point2f(col, maxRow));
        }
    }

    return points_vec;
}

// 灰度重心法
vector<Point2f> GetLinePoints_GravityCenter(Mat& src, int gray_Thed, int Min, int Max, int Type) {
    Mat srcGray;
    cvtColor(src, srcGray, COLOR_BGR2GRAY);
    Mat binary;
    threshold(srcGray, binary, gray_Thed, 255, THRESH_BINARY);
    vector<Point2f> points_vec;
    for (int col = Min; col < Max; col++) {
        long sumGray = 0, sumPosY = 0;
        for (int row = 0; row < srcGray.rows; row++) {
            int grayValue = binary.at<uchar>(row, col);
            if (grayValue > 0) {
                sumGray += grayValue;
                sumPosY += row * grayValue;
            }
        }
        if (sumGray > 0) {
            float centerY = sumPosY / static_cast<float>(sumGray);
            points_vec.emplace_back(Point2f(col, centerY));
        }
    }
    return points_vec;
}

// Steger法
vector<Point2f> GetLinePoints_Steger(Mat& src, int gray_Thed, int Min, int Max, int Type) {
    Mat srcGray, srcGray1;
    cvtColor(src, srcGray1, COLOR_BGR2GRAY);
    srcGray = srcGray1.clone();
    srcGray.convertTo(srcGray, CV_32FC1);
    GaussianBlur(srcGray, srcGray, Size(0, 0), 6, 6);
    Mat m1 = (Mat_<float>(1, 2) << 1, -1);
    Mat m2 = (Mat_<float>(2, 1) << 1, -1);
    Mat dx, dy;
    filter2D(srcGray, dx, CV_32FC1, m1);
    filter2D(srcGray, dy, CV_32FC1, m2);
    Mat m3 = (Mat_<float>(1, 3) << 1, -2, 1);
    Mat m4 = (Mat_<float>(3, 1) << 1, -2, 1);
    Mat m5 = (Mat_<float>(2, 2) << 1, -1, -1, 1);
    Mat dxx, dyy, dxy;
    filter2D(srcGray, dxx, CV_32FC1, m3);
    filter2D(srcGray, dyy, CV_32FC1, m4);
    filter2D(srcGray, dxy, CV_32FC1, m5);
    vector<Point2f> points_vec;
    for (int i = Min; i < Max; i++) {
        for (int j = 0; j < srcGray.rows; j++) {
            if (srcGray.at<float>(j, i) > gray_Thed) {
                Mat hessian(2, 2, CV_32FC1);
                hessian.at<float>(0, 0) = dxx.at<float>(j, i);
                hessian.at<float>(0, 1) = dxy.at<float>(j, i);
                hessian.at<float>(1, 0) = dxy.at<float>(j, i);
                hessian.at<float>(1, 1) = dyy.at<float>(j, i);
                Mat eValue, eVectors;
                eigen(hessian, eValue, eVectors);
                double nx = eVectors.at<float>(0, 0);
                double ny = eVectors.at<float>(0, 1);
                double t = -(nx*dx.at<float>(j, i) + ny*dy.at<float>(j, i)) / (nx*nx*dxx.at<float>(j, i) + 2 * nx*ny*dxy.at<float>(j, i) + ny*ny*dyy.at<float>(j, i));
                if (fabs(t*nx) <= 0.5 && fabs(t*ny) <= 0.5) {
                    points_vec.emplace_back(Point2f(i, j));
                }
            }
        }
    }
    return points_vec;
}

int main() {
    Mat src = imread("E:\\Github\\GitProject\\LineLaser\\Photos\\1A.BMP", IMREAD_COLOR);
    if (src.empty()) {
        cerr << "Image not found!" << endl;
        return -1;
    }
    // 调用四种方法，获取中心点
    vector<Point2f> linePoints_Steger = GetLinePoints_Steger(src, 50, 0, src.cols, 0);
    vector<Point2f> linePoints_Gravity = GetLinePoints_GravityCenter(src, 50, 0, src.cols, 0);
    vector<Point2f> linePoints_Peak = GetLinePoints_PeakDetection(src, 0, src.cols, 1, 50);
    vector<Point2f> linePoints_NormalCentroid = GetLinePoints_NormalCentroidMethod(src, 0, src.cols, 50, 100);


    // 输出结果
    for (const auto& pt : linePoints_Steger) {
        cout << "CenterPoint_Steger: (" << pt.x << ", " << pt.y << ")" << std::endl;
    }

    for (const auto& pt : linePoints_Gravity) {
        cout << "CenterPoint_GravityCenter: (" << pt.x << ", " << pt.y << ")" << std::endl;
    }

    for (const auto& pt : linePoints_NormalCentroid) {
        cout << "CenterPoint_NormalCentroid: (" << pt.x << ", " << pt.y << ")" << std::endl;
    }

    for (const auto& pt : linePoints_Peak) {
        cout << "CenterPoint_PeakDetection: (" << pt.x << ", " << pt.y << ")" << std::endl;
    }

    Mat drawImage_Steger = src.clone();
    Mat drawImage_Gravity = src.clone();
    Mat drawImage_Peak = src.clone();
    Mat drawImage_NormalCentroid = src.clone();

    for (const auto& pt : linePoints_Steger) {
        circle(drawImage_Steger, pt, 1, Scalar(0, 0, 255), 1);
    }

    for (const auto& pt : linePoints_Gravity) {
        circle(drawImage_Gravity, pt, 1, Scalar(255, 0, 0), 1);
    }

    for (const auto& pt : linePoints_Peak) {
        circle(drawImage_Peak, pt, 1, Scalar(0, 255, 0), 1);
    }

    for (const auto& pt : linePoints_NormalCentroid) {
        circle(drawImage_NormalCentroid, pt, 1, Scalar(0, 0, 255), 2);
    }

    // 保存新图像
    imwrite("E:\\Github\\GitProject\\LineLaser\\Photos\\1A_Steger.BMP", drawImage_Steger);
    imwrite("E:\\Github\\GitProject\\LineLaser\\Photos\\1A_Gravity.BMP", drawImage_Gravity);
    imwrite("E:\\Github\\GitProject\\LineLaser\\Photos\\1A_Peak.BMP", drawImage_Peak);
    imwrite("E:\\Github\\GitProject\\LineLaser\\Photos\\1A_NormalCentroid.BMP", drawImage_NormalCentroid);


    namedWindow("Steger Method Result", WINDOW_NORMAL);
    imshow("Steger Method Result", drawImage_Steger);
    waitKey(0);

    namedWindow("Gravity Method Result", WINDOW_NORMAL);
    imshow("Gravity Method Result", drawImage_Gravity);
    waitKey(0);

    namedWindow("Normal Centroid Method Result", WINDOW_NORMAL);
    imshow("Normal Centroid Method Result", drawImage_Peak);
    waitKey(0);

    namedWindow("Peak Detection Result", WINDOW_NORMAL);
    imshow("Peak Detection Result", drawImage_NormalCentroid);
    waitKey(0);

    // 输出 Steger 算法得到的点到 CSV 文件
    ofstream csvFile("E:\\Github\\GitProject\\LineLaser\\Results\\StegerPoints_1.csv");
    if (!csvFile.is_open()) {
        cerr << "Failed to open file for writing." << endl;
        return -1;
    }

    // 写入 CSV 文件的头部
    csvFile << "X,Y\n";
    for (const auto& pt : linePoints_Steger) {
        csvFile << pt.x << "," << pt.y << "\n";
    }
    csvFile.close();

    return 0;
}
