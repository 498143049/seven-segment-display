#if !defined WATERSHS
#define WATERSHS

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class WatershedSegmenter {

private:

    cv::Mat markers;

public:

    void setMarkers(const cv::Mat& markerImage) {

        // Convert to image of ints
        markerImage.convertTo(markers,CV_32S);
    }

    cv::Mat process(const cv::Mat &image) {

        // Apply watershed
        cv::watershed(image,markers);

        return markers;
    }

    // Return result in the form of an image
    cv::Mat getSegmentation() {

        cv::Mat tmp;
        // all segment with label higher than 255
        // will be assigned value 255
        markers.convertTo(tmp,CV_8U);

        return tmp;
    }

    // Return watershed in the form of an image以图像的形式返回分水岭
    cv::Mat getWatersheds() {

        cv::Mat tmp;
        //在变换前，把每个像素p转换为255p+255（在conertTo中实现）
        markers.convertTo(tmp,CV_8U,255,255);

        return tmp;
    }
};
#endif