//
//  main.cpp
//  OpenCv
//
//  Created by Shokhina Badrieva on 3/5/21.
//  Copyright © 2021 Shokhina Badrieva. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include "MeanFilteringClass.hpp"
#include "NoiseClass.hpp"
#include "QualityMetricsClass.hpp"
#include "OrderStatisticFiltersClass.hpp"
#include "SharpeningFiltersClass.hpp"
#include "BitPlaneSlicingClass.hpp"
#include "BitPlaneHw.hpp"
#include "AlgebraicOperationsClass.hpp"
#include "EnhancementClass.hpp"
#include "ThresholdingClass.hpp"
#include "EdgeDetectionClass.hpp"
#include "MorphologicalFilterClass.hpp"



using namespace std;
using namespace cv;

//Three road detection algorithms are investigated in this main file. The best algorithm, according to our experiments is finalAlgorithm function. It can be tested by itself by passing the source image to the function in main.

void finalAlgorithm2(Mat &src,Mat&dst){
    
    
    Mat gammaImg,logImg,filteredImg,edgeImg,gaussianImg ;
    
    //filters
    gammaCorrect(src, gammaImg, 1.8);
    
    logTransform(gammaImg, logImg);
    
    multiStageMedianFilter(logImg, filteredImg, 3);
    
    //morphological opening
    int morph_size = 2;
    Mat element = getStructuringElement(
                                        MORPH_RECT,
                                        Size(2 * morph_size + 1,
                                             2 * morph_size + 1),
                                        Point(morph_size,
                                              morph_size));
    
    morphologyEx(filteredImg, filteredImg,
                 MORPH_OPEN, element,
                 Point(-1, -1), 2);
    
    
    //edge detection
    morphEdgeDetector(filteredImg, dst);
    
}

void finalAlgorithm1(Mat&src,Mat &dst){
    
    Mat zeroCrossing,newImg,filteredImg,edgeImg,gammaImg,logImg;
    
    //filters
    zeroCrossingEdgeDetection(src, zeroCrossing);
    
    Laplacian(src, zeroCrossing, -1);
    
    newImg=src+(src-zeroCrossing);
    
    arithMeanFilter(newImg, filteredImg, 3);

    multiStageMedianFilter(filteredImg, filteredImg, 3);
    
    gammaCorrect(filteredImg, gammaImg, 1.8);
    
    logTransform(gammaImg, logImg);
    
    //morphological opening
    int morph_size = 2;
    Mat element = getStructuringElement(
                                        MORPH_ELLIPSE,
                                        Size(2 * morph_size + 1,
                                             2 * morph_size + 1),
                                        Point(morph_size,
                                              morph_size));
    
    morphologyEx(logImg, filteredImg,
                 MORPH_OPEN, element,
                 Point(-1, -1), 2);
    
    
    //edge detection
    morphEdgeDetector(filteredImg, dst);

}



void compareSystems(Mat&src){
    imshow("Original Image: ",src);
    Mat dst,dst2,dst3;
    finalAlgorithm1(src,dst);
    finalAlgorithm2(src,dst2);
    imshow("Algorithm 1:",dst);
    imshow("Algorithm 2:",dst2);
    addWeighted(dst, 0.5, dst2, 0.5, 0.0, dst3);
    imshow("Alpha Blended",dst3);
    double thresh=0;
    double maxValue=255;
    threshold(dst3,dst3,thresh,maxValue,THRESH_OTSU);
    imshow("Alpha Blended Otsu",dst3);
    double ssim1=mySsim(src, dst);
    cout<<"Ssim of original compared to algorithm 1:  "<<ssim1<<endl;
    double ssim2=mySsim(src, dst2);
    cout<<"Ssim of original compared to algorithm 2:  "<<ssim2<<endl;
    double ssim3=mySsim(src, dst3);
    cout<<"Ssim of original compared to alpha blended image 3:  "<<ssim3<<endl;

    waitKey(0);
    destroyAllWindows();
}

int main(){
    //replace with image as well path
    Mat roadImg=imread("/Users/sbadrieva/roadImages/roadImg1.jpg",0);

    compareSystems(roadImg);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
