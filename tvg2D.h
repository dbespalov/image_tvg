/******************************************************************************************/
/*                                                                                        */
/* Texture descriptors for images based on geometric total variation energy density       */
/*                                                                                        */
/* Copyright (C) 2010  Drexel University                                                  */
/* Author: Dmitriy Bespalov (bespalov@gmail.com)                                          */
/*                                                                                        */
/*     This file is part of tvg_texture.                                                  */
/*     tvg_texture is free software: you can redistribute it and/or modify                */
/*     it under the terms of the GNU General Public License as published by               */
/*     the Free Software Foundation, either version 3 of the License, or                  */
/*     (at your option) any later version.                                                */
/*                                                                                        */
/*     tvg_texture is distributed in the hope that it will be useful,                     */
/*     but WITHOUT ANY WARRANTY; without even the implied warranty of                     */
/*     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                      */
/*     GNU General Public License for more details.                                       */
/*                                                                                        */
/*     You should have received a copy of the GNU General Public License                  */
/*     along with tvg_texture.  If not, see <http://www.gnu.org/licenses/>.               */
/*                                                                                        */
/* ************************************************************************************** */

#ifndef __TVG2D_H__
#define __TVG2D_H__

#include <stdlib.h>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <queue>

#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <math.h>

#include <include.h>

#include <siftfast/siftfast.h>


#include <newmat.h>
#include <newmatap.h>
#include <newmatnl.h>

/****** PARAMETERS BEGIN ******/
static const double PI = 3.14159265358979323846;

static const double LAMBDA = 1; // regularization parameter
static const double ALPHA = 0.5; // structure sensitive parameter

static const double GRADIENT_SMOOTH_PAR = 2.4; // "sigma" smoothing parameter for regression-based gradient estimation

static const int STEEPEST_DESCEND_WINDOW_RADIUS = 3; // radius of the window for estimating steerable parameters
static const int GRADIENT_WINDOW_RADIUS = 3;  // radius of the window for regression -based gradient estimation

static const double TVG_SMOOTH_PAR = 3.6; // smoothing parameter for weighting gradients when computing TV Geometry energies
static const int TVG_WINDOW_RADIUS = 3;    // radius of the window for TV Geometry estimation

static const int BORDER_RADIUS = 5; // extension of the border using replication method

const static int SIFT_FEATURE_DIMS = 128;

const static int NUM_TVG_FEATURES_PER_SUBWIN=6;
const static int TVG_WINDOW_DIV_FACTOR = 4;
const static int TVG_SUBWIN_SAMPLES = 10;

// used to convert interest point scale into radius of the interest point (in pixels)
const static double SIFT_WINDOW_SCALE_FACTOR = 1.6;  

static int TVG_FEATURE_DIMS = NUM_TVG_FEATURES_PER_SUBWIN   *   TVG_WINDOW_DIV_FACTOR   *   TVG_WINDOW_DIV_FACTOR;

static vector<ColumnVector*> TMP_TVG;
/****** PARAMETERS END ******/


class MyKeypoint{

 private:
  MyKeypoint(){}

 public:
    
  MyKeypoint(string descriptor_name, int desc_dim){
    
    fv = new ColumnVector(desc_dim);
    
    row = -1;
    col = -1;
    fradius = -1; // extend (radius) of the feature 
    ori = -1;
    selected = false;
    dname = descriptor_name;
  }
  
  MyKeypoint(string descriptor_name, int desc_dim, MyKeypoint * keypoint){
    
    fv = new ColumnVector(desc_dim);
      
    row = keypoint->row;
    col = keypoint->col;
    fradius = keypoint->fradius;
    ori = keypoint->ori;
    selected = keypoint->selected;
    
    dname = descriptor_name;
  }
  
  
  ~MyKeypoint(){
    delete fv;
  }
  
  ColumnVector * fv;
  float row, col, fradius, ori; // ori \in [-PI, PI]
  bool selected;
  string dname; // descriptor name
};


// Computes SIFT descriptors for interest points in img using libsiftfast 
// (exact C++ implementation of Lowe's sift program by zerofrog@gmail.com)
// input: img -- input  image
// output: key_points -- detected interest points and their descriptors
void computeSiftFeaturesFromIplImage(IplImage* img, vector<MyKeypoint*> & key_points);

// Computes texture descriptors based on geometric total variation (TVG) energy density of degree 1 and 2
// input: keypoints -- interest points (detected using libsiftfast)
//        o1, o2    -- tvg energy of degree 1 and 2
// output: tvg_keypoints   --  tvg descriptors for interest points
void computeTvgDescriptorsForFeatures(vector<MyKeypoint*> & keypoints, vector<MyKeypoint*> & tvg_keypoints, IplImage * o1, IplImage * o2);


// main function to compute geometric total variation (TVG) energy density of degree 1 and 2
// Regression with local steering kernels is used to estimate gradients dx and dy 
// input: gray_img -- grayscale image
// output: order1, order2 -- TVG Energy, degree 1 and 2
void getTvg2D( IplImage * gray_img, IplImage * order1, IplImage * order2 );


// main function to compute geometric total variation (TVG) energy density of degree 1 and 2
// Sobel filters are used to estimate gradients dx and dy 
// (i.e., getFastTvg2D does not use regression with local steering kernels)
// input: gray_img -- grayscale image
// output: order1, order2 -- TV Geometry, order 1 and 2
void getFastTvg2D( IplImage * gray_img, IplImage * order1, IplImage * order2 );


// computes geometric total variation (TVG) energy density of degree 1 and 2
// inputs: dx, dy -- derivatives
//         all_Cs -- steering covariance matrix, filled in by estimateSteerCovarianceMatrix() function
// outputs: order1, order2 -- TVG energy, degree 1 and 2
void computeTvgEnergy2D(IplImage * dx, IplImage * dy, IplImage * order1, IplImage * order2, vector<IplImage*> & all_Cs);


// performs regression to estimate gradients
// inputs: chan -- input intensity image
//         all_Cs -- steering covariance matrix, filled in by estimateSteerCovarianceMatrix() function
// outputs: dx, dy -- derivative images
//          new_chan -- new estimates of intensities, can be used for steer based smoothing, otherwise, just disregard
void estimateGradientsWithRegression(IplImage * chan, vector<IplImage*> & all_Cs, IplImage * dx, IplImage * dy, IplImage * new_chan);


// estimates covariance matrix for every pixel, 
// inputs:  dx, dy -- image derivatives
// outputs: all_Cs -- steering covariance matrix, filled 
void estimateSteerCovarianceMatrix(IplImage * dx, IplImage * dy, vector<IplImage*> & all_Cs);







#endif
