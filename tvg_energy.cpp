/******************************************************************************************/
/*                                                                                        */
/* Texture descriptors for images based on geometric total variation energy density       */
/*                                                                                        */
/* Copyright (C) 2012  Drexel University                                                  */
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


#include "tvg2D.h"

using namespace std;

// computes geometric total variation (TVG) energy density of degree 1 and 2
// inputs: dx, dy -- derivatives
//         all_Cs -- steering covariance matrix, filled in by estimateSteerCovarianceMatrix() function
// outputs: order1, order2 -- TVG energy, degree 1 and 2
void computeTvgEnergy2D(IplImage * dx, IplImage * dy, IplImage * order1, IplImage * order2, vector<IplImage*> & all_Cs){
  
  CvSize chan_size = cvGetSize(dx);
  CvSize border_size = cvSize(chan_size.width+2*BORDER_RADIUS, chan_size.height+2*BORDER_RADIUS);
  
  IplImage * dx_border = cvCreateImage(border_size, IPL_DEPTH_32F, 1);
  IplImage * dy_border = cvCreateImage(border_size, IPL_DEPTH_32F, 1);

  cvCopyMakeBorder( dx, dx_border, cvPoint(BORDER_RADIUS, BORDER_RADIUS), IPL_BORDER_REPLICATE);
  cvCopyMakeBorder( dy, dy_border, cvPoint(BORDER_RADIUS, BORDER_RADIUS), IPL_BORDER_REPLICATE);
  
  int window_length = 2*TVG_WINDOW_RADIUS+1;
  int row_count = window_length*window_length;
  
  IplImage * sub_x_diff = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_y_diff = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_yy = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_xx = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_xy = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_sum = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_mres = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * dx_weighted = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * dy_weighted = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * dxx_weighted = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * dyy_weighted = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * dx_rotated = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * dy_rotated = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  
  IplImage * gradients_img = cvCreateImage(cvSize(window_length*20, window_length*20), IPL_DEPTH_8U, 3);
  
  IplImage * sub_weights = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  
  IplImage * C11 = cvCreateImage(border_size, IPL_DEPTH_32F, 1);
  IplImage * C1221 = cvCreateImage(border_size, IPL_DEPTH_32F, 1);
  IplImage * C22 = cvCreateImage(border_size, IPL_DEPTH_32F, 1);
  
  cvCopyMakeBorder( all_Cs[0], C11, cvPoint(BORDER_RADIUS, BORDER_RADIUS), IPL_BORDER_REPLICATE);
  cvCopyMakeBorder( all_Cs[1], C1221, cvPoint(BORDER_RADIUS, BORDER_RADIUS), IPL_BORDER_REPLICATE);
  cvCopyMakeBorder( all_Cs[2], C22, cvPoint(BORDER_RADIUS, BORDER_RADIUS), IPL_BORDER_REPLICATE);

  int x1 = TVG_WINDOW_RADIUS;
  int y1 = TVG_WINDOW_RADIUS;
  
  CvScalar s;
  
  // create initial x_diff and y_diff values
  for(int i = 0; i < window_length; i++){
    for(int j = 0; j < window_length; j++){
      
      cvSet2D(sub_x_diff, i, j, cvScalarAll(x1-j));
      cvSet2D(sub_y_diff, i, j, cvScalarAll(y1-i));
    }
  }

  IplImage * sub_img = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 2);
  
  CvMat * gradients_mat = cvCreateMat(window_length*window_length, 2, CV_32F );
  CvMat * W = cvCreateMat(2, 2, CV_32F );
  
  double sum_weights, Vx, Vy, D1, D2, theta, S1, S2, gamma; 
  
  cvMul( sub_x_diff, sub_x_diff, sub_xx );
  cvMul( sub_y_diff, sub_y_diff, sub_yy );
  cvMul( sub_x_diff, sub_y_diff, sub_xy );
  
  double std_dev = TVG_SMOOTH_PAR;
  
  int x_border, y_border;
  
  double norm_weight = 0;
  double scale_factor = 0;
  
  for(int i = 0; i < chan_size.height; i++){
    
    for(int j = 0; j < chan_size.width; j++){
      
      x1 = j;
      y1 = i;
     
      x_border = x1+BORDER_RADIUS;
      y_border = y1+BORDER_RADIUS;

      cvSetImageROI( dx_border, cvRect(x_border-TVG_WINDOW_RADIUS, y_border-TVG_WINDOW_RADIUS, window_length, window_length));
      cvSetImageROI( dy_border, cvRect(x_border-TVG_WINDOW_RADIUS, y_border-TVG_WINDOW_RADIUS, window_length, window_length));
      cvSetImageROI( C11, cvRect(x_border-TVG_WINDOW_RADIUS, y_border-TVG_WINDOW_RADIUS, window_length, window_length));
      cvSetImageROI( C1221, cvRect(x_border-TVG_WINDOW_RADIUS, y_border-TVG_WINDOW_RADIUS, window_length, window_length));
      cvSetImageROI( C22, cvRect(x_border-TVG_WINDOW_RADIUS, y_border-TVG_WINDOW_RADIUS, window_length, window_length));
      
      cvMul( sub_xx, C11, sub_sum, 1.0/((-2.0)*std_dev));
      cvMul( sub_yy, C22, sub_mres, 1.0/((-2.0)*std_dev));
      cvAdd( sub_sum, sub_mres, sub_sum);
      cvMul( sub_xy, C1221, sub_mres, 1.0/((-2.0)*std_dev));
      cvAdd( sub_sum, sub_mres, sub_sum);
      
      cvExp(sub_sum, sub_weights);

      sum_weights = cvSum( sub_weights ).val[0];

      cvMul(dx_border, sub_weights, dx_weighted);
      cvMul(dy_border, sub_weights, dy_weighted);
      
      cvMerge(dx_weighted, dy_weighted, NULL, NULL, sub_img);
      
      cvReshape( sub_img, gradients_mat, 1, row_count);
      
      cvSVD(gradients_mat, W, NULL, NULL, CV_SVD_MODIFY_A );
      
      D1 = cvmGet( W, 0, 0 );
      D2 = cvmGet( W, 1, 1 );
      
      cvSet2D(order1, y1, x1, cvScalarAll(D1+D2));
      cvSet2D(order2, y1, x1, cvScalarAll(D1*D2));

    }
  }
  
  cvReleaseImage(&sub_img);
  cvReleaseImage(&dx_rotated);
  cvReleaseImage(&dy_rotated);
  
  cvReleaseImage(&gradients_img);
  
  cvReleaseImage(&dx_weighted);
  cvReleaseImage(&dy_weighted);
  cvReleaseImage(&dxx_weighted);
  cvReleaseImage(&dyy_weighted);

  cvReleaseMat(&gradients_mat);
  cvReleaseMat(&W);

  cvReleaseImage(&dx_border);
  cvReleaseImage(&dy_border);

  cvReleaseImage(&C11);
  cvReleaseImage(&C1221);
  cvReleaseImage(&C22);

  cvReleaseImage(&sub_x_diff);
  cvReleaseImage(&sub_y_diff);
  cvReleaseImage(&sub_yy);
  cvReleaseImage(&sub_xx);
  cvReleaseImage(&sub_xy);
  cvReleaseImage(&sub_sum);
  cvReleaseImage(&sub_mres);
  cvReleaseImage(&sub_weights);
}

// main function to compute geometric total variation (TVG) energy density of degree 1 and 2
// Regression with local steering kernels is used to estimate gradients dx and dy 
// input: gray_img -- grayscale image
// output: order1, order2 -- TVG Energy, degree 1 and 2
void getTvg2D( IplImage * gray_img, IplImage * order1, IplImage * order2 ){
  
  CvSize img_size = cvGetSize(gray_img);
    
  vector<IplImage *> all_Cs;
  
  IplImage * img_ptr;

  IplImage * chan = cvCreateImage(img_size, IPL_DEPTH_32F, 1);
  IplImage * new_chan = cvCreateImage(img_size, IPL_DEPTH_32F, 1);
  IplImage * dx = cvCreateImage(img_size, IPL_DEPTH_32F, 1);
  IplImage * dy = cvCreateImage(img_size, IPL_DEPTH_32F, 1);

  cvConvert(gray_img, chan);
  
  for(int i= 0; i < 3; i++){

    img_ptr = cvCreateImage(img_size, IPL_DEPTH_32F, 1);
    if(i==0 || i==2)
      cvSet(img_ptr, cvScalarAll(1.0));
    else
      cvSet(img_ptr, cvScalarAll(0.0));
    
    all_Cs.push_back(img_ptr);
  }

  cvSobel( chan, dx, 1, 0);
  cvSobel( chan, dy, 0, 1);
  
  cvConvertScale( dx, dx, 1.0/(8.0), 0 );
  cvConvertScale( dy, dy, 1.0/(8.0), 0 );
  
  for(int i= 0; i< 2; i++){
    
    estimateGradientsWithRegression(chan, all_Cs, dx, dy, new_chan);
    cvCopy(new_chan, chan);
    estimateSteerCovarianceMatrix(dx, dy, all_Cs);
  }
  
  computeTvgEnergy2D(dx, dy, order1, order2, all_Cs);
  
  cvReleaseImage(&chan);
  cvReleaseImage(&new_chan);

  cvReleaseImage(&dx);
  cvReleaseImage(&dy);
  
  for(int i= 0; i < all_Cs.size(); i++){
    cvReleaseImage(&all_Cs[i]);
  }

}



// main function to compute geometric total variation (TVG) energy density of degree 1 and 2
// Sobel filters are used to estimate gradients dx and dy 
// (i.e., getFastTvg2D does not use regression with local steering kernels)
// input: gray_img -- grayscale image
// output: order1, order2 -- TV Geometry, order 1 and 2
void getFastTvg2D( IplImage * gray_img, IplImage * order1, IplImage * order2 ){
  
  CvSize img_size = cvGetSize(gray_img);
    
  vector<IplImage *> all_Cs;
  
  IplImage * img_ptr;

  IplImage * chan = cvCreateImage(img_size, IPL_DEPTH_32F, 1);
  IplImage * new_chan = cvCreateImage(img_size, IPL_DEPTH_32F, 1);
  IplImage * dx = cvCreateImage(img_size, IPL_DEPTH_32F, 1);
  IplImage * dy = cvCreateImage(img_size, IPL_DEPTH_32F, 1);

  cvConvert(gray_img, chan);
  
  for(int i= 0; i < 3; i++){

    img_ptr = cvCreateImage(img_size, IPL_DEPTH_32F, 1);
    if(i==0 || i==2)
      cvSet(img_ptr, cvScalarAll(1.0));
    else
      cvSet(img_ptr, cvScalarAll(0.0));
    
    all_Cs.push_back(img_ptr);
  }

  cvSobel( chan, dx, 1, 0);
  cvSobel( chan, dy, 0, 1);
  
  cvConvertScale( dx, dx, 1.0/(8.0), 0 );
  cvConvertScale( dy, dy, 1.0/(8.0), 0 );
  
  estimateSteerCovarianceMatrix(dx, dy, all_Cs);

  computeTvgEnergy2D(dx, dy, order1, order2, all_Cs);
  
  cvReleaseImage(&chan);
  cvReleaseImage(&new_chan);

  cvReleaseImage(&dx);
  cvReleaseImage(&dy);
  
  for(int i= 0; i < all_Cs.size(); i++){
    cvReleaseImage(&all_Cs[i]);
  }

}
