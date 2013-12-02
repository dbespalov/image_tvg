
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

void createPillBoxFilter(IplImage * filter, int radius){
  
  int mid_x = radius;
  int mid_y = radius;
  
  CvSize size = cvGetSize(filter);
  
  for(int i = 0; i < size.height; i++){
    for(int j = 0; j < size.width; j++){
      
      int x = mid_x-j;
      int y = mid_y-i;
      
      if((x*x)+(y*y) <= radius*radius){
	cvSet2D(filter, i, j, cvScalarAll(1.0));
	//printf(" 1.0 ");
      }
      else{
	cvSet2D(filter, i, j, cvScalarAll(0.0));
	//printf(" 0.0 ");
      }
    }
    
    //printf("\n");
  }
  
}


// get points coordinates after transformation
double warpPoint(double x1, double y1, double & x2, double & y2, CvMat * map_mat){
  
  double nx1 = (cvmGet(map_mat, 0, 0)*double(x1) + cvmGet(map_mat, 0, 1)*double(y1) + cvmGet(map_mat, 0, 2)) / (cvmGet(map_mat, 2, 0)*double(x1) + cvmGet(map_mat, 2, 1)*double(y1)+cvmGet(map_mat, 2, 2));
  double ny1 = (cvmGet(map_mat, 1, 0)*double(x1) + cvmGet(map_mat, 1, 1)*double(y1) + cvmGet(map_mat, 1, 2)) / (cvmGet(map_mat, 2, 0)*double(x1) + cvmGet(map_mat, 2, 1)*double(y1)+cvmGet(map_mat, 2, 2));
  
  x2 = nx1;
  y2 = ny1;
}



// performs regression to estimate gradients
// inputs: chan -- input intensity image
//         all_Cs -- steering covariance matrix, filled in by estimateSteerCovarianceMatrix() function
// outputs: dx, dy -- derivative images
//          new_chan -- new estimates of intensities, can be used for steer based smoothing, otherwise, just disregard
void estimateGradientsWithRegression(IplImage * chan, vector<IplImage*> & all_Cs, IplImage * dx, IplImage * dy, IplImage * new_chan=NULL){

  try{

    CvSize chan_size = cvGetSize(chan);

    CvSize border_size = cvSize(chan_size.width+2*BORDER_RADIUS, chan_size.height+2*BORDER_RADIUS);
  
    IplImage * chan_border = cvCreateImage(border_size, IPL_DEPTH_32F, 1);
  
    cvCopyMakeBorder( chan, chan_border, cvPoint(BORDER_RADIUS, BORDER_RADIUS), IPL_BORDER_REPLICATE);//, cvScalarAll(0));//IPL_BORDER_REPLICATE);
  
    int window_length = 2*GRADIENT_WINDOW_RADIUS+1;
    int row_count = window_length*window_length;
  
    IplImage * sub_x_diff = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
    IplImage * sub_y_diff = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);

    IplImage * sub_intensity = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);

    IplImage * sub_yy = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
    IplImage * sub_xx = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
    IplImage * sub_xy = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
    IplImage * sub_mres = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
    IplImage * sub_sum = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
    IplImage * Xx = cvCreateImage(cvSize(6, row_count), IPL_DEPTH_32F, 1);
    IplImage * Xw = cvCreateImage(cvSize(6, row_count), IPL_DEPTH_32F, 1);
    IplImage * corr_mat = cvCreateImage(cvSize(6, 6), IPL_DEPTH_32F, 1);
    IplImage * mat_to_inv = cvCreateImage(cvSize(6, 6), IPL_DEPTH_32F, 1);
    IplImage * inv_mat = cvCreateImage(cvSize(6, 6), IPL_DEPTH_32F, 1);
    IplImage * A_mat = cvCreateImage(cvSize(row_count, 6), IPL_DEPTH_32F, 1);
    IplImage * res_mat = cvCreateImage(cvSize(1, 6), IPL_DEPTH_32F, 1);
  
    IplImage * C11 = cvCreateImage(border_size, IPL_DEPTH_32F, 1);
    IplImage * C1221 = cvCreateImage(border_size, IPL_DEPTH_32F, 1);
    IplImage * C22 = cvCreateImage(border_size, IPL_DEPTH_32F, 1);

    cvCopyMakeBorder( all_Cs[0], C11, cvPoint(BORDER_RADIUS, BORDER_RADIUS), IPL_BORDER_REPLICATE);
    cvCopyMakeBorder( all_Cs[1], C1221, cvPoint(BORDER_RADIUS, BORDER_RADIUS), IPL_BORDER_REPLICATE);
    cvCopyMakeBorder( all_Cs[2], C22, cvPoint(BORDER_RADIUS, BORDER_RADIUS), IPL_BORDER_REPLICATE);

    CvMat * reshape_mat = cvCreateMat(row_count, 1, CV_32F );
    CvMat * y_mat = cvCreateMat(row_count, 1, CV_32F );
    CvMat * weights_mat = cvCreateMat(row_count, 1, CV_32F );

    int x1 = GRADIENT_WINDOW_RADIUS;
    int y1 = GRADIENT_WINDOW_RADIUS;
  
    for(int i = 0; i < window_length; i++){
      for(int j = 0; j < window_length; j++){
      
	//s.val[0] = x1-j;
	cvSet2D(sub_x_diff, i, j, cvScalarAll(x1-j));
      
	//s.val[0] = y1-i;
	cvSet2D(sub_y_diff, i, j, cvScalarAll(y1-i));
      }
    }

    cvMul( sub_x_diff, sub_x_diff, sub_xx );
    cvMul( sub_y_diff, sub_y_diff, sub_yy );
    cvMul( sub_x_diff, sub_y_diff, sub_xy );
  
    cvZero(corr_mat);
    cvSet2D(corr_mat, 0, 0, cvScalarAll(0.0000001));
    cvSet2D(corr_mat, 1, 1, cvScalarAll(0.0000001));
    cvSet2D(corr_mat, 2, 2, cvScalarAll(0.0000001));
    cvSet2D(corr_mat, 3, 3, cvScalarAll(0.0000001));
    cvSet2D(corr_mat, 4, 4, cvScalarAll(0.0000001));
    cvSet2D(corr_mat, 5, 5, cvScalarAll(0.0000001));
  
    // fill in Xx matrix
    cvSetImageROI( Xx, cvRect(0, 0, 1, row_count));
    cvSet(Xx, cvScalarAll(1.0));
  
    cvReshape( sub_x_diff, reshape_mat, 1, row_count);
    cvSetImageROI( Xx, cvRect(1, 0, 1, row_count));
    cvCopy(reshape_mat, Xx);

    cvReshape( sub_y_diff, reshape_mat, 1, row_count);
    cvSetImageROI( Xx, cvRect(2, 0, 1, row_count));
    cvCopy(reshape_mat, Xx);

    cvReshape( sub_xx, reshape_mat, 1, row_count);
    cvSetImageROI( Xx, cvRect(3, 0, 1, row_count));
    cvCopy(reshape_mat, Xx);

    cvReshape( sub_xy, reshape_mat, 1, row_count);
    cvSetImageROI( Xx, cvRect(4, 0, 1, row_count));
    cvCopy(reshape_mat, Xx);

    cvReshape( sub_yy, reshape_mat, 1, row_count);
    cvSetImageROI( Xx, cvRect(5, 0, 1, row_count));
    cvCopy(reshape_mat, Xx);
    cvResetImageROI( Xx );
    // done filling Xx
  
    double std_dev = GRADIENT_SMOOTH_PAR;
    double sum_weights;

    int x_border, y_border;
  
    for(int i = 0; i < chan_size.height; i++){
    
      for(int j = 0; j < chan_size.width; j++){

	x1 = j;
	y1 = i;
      
	x_border = x1+BORDER_RADIUS;
	y_border = y1+BORDER_RADIUS;

	cvSetImageROI( chan_border, cvRect(x_border-GRADIENT_WINDOW_RADIUS, y_border-GRADIENT_WINDOW_RADIUS, window_length, window_length));
	cvSetImageROI( C11, cvRect(x_border-GRADIENT_WINDOW_RADIUS, y_border-GRADIENT_WINDOW_RADIUS, window_length, window_length));
	cvSetImageROI( C1221, cvRect(x_border-GRADIENT_WINDOW_RADIUS, y_border-GRADIENT_WINDOW_RADIUS, window_length, window_length));
	cvSetImageROI( C22, cvRect(x_border-GRADIENT_WINDOW_RADIUS, y_border-GRADIENT_WINDOW_RADIUS, window_length, window_length));
      
	cvMul( sub_xx, C11, sub_sum, 1.0/((-2.0)*std_dev));
	cvMul( sub_yy, C22, sub_mres, 1.0/((-2.0)*std_dev));
	cvAdd( sub_sum, sub_mres, sub_sum);
	cvMul( sub_xy, C1221, sub_mres, 1.0/((-2.0)*std_dev));
	cvAdd( sub_sum, sub_mres, sub_sum);
      
	cvExp(sub_sum, sub_sum);

	sum_weights = cvSum( sub_sum ).val[0];
      
	cvReshape(sub_sum, weights_mat, 0, row_count);
	cvConvert(chan_border, sub_intensity);
	cvReshape(sub_intensity, y_mat, 0, row_count);
      
	cvSetImageROI( Xw, cvRect(0, 0, 1, row_count));
	cvCopy(weights_mat, Xw);
	cvSetImageROI( Xw, cvRect(1, 0, 1, row_count));
	cvCopy(weights_mat, Xw);
	cvSetImageROI( Xw, cvRect(2, 0, 1, row_count));
	cvCopy(weights_mat, Xw);
	cvSetImageROI( Xw, cvRect(3, 0, 1, row_count));
	cvCopy(weights_mat, Xw);
	cvSetImageROI( Xw, cvRect(4, 0, 1, row_count));
	cvCopy(weights_mat, Xw);
	cvSetImageROI( Xw, cvRect(5, 0, 1, row_count));
	cvCopy(weights_mat, Xw);
	cvResetImageROI( Xw );
      
	cvMul(Xx, Xw, Xw);
      
	cvGEMM( Xx, Xw, 1.0, corr_mat, 1.0, mat_to_inv, CV_GEMM_A_T );
      
	cvInvert( mat_to_inv, inv_mat);

	cvGEMM( inv_mat, Xw, 1.0, NULL, 0, A_mat, CV_GEMM_B_T );
	cvGEMM( A_mat, y_mat, 1.0, NULL, 0, res_mat, 0 );
	
	if(new_chan!=NULL)
	  cvSet2D(new_chan, y1, x1, cvGet2D(res_mat, 0, 0));
	
	cvSet2D(dx, y1, x1, cvGet2D(res_mat, 1, 0));
	cvSet2D(dy, y1, x1, cvGet2D(res_mat, 2, 0));

      }
    }
  
    cvResetImageROI( chan );
    cvResetImageROI( C11 );
    cvResetImageROI( C1221 );
    cvResetImageROI( C22 );

    cvReleaseImage(&sub_x_diff);
    cvReleaseImage(&sub_y_diff);
    cvReleaseImage(&sub_yy);
    cvReleaseImage(&sub_xx);
    cvReleaseImage(&sub_xy);
    cvReleaseImage(&sub_mres);
    cvReleaseImage(&sub_sum);
    cvReleaseImage(&Xx);
    cvReleaseImage(&Xw);
    cvReleaseImage(&sub_intensity);
    cvReleaseImage(&corr_mat);
    cvReleaseImage(&mat_to_inv);
    cvReleaseImage(&inv_mat);
    cvReleaseImage(&A_mat);
    cvReleaseImage(&res_mat);
  
    cvReleaseImage(&C11);
    cvReleaseImage(&C1221);
    cvReleaseImage(&C22);
  
    cvReleaseMat(&reshape_mat);
    cvReleaseMat(&y_mat);
    cvReleaseMat(&weights_mat);

  }
  catch(Exception  e)  { 
    cout << e.what() << endl ;  return;
  }
}
 

// estimates covariance matrix for every pixel, 
// inputs:  dx, dy -- image derivatives
// outputs: all_Cs -- steering covariance matrix, filled 
void estimateSteerCovarianceMatrix(IplImage * dx, IplImage * dy, vector<IplImage*> & all_Cs){
  
  CvSize chan_size = cvGetSize(dx);
  
  CvSize border_size = cvSize(chan_size.width+2*BORDER_RADIUS, chan_size.height+2*BORDER_RADIUS);
  
  IplImage * dx_border = cvCreateImage(border_size, IPL_DEPTH_32F, 1);
  IplImage * dy_border = cvCreateImage(border_size, IPL_DEPTH_32F, 1);
  
  cvCopyMakeBorder( dx, dx_border, cvPoint(BORDER_RADIUS, BORDER_RADIUS), IPL_BORDER_REPLICATE);
  cvCopyMakeBorder( dy, dy_border, cvPoint(BORDER_RADIUS, BORDER_RADIUS), IPL_BORDER_REPLICATE);

  IplImage * merged_d = cvCreateImage(border_size, IPL_DEPTH_32F, 2);
  
  int window_length = 2*STEEPEST_DESCEND_WINDOW_RADIUS+1;
  int row_count = window_length*window_length;
  
  IplImage * sub_x_diff = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_y_diff = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_yy = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_xx = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_xy = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_sum = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_mres = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 1);
  IplImage * sub_weights = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 2);

 
  int x1 = STEEPEST_DESCEND_WINDOW_RADIUS;
  int y1 = STEEPEST_DESCEND_WINDOW_RADIUS;
  
  CvScalar s;
  
  // create initial x_diff and y_diff values
  for(int i = 0; i < window_length; i++){
    for(int j = 0; j < window_length; j++){
      
      s.val[0] = x1-j;
      cvSet2D(sub_x_diff, i, j, s);
      
      s.val[0] = y1-i;
      cvSet2D(sub_y_diff, i, j, s);
    }
  }
  
  IplImage * sub_img = cvCreateImage(cvSize(window_length, window_length), IPL_DEPTH_32F, 2);
  
  CvMat * gradients_mat = cvCreateMat(window_length*window_length, 2, CV_32F );
  CvMat * W = cvCreateMat(2, 2, CV_32F );
  CvMat * V = cvCreateMat(2, 2, CV_32F );
  
  cvMerge(dx_border, dy_border, NULL, NULL, merged_d);
  
  Matrix Ut(2,2);
  Matrix L(2,2);
  Matrix C(2,2);
  
  double sum_weights, Vx, Vy, D1, D2, theta, S1, S2, gamma; 
  
  cvMul( sub_x_diff, sub_x_diff, sub_xx );
  cvMul( sub_y_diff, sub_y_diff, sub_yy );
  cvMul( sub_x_diff, sub_y_diff, sub_xy );
      
  createPillBoxFilter(sub_sum, STEEPEST_DESCEND_WINDOW_RADIUS);
  cvMerge(sub_sum, sub_sum, NULL, NULL, sub_weights);
  s = cvSum( sub_sum );
  sum_weights = s.val[0];
  
  int x_border, y_border;

  for(int i = 0; i < chan_size.height; i++){
    
    for(int j = 0; j < chan_size.width; j++){
      
      x1 = j;
      y1 = i;
      
      x_border = x1+BORDER_RADIUS;
      y_border = y1+BORDER_RADIUS;
      
      cvSetImageROI( merged_d, cvRect(x_border-STEEPEST_DESCEND_WINDOW_RADIUS, y_border-STEEPEST_DESCEND_WINDOW_RADIUS, window_length, window_length));
      
      cvMul(merged_d, sub_weights, sub_img);
      
      cvReshape( sub_img, gradients_mat, 1, row_count);
      
      cvSVD(gradients_mat, W, NULL, V, CV_SVD_MODIFY_A );
      
      Vx = cvmGet( V, 0, 1 );
      Vy = cvmGet( V, 1, 1 );
      
      D1 = cvmGet( W, 0, 0 );
      D2 = cvmGet( W, 1, 1 );
      
      theta = atan(Vx/(Vy+0.000001));
      S1 = (D1+LAMBDA) / (D2+LAMBDA);
      S2 = (D2+LAMBDA) / (D1+LAMBDA);
      gamma = pow((D1*D2+0.001)/sum_weights, ALPHA);
      
      Ut(1,1) = cos(theta);
      Ut(1,2) = sin(theta);
      Ut(2,1) = -sin(theta);
      Ut(2,2) = cos(theta);
      
      L = 0.0;
      L(1,1) = S1;
      L(2,2) = S2;
      
      C = Ut*L*Ut.t();
      C *= gamma;

      cvSet2D(all_Cs[0], y1, x1, cvScalarAll(C(1,1)));
      cvSet2D(all_Cs[1], y1, x1, cvScalarAll(C(1,2)+C(2,1)));
      cvSet2D(all_Cs[2], y1, x1, cvScalarAll(C(2,2)));
      
    }
  }
  
  cvResetImageROI( merged_d );
  
  cvReleaseImage(&sub_img);
  
  cvReleaseMat(&gradients_mat);
  cvReleaseMat(&W);
  cvReleaseMat(&V);

  cvReleaseImage(&dx_border);
  cvReleaseImage(&dy_border);

  cvReleaseImage(&merged_d);
  cvReleaseImage(&sub_x_diff);
  cvReleaseImage(&sub_y_diff);
  cvReleaseImage(&sub_yy);
  cvReleaseImage(&sub_xx);
  cvReleaseImage(&sub_xy);
  cvReleaseImage(&sub_sum);
  cvReleaseImage(&sub_mres);
  cvReleaseImage(&sub_weights);
  
}



