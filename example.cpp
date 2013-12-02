/******************************************************************************************/
/*                                                                                        */
/* Texture descriptors for images based on geometric total variation energy density       */
/*                                                                                        */
/* Copyright (C) 2012  Drexel University                                                  */
/* Implemented by:  Dmitriy Bespalov (bespalov@gmail.com)                                 */
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

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <exception>
//#include <stdexcept>
#include <queue>

#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <math.h>

#include "tvg2D.h"

using namespace std;

static const float MIN_FRADIUS_DISPLAY = 1.5;

void drawOneFeature(IplImage * wimg, MyKeypoint * kp){
  
  float theta = kp->ori;
  float x = kp->col;
  float y = kp->row;
  float window_len = kp->fradius;
  
  float x1,y1,x2,y2, x3, y3, x4, y4, dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, x5, y5, dx5, dy5;
  
  dx1 = -window_len; dy1 = -window_len;
  dx2 = -window_len; dy2 = window_len;
  dx3 = window_len; dy3 = window_len;
  dx4 = window_len; dy4 = -window_len;
      
  dx5 = window_len; dy5 = 0.0;
      
  theta = -theta;
      
  x1 = x + (dx1*cos(theta)-dy1*sin(theta));
  y1 = y + (dx1*sin(theta)+dy1*cos(theta));
      
  x2 = x + (dx2*cos(theta)-dy2*sin(theta));
  y2 = y + (dx2*sin(theta)+dy2*cos(theta));
	
  x3 = x + (dx3*cos(theta)-dy3*sin(theta));
  y3 = y + (dx3*sin(theta)+dy3*cos(theta));
      
  x4 = x + (dx4*cos(theta)-dy4*sin(theta));
  y4 = y + (dx4*sin(theta)+dy4*cos(theta));
      
  x5 = x + (dx5*cos(theta)-dy5*sin(theta));
  y5 = y + (dx5*sin(theta)+dy5*cos(theta));
      
  cvLine(wimg, cvPoint(x1,y1), cvPoint(x2,y2), CV_RGB(0,255,0), 1);
  cvLine(wimg, cvPoint(x2,y2), cvPoint(x3,y3), CV_RGB(0,255,0), 1);
  cvLine(wimg, cvPoint(x3,y3), cvPoint(x4,y4), CV_RGB(0,255,0), 1);
  cvLine(wimg, cvPoint(x4,y4), cvPoint(x1,y1), CV_RGB(0,255,0), 1);
  
  cvLine(wimg, cvPoint(x,y), cvPoint(x5,y5), CV_RGB(255,0,0), 1);
  
  //printf("Drawing %s keypoint at (%.2f,%.2f), ori=%.3f, extend=%.2f\n", kp->dname.c_str(), x, y, theta, window_len);
}

void drawAllFeatures(IplImage * one_frame, vector<MyKeypoint*> & keypoints){
  
  for(int i = 0; i < keypoints.size(); i++){
    
    if (keypoints[i]->fradius >= MIN_FRADIUS_DISPLAY )
      drawOneFeature(one_frame, keypoints[i]);
  }
}


IplImage * drawDescriptorHistogram(vector<MyKeypoint*> & keypoints){
  
  if (keypoints.size()==0)
    return NULL;
  
  //  cvCopy(one_frame, wimg);
  int desc_dim = keypoints[0]->fv->Nrows();
  int num_bins = 30;
  double bin_scale = double(num_bins) / 200.0;
  
  int bin_pix = 10;
  
  IplImage * hist_img = cvCreateImage(cvSize(desc_dim*bin_pix, num_bins*bin_pix) , IPL_DEPTH_8U, 3 );
  
  Matrix * tmp_vals = new Matrix(num_bins, desc_dim);
  
  *tmp_vals = 0.0;
  
  int bin_idx;
      
  for(int i = 0; i < keypoints.size(); i++){
    
    for (int j = 0; j < desc_dim; j++){
      
      bin_idx = cvFloor(keypoints[i]->fv->element(j)*bin_scale);
      
      if (bin_idx >= num_bins)
	bin_idx = num_bins-1;

      tmp_vals->element(bin_idx, j)+= 1.0;
    }
  }
  
  *tmp_vals /= double(keypoints.size());

  for (int j = 0; j < desc_dim; j++){  
    for (int i = 0; i < num_bins; i++){

      
      int hist_val = cvFloor(tmp_vals->element(i, j)*1024.0);
      
      if (hist_val > 255.0)
	hist_val = 255.0;
      
      cvRectangle( hist_img, cvPoint( j*bin_pix, i*bin_pix ),
		   cvPoint( (j+1)*bin_pix - 1, (i+1)*bin_pix - 1),
		   CV_RGB(hist_val, hist_val, hist_val),
		   CV_FILLED );
    }
  }
  
  delete tmp_vals;
  
  return hist_img;
}


void processFrameForSiftFeatures(IplImage* one_frame, vector<MyKeypoint*> & keypoints, bool draw_keypoints=false){
  
  MyKeypoint* one_keypoint;
  
  CvSize frame_size = cvGetSize(one_frame);
  IplImage * gray_frame = cvCreateImage(frame_size , IPL_DEPTH_8U, 1 );
  cvCvtColor( one_frame, gray_frame, CV_BGR2GRAY ); // convert frame to grayscale
  computeSiftFeaturesFromIplImage(gray_frame, keypoints);
  cvReleaseImage(&gray_frame);
  
}

double getStatisticsInfoForMatrix(IplImage * order1_img, string name, double osv=0.0){

  ColumnVector * tmp_vals1 = new ColumnVector(order1_img->width*order1_img->height);

  int tind = 0;
  for(int y = 0; y < order1_img->height; y++){
    for(int x = 0; x < order1_img->width; x++){
      
      tmp_vals1->element(tind) = cvGet2D(order1_img, y, x).val[0];
      tind++;
    }
  }
  
  SortAscending(*tmp_vals1);
  
  int vec_size = order1_img->width*order1_img->height;

  // printf("%s: %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf\n", name.c_str(), 
  // 	 tmp_vals1->element(0), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.01)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.05)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.1)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.2)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.3)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.4)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.5)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.6)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.7)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.8)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.9)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.95)), 
  // 	 tmp_vals1->element(cvRound(double(vec_size)*0.99)), 
  // 	 tmp_vals1->element(vec_size-1));
  
  double ret_val = 0.0;

  if(osv == 0.0)
    ret_val = tmp_vals1->element(0);
  else if(osv == 1.0)
    ret_val = tmp_vals1->element(vec_size-1);
  else
    ret_val = tmp_vals1->element(cvRound(double(vec_size)*osv));
  
  delete tmp_vals1;
  
  
  return ret_val;
}



IplImage* createDisplayImages(IplImage * tvg_o1){
  
  CvSize frame_size = cvGetSize(tvg_o1);
  
  IplImage *disp_o1 = cvCreateImage(frame_size , IPL_DEPTH_8U, 3 );
  IplImage * tmp_o1 = cvCreateImage(frame_size , IPL_DEPTH_8U, 1 );
  
  double low_o1 = getStatisticsInfoForMatrix(tvg_o1, "degree1", 0.05);
  double hi_o1 = getStatisticsInfoForMatrix(tvg_o1, "degree1", 0.95);
  
  cvConvertScale(tvg_o1, tmp_o1, 255.0/(hi_o1-low_o1), (-low_o1)*255.0/(hi_o1-low_o1));
  
  cvThreshold(tmp_o1, tmp_o1, 0.0, 0, CV_THRESH_TOZERO );
  cvThreshold(tmp_o1, tmp_o1, 255.0, 0, CV_THRESH_TRUNC );

  cvMerge(tmp_o1, tmp_o1, tmp_o1, NULL, disp_o1);

  cvReleaseImage(&tmp_o1);
  
  return disp_o1;
}


int main(int argc, char *argv[]) {
  
  try {

    char filename[1000] = "";
    
    IplImage * tmp_img = cvLoadImage(argv[1]);
      
    double imsize_scale = 500.0 / double(max(tmp_img->width, tmp_img->height));
    CvSize frame_size = cvSize(cvRound(double(tmp_img->width)*imsize_scale), cvRound(double(tmp_img->height)*imsize_scale));

    IplImage * i_img = cvCreateImage(frame_size, IPL_DEPTH_8U, 3 );
    cvResize(tmp_img, i_img, CV_INTER_LINEAR);

    IplImage * gray_img = cvCreateImage(frame_size , IPL_DEPTH_8U, 1 );
    cvCvtColor( i_img, gray_img, CV_BGR2GRAY ); // convert frame to grayscale
      
    IplImage * img1_o1 = cvCreateImage(frame_size , IPL_DEPTH_32F, 1 );
    IplImage * img1_o2 = cvCreateImage(frame_size , IPL_DEPTH_32F, 1 );
    
    printf("Computing TVG Energy (degree 1 and 2) ...\n");
    getTvg2D( gray_img, img1_o1, img1_o2 );
    
    // printf("Computing Fast TVG Energy...\n");
    // getFastTvg2D( gray_img, img1_o1, img1_o2 );
    

    IplImage * disp_o1;
    IplImage * disp_o2;
    
    //create an illustration for TVG energies of degree 1 and 2
    disp_o1=createDisplayImages(img1_o1);
    disp_o2=createDisplayImages(img1_o2);
    
    printf("Computing SIFT descriptors ...\n");
    vector<MyKeypoint*> sift_keypoints;
    computeSiftFeaturesFromIplImage(gray_img, sift_keypoints);
    //    printf("SIFT: %i\n", (int)sift_keypoints.size());
    
    printf("Computing TVG descriptors ...\n");
    vector<MyKeypoint*> tvg_keypoints;
    computeTvgDescriptorsForFeatures(sift_keypoints, tvg_keypoints, img1_o1, img1_o2);
    //    printf("TVG: %i\n", (int)tvg_keypoints.size());
      
    // draw detected SIFT interest points 
    drawAllFeatures(i_img, sift_keypoints);
    
    // compute histograms for SIFT and TVG descriptors (each dimension is handled independently)
    IplImage * sift_hist_img = drawDescriptorHistogram(sift_keypoints);
    IplImage * tvg_hist_img = drawDescriptorHistogram(tvg_keypoints);
    
    sprintf(filename, "%s.tvg-deg1.png", argv[1]);
    cvSaveImage(filename, disp_o1);

    sprintf(filename, "%s.tvg-deg2.png", argv[1]);
    cvSaveImage(filename, disp_o2);

    sprintf(filename, "%s.interest-pts.png", argv[1]);
    cvSaveImage(filename, i_img);

    sprintf(filename, "%s.sift-hist.png", argv[1]);
    cvSaveImage(filename, sift_hist_img);

    sprintf(filename, "%s.tvg-hist.png", argv[1]);
    cvSaveImage(filename, tvg_hist_img);

    // cvNamedWindow( "sift_hist", CV_WINDOW_AUTOSIZE);
    // cvNamedWindow( "tvg_hist", CV_WINDOW_AUTOSIZE);

    // cvNamedWindow( "tvg-o1", CV_WINDOW_AUTOSIZE);
    // cvNamedWindow( "tvg-o2", CV_WINDOW_AUTOSIZE);

    // cvNamedWindow( "original img", CV_WINDOW_AUTOSIZE);

    // cvMoveWindow("tvg-o1", 20, 20);
    // cvMoveWindow("tvg-o2", 20, 600);
   
    // cvMoveWindow("original img", 900, 200);

    // cvMoveWindow("sift_hist", 300, 800);
    // cvMoveWindow("tvg_hist", 500, 800);
	
    // cvShowImage( "original img", i_img );
   
    // cvShowImage( "sift_hist", sift_hist_img );
    // cvShowImage( "tvg_hist", tvg_hist_img );
    
    // cvShowImage( "tvg-o1", disp_o1 );
    // cvShowImage( "tvg-o2", disp_o2 );
    // // cvShowImage( "tvg-o1-fast", disp_img2_o1 );
    // // cvShowImage( "tvg-o2-fast", disp_img2_o2 );

    // cvWaitKey(0);
    
    cvReleaseImage(&gray_img);
    cvReleaseImage(&tmp_img);
    cvReleaseImage(&i_img);
    //cvReleaseImage(...);
    
  }

  catch(exception & ex) { 
    cout << ex.what() << endl; 
    return 1; 
  }

  return 0;
  
}
