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

void releaseKeyPoints(vector<MyKeypoint*> & keypoints){
  
  for(int i = 0; i < keypoints.size(); i++)
    delete keypoints[i];
  
  keypoints.clear();
}

Image createImageFromIplImage(IplImage* img){
  
  CvSize frame_size = cvGetSize(img);
  Image result_img;
    
  CvScalar s;
  
  result_img = CreateImage(frame_size.height, frame_size.width);
  for (int r = 0; r < frame_size.height; r++){
    for (int c = 0; c < frame_size.width; c++){
      s=cvGet2D(img,r,c); // get the (i,j) pixel value
      result_img->pixels[r*result_img->stride+c] = float(s.val[0]);
    }
  }

  return result_img;
}


bool getDxDyHaar(IplImage * integral_img, double nxr, double nyr, double kernel_size, double & dxv, double & dyv){
  
  int x = cvRound(nxr);
  int y = cvRound(nyr);
  
  int ks = cvRound(kernel_size);
  
  if(ks < 3)
    ks = 3;
  
  if(ks % 2 == 0)
    ks++;
  
  int kr = (ks-1)/2;
  
  if(y-kr < 0 || x-kr < 0 || y+kr+1 >= integral_img->height || x+kr+1 >= integral_img->width)
    return false;
  
  double tlv = cvGet2D(integral_img, y-kr, x-kr).val[0];
  double brv = cvGet2D(integral_img, y+kr+1, x).val[0];
  
  double trv = cvGet2D(integral_img, y-kr, x).val[0];
  double blv = cvGet2D(integral_img, y+kr+1, x-kr).val[0];
  
  double left_value = -(tlv+brv-trv-blv);
  if (left_value > 0.0)
    left_value = 0.0;

  tlv = cvGet2D(integral_img, y-kr, x+1).val[0];
  brv = cvGet2D(integral_img, y+kr+1, x+kr+1).val[0];
  
  trv = cvGet2D(integral_img, y-kr, x+kr+1).val[0];
  blv = cvGet2D(integral_img, y+kr+1, x+1).val[0];
  
  double right_value = (tlv+brv-trv-blv);
  if (right_value < 0.0)
    right_value = 0.0;
  
  tlv = cvGet2D(integral_img, y-kr, x-kr).val[0];
  brv = cvGet2D(integral_img, y, x+kr+1).val[0];
  
  trv = cvGet2D(integral_img, y-kr, x+kr+1).val[0];
  blv = cvGet2D(integral_img, y, x-kr).val[0];
  
  double top_value = -(tlv+brv-trv-blv);
  if (top_value > 0.0)
    top_value = 0.0;

  tlv = cvGet2D(integral_img, y+1, x-kr).val[0];
  brv = cvGet2D(integral_img, y+kr+1, x+kr+1).val[0];
  
  trv = cvGet2D(integral_img, y+1, x+kr+1).val[0];
  blv = cvGet2D(integral_img, y+kr+1, x-kr).val[0];
  
  double bottom_value = (tlv+brv-trv-blv);
  if (bottom_value < 0.0)
    bottom_value = 0.0;

  dxv = (left_value+right_value)/double(2*ks*kr);
  dyv = (top_value+bottom_value)/double(2*ks*kr);
  
  
  return true;
}

bool getTvgHaar(IplImage * o_tvg, IplImage * integral_img, double nxr, double nyr, double kernel_size, double & ov){
  
  int ks = cvRound(kernel_size);
  
  if(ks < 3){ // bi-linear interpolation
    
    if(cvFloor(nyr) < 0 || cvFloor(nxr) < 0 || cvCeil(nyr) >= o_tvg->height || cvCeil(nxr) >= o_tvg->width)
      return false;
    
    double fx0y0 = cvGet2D(o_tvg, cvFloor(nyr), cvFloor(nxr)).val[0];
    double fx0y1 = cvGet2D(o_tvg, cvCeil(nyr), cvFloor(nxr)).val[0];
    double fx1y1 = cvGet2D(o_tvg, cvCeil(nyr), cvCeil(nxr)).val[0];
    double fx1y0 = cvGet2D(o_tvg, cvFloor(nyr), cvCeil(nxr)).val[0];
    
    double px = nxr - double(cvFloor(nxr));
    double py = nyr - double(cvFloor(nyr));
    
    ov = fx0y0*(1-px)*(1-py) + fx0y1*(1-px)*py + fx1y0*px*(1-py) + fx1y1*px*py;
    
    if (ov < 0.0)
      return false;
  }
  else{ // aggregate values

    int x = cvRound(nxr);
    int y = cvRound(nyr);
    
    if(ks % 2 == 0)
      ks++;
    
    int kr = (ks-1)/2;
    
    if(y-kr < 0 || x-kr < 0 || y+kr+1 >= integral_img->height || x+kr+1 >= integral_img->width)
      return false;
    
    double tlv = cvGet2D(integral_img, y-kr, x-kr).val[0];
    double brv = cvGet2D(integral_img, y+kr+1, y+kr+1).val[0];
    
    double trv = cvGet2D(integral_img, y-kr, x+kr+1).val[0];
    double blv = cvGet2D(integral_img, y+kr+1, x-kr).val[0];
    
    ov = (tlv+brv-trv-blv);
    
    if (ov < 0.0)
      return false;
  }
  
    
  return true;
  
}


void buildTvgDescriptor(MyKeypoint* kp, ColumnVector * tvg_vec,
			IplImage * o1, IplImage * o2, IplImage * integral_o1, IplImage * integral_o2){
  
  
  if (TMP_TVG.size()==0){
    for (int i = 0; i < NUM_TVG_FEATURES_PER_SUBWIN; i++)
      TMP_TVG.push_back(new ColumnVector(TVG_FEATURE_DIMS/NUM_TVG_FEATURES_PER_SUBWIN));
  }
  
  CvSize frame_size = cvGetSize(o1);
  
  double win_size = double(kp->fradius)*2.0;
  double win_scale = 1.5*double(kp->fradius)/SIFT_WINDOW_SCALE_FACTOR;
  double subwin_size = win_size / double(TVG_WINDOW_DIV_FACTOR);
  double kernel_size = win_size / 10.0;
  double surf_sigma = win_size / 20.0;
  
  int tvg_vec_ind = 0;

  double x = double(kp->col);
  double y = double(kp->row);
  
  double dxv, dyv, nx, ny, nxr, nyr, new_dxv, new_dyv, o1v, o2v, o1dx, o1dy;
  
  int n_samples = 0;
  bool ret_v;
  
  double theta = -double(kp->ori);
  
  double off_set  = subwin_size/double(TVG_SUBWIN_SAMPLES*2);
    
  double mean1, mean2, stddev1, stddev2;

  for (int i = 0; i < NUM_TVG_FEATURES_PER_SUBWIN; i++)
    *(TMP_TVG[i])=0.0;
  *tvg_vec = 0.0;
  
  double x_pos_weight, y_pos_weight, x_pos_weight2, y_pos_weight2, gauss_weight, weight_distance;
  
  for(double x_start = win_size/2.0 - off_set; x_start > -win_size/2.0; x_start-=subwin_size){
    for(double y_start = -win_size/2.0 + off_set; y_start < win_size/2.0; y_start+=subwin_size){
      
      x_pos_weight = x_start - (subwin_size/2.0) + off_set;
      y_pos_weight = y_start + (subwin_size/2.0) - off_set;
      
      n_samples=0;
      
      for(double dx = 0.0; dx < subwin_size; dx += subwin_size/double(TVG_SUBWIN_SAMPLES)){
	for(double dy = 0.0; dy < subwin_size; dy += subwin_size/double(TVG_SUBWIN_SAMPLES)){

	  nx = x_start - dx;
	  ny = y_start + dy;
	  
	  nxr = x + (nx*cos(theta)-ny*sin(theta));
	  nyr = y + (nx*sin(theta)+ny*cos(theta));
	  
	  ret_v = getTvgHaar(o1, integral_o1, nxr, nyr, kernel_size, o1v);
	  if(!ret_v)
	    o1v=0.0;

	  ret_v = getTvgHaar(o2, integral_o2, nxr, nyr, kernel_size, o2v);
	  if(!ret_v)
	    o2v=0.0;
	  
	  TMP_TVG[0]->element(tvg_vec_ind) += o1v;
	  TMP_TVG[1]->element(tvg_vec_ind) += o2v;
	  
	  ret_v = getDxDyHaar(integral_o1, nxr, nyr, kernel_size, o1dx, o1dy);
	  
	  if(!ret_v){
	    o1dx=0.0;
	    o1dy=0.0;
	  }
	  
	  if (NUM_TVG_FEATURES_PER_SUBWIN == 4) {
	    TMP_TVG[2]->element(tvg_vec_ind) += abs(o1dx);
	    TMP_TVG[3]->element(tvg_vec_ind) += abs(o1dy);
	  }
	  else if (NUM_TVG_FEATURES_PER_SUBWIN == 6){

	    if (o1dx >= 0.0)
	      TMP_TVG[2]->element(tvg_vec_ind) += abs(o1dx);
	    else
	      TMP_TVG[4]->element(tvg_vec_ind) += abs(o1dx);
	    
	    if (o1dx >= 0.0)
	      TMP_TVG[3]->element(tvg_vec_ind) += abs(o1dy);
	    else
	      TMP_TVG[5]->element(tvg_vec_ind) += abs(o1dy);

	  }

	  
	  n_samples++;
	}
      }
      
      x_pos_weight2 = x + (x_pos_weight*cos(theta)-y_pos_weight*sin(theta));
      y_pos_weight2 = y + (x_pos_weight*sin(theta)+y_pos_weight*cos(theta));
      
      weight_distance = sqrt((x_pos_weight2-x)*(x_pos_weight2-x)+(y_pos_weight2-y)*(y_pos_weight2-y));
      gauss_weight = exp(-(weight_distance*weight_distance/(2.0*win_scale*win_scale)));
      
      for (int i = 0; i < NUM_TVG_FEATURES_PER_SUBWIN; i++)
	TMP_TVG[i]->element(tvg_vec_ind) *= gauss_weight;
      
      tvg_vec_ind++;
    }
  }
  
  double tvg_eps_thresh = 0.00000001;
  
  // first two elements in TMP_TVG are for sums of TVG energies for degree 1 and 2  -- scale independenly
  double sum_sq_val = 0.0;
  for (int i = 0; i < 2; i++){
    for (int pass_id = 0; pass_id < 2; pass_id++){
      sum_sq_val = TMP_TVG[i]->SumSquare();
      if (sum_sq_val < tvg_eps_thresh)
	*(TMP_TVG[i])=0.0;
      else
	*(TMP_TVG[i]) /= sqrt(sum_sq_val);
      
      if (pass_id == 0){
	for(int j = 0; j < TVG_FEATURE_DIMS/NUM_TVG_FEATURES_PER_SUBWIN; j++){
	  if (TMP_TVG[i]->element(j) > 0.2)
	    TMP_TVG[i]->element(j) = 0.2;
	}
      }
    }
  }

  for (int pass_id = 0; pass_id < 2; pass_id++){  
    sum_sq_val = 0.0;
    // rest of elements in TMP_TVG are for gradients -- scale together
    for (int i = 2; i < NUM_TVG_FEATURES_PER_SUBWIN; i++)
      sum_sq_val += TMP_TVG[i]->SumSquare();
    for (int i = 2; i < NUM_TVG_FEATURES_PER_SUBWIN; i++){
      if (sum_sq_val < tvg_eps_thresh)
	*(TMP_TVG[i])=0.0;
      else
	*(TMP_TVG[i]) /= sqrt(sum_sq_val);
    }
    
    if (pass_id == 0){
      for (int i = 2; i < NUM_TVG_FEATURES_PER_SUBWIN; i++){
	for(int j = 0; j < TVG_FEATURE_DIMS/NUM_TVG_FEATURES_PER_SUBWIN; j++){
	  if (TMP_TVG[i]->element(j) > 0.2)
	    TMP_TVG[i]->element(j) = 0.2;
	}
      }
    }
  }
  
  for (int j = 0; j < NUM_TVG_FEATURES_PER_SUBWIN; j++)
    for(int i = 0; i < TVG_FEATURE_DIMS/NUM_TVG_FEATURES_PER_SUBWIN; i++){
      tvg_vec->element(j*TVG_FEATURE_DIMS/NUM_TVG_FEATURES_PER_SUBWIN+i) = TMP_TVG[j]->element(i);
  }
  
  // sum_sq_val = tvg_vec->SumSquare();
  // if (sum_sq_val < tvg_eps_thresh)
  //   *tvg_vec = 0.0;
  // else
  //   *tvg_vec /= sqrt(sum_sq_val);

  *tvg_vec *= 512.0;
  
  for(int i = 0; i < TVG_FEATURE_DIMS; i++){
    if(tvg_vec->element(i)>255.0)
      tvg_vec->element(i) = 255.0;
  }
  
}




// Computes SIFT descriptors for interest points in img using libsiftfast 
// (exact C++ implementation of lowe's sift program by zerofrog@gmail.com)
// input: img -- input  image
// output: key_points -- detected interest points and their descriptors
void computeSiftFeaturesFromIplImage(IplImage* img, vector<MyKeypoint*> & key_points){


  Image tmp_img = createImageFromIplImage(img);
  
  Keypoint keypts;
  float fproctime;
  
  releaseKeyPoints(key_points);
    
  keypts = GetKeypoints(tmp_img);
  
  MyKeypoint* one_keypoint;
  
  // write the keys to the output
  int numkeys = 0;
  Keypoint key = keypts;
  while(key) {
    one_keypoint = new MyKeypoint("SIFT", SIFT_FEATURE_DIMS);
    one_keypoint->row = key->row;
    one_keypoint->col = key->col;
    one_keypoint->fradius = float(double(key->scale)*SIFT_WINDOW_SCALE_FACTOR);
    one_keypoint->ori = key->ori;
    
    for(int i = 0; i < SIFT_FEATURE_DIMS; ++i) {
      int intdesc = (int)(key->descrip[i]*512.0f);
      
      assert( intdesc >= 0 );
      
      if( intdesc > 255 ){
	intdesc = 255;
      } 
      
      one_keypoint->fv->element(i) = intdesc;
    }
    
    key_points.push_back(one_keypoint);
    
    key = key->next;
  }
  
  FreeKeypoints(keypts);
  DestroyAllImages();
  DestroyAllResources();
}



// Computes texture descriptors based on geometric total variation (TVG) energy density of degree 1 and 2
// input: keypoints -- interest points (detected using libsiftfast)
//        o1, o2    -- tvg energy of degree 1 and 2
// output: tvg_keypoints   --  tvg descriptors for interest points
void computeTvgDescriptorsForFeatures(vector<MyKeypoint*> & keypoints, vector<MyKeypoint*> & tvg_keypoints, IplImage * o1, IplImage * o2){
  
  releaseKeyPoints(tvg_keypoints);
  
  string descriptor_name = "TVG2D";
  
  MyKeypoint * kp;
  
  CvSize frame_size = cvGetSize(o1);
  
  IplImage * integral_o1 = cvCreateImage(cvSize(frame_size.width+1, frame_size.height+1) , IPL_DEPTH_64F, 1 );
  IplImage * integral_o2 = cvCreateImage(cvSize(frame_size.width+1, frame_size.height+1) , IPL_DEPTH_64F, 1 );

  cvIntegral( o1, integral_o1);
  cvIntegral( o2, integral_o2);
  
  for(int nPts = 0; nPts < keypoints.size(); nPts++){
    kp = new MyKeypoint(descriptor_name, TVG_FEATURE_DIMS, keypoints[nPts]);
    buildTvgDescriptor(keypoints[nPts], kp->fv, o1, o2, integral_o1, integral_o2);
    tvg_keypoints.push_back(kp);
  }
  
  cvReleaseImage(&integral_o1);
  cvReleaseImage(&integral_o2);
}







