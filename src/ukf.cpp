#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

// visualize the NIS
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using Eigen::MatrixXd;
using Eigen::VectorXd;
std::vector<double> UKF::lidar_nis_; 
std::vector<double> UKF::radar_nis_; 
/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 10;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  // set vector for weights
  weights_ = VectorXd(2*n_aug_+1);
  weights_.setOnes();
  weights_ *= 0.5/(lambda_+n_aug_);
  weights_(0) = lambda_/(lambda_+n_aug_);
  P_ = MatrixXd(n_x_, n_x_);
  P_.setIdentity();
  P_(0,0) = std_laspx_ * std_laspx_;
  P_(1,1) = std_laspy_ * std_laspy_;
  P_(2,2) = std_radr_ * std_radr_;
  P_(3,3) = std_radphi_ * std_radphi_;
  P_(4,4) = std_radrd_ * std_radrd_;
  
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if(!is_initialized_){
    // initialize state and covariance matrixs
    if(meas_package.sensor_type_ == meas_package.LASER){
      x_.head<2>() = meas_package.raw_measurements_.head<2>();
      x_.tail<3>().setZero();
    }
    else if(meas_package.sensor_type_ == meas_package.RADAR){
      // radar can measur r, phi, r_dot
      double r = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double r_dot = meas_package.raw_measurements_(2);
      x_(0) = r * std::cos(phi);
      x_(1) = r * std::sin(phi);
      x_(2) = r_dot; // actually r_dot is not the vel_abs
      x_(3) = phi;
      x_(4) = 0.0;
    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
  }
  else{
    // predict then update
    double delta_t = static_cast<double>(meas_package.timestamp_ - time_us_) / 1e6;
    
    
    if(meas_package.sensor_type_ == meas_package.LASER && use_laser_){
      Prediction(delta_t);
      UpdateLidar(meas_package);
      time_us_ = meas_package.timestamp_;
    }
    else if(meas_package.sensor_type_ == meas_package.RADAR && use_radar_){
      Prediction(delta_t);
      UpdateRadar(meas_package);
      time_us_ = meas_package.timestamp_;
    }
    
  }

  
}

void UKF::SigmaPointPrediction(MatrixXd& Xsig_aug, double delta_t) {


  // create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.setZero();
  // predict sigma points
  for(size_t i=0; i<2*n_aug_+1; i++){
    if(fabs(Xsig_aug(4,i)) < 1e-8){
        // avoid division by zero
        Xsig_pred_(0,i) = Xsig_aug(0,i) + Xsig_aug(2,i)*cos(Xsig_aug(3,i))*delta_t;
        Xsig_pred_(1,i) = Xsig_aug(1,i) + Xsig_aug(2,i)*sin(Xsig_aug(3,i))*delta_t;
        
    }
    else{
        Xsig_pred_(0,i) = Xsig_aug(0,i) + Xsig_aug(2,i) / Xsig_aug(4,i) * (sin(Xsig_aug(3,i) + Xsig_aug(4,i) * delta_t) - sin(Xsig_aug(3,i)));
        Xsig_pred_(1,i) = Xsig_aug(1,i) + Xsig_aug(2,i) / Xsig_aug(4,i) * (-cos(Xsig_aug(3,i) + Xsig_aug(4,i) * delta_t) + cos(Xsig_aug(3,i)));
    }
    Xsig_pred_(2,i) = Xsig_aug(2,i);
    Xsig_pred_(3,i) = Xsig_aug(3,i) + Xsig_aug(4,i)*delta_t;
    Xsig_pred_(4,i) = Xsig_aug(4,i);
    
    Xsig_pred_(0,i) += 0.5 * pow(delta_t,2) * cos(Xsig_aug(3,i)) * Xsig_aug(5,i);
    Xsig_pred_(1,i) += 0.5 * pow(delta_t,2) * sin(Xsig_aug(3,i)) * Xsig_aug(5,i);
    Xsig_pred_(2,i) += delta_t * Xsig_aug(5,i);
    Xsig_pred_(3,i) += 0.5 * pow(delta_t,2) * Xsig_aug(6,i);
    Xsig_pred_(4,i) += delta_t * Xsig_aug(6,i);

  }
}
void UKF::PredictMeanAndCovariance(VectorXd& x_pred, MatrixXd& P_pred) {


  // create vector for predicted state
  x_pred = VectorXd(n_x_);

  // create covariance matrix for prediction
  P_pred = MatrixXd(n_x_, n_x_);

  // predict state mean
  x_pred.setZero();
  for(size_t i=0; i<2*n_aug_+1; ++i){
      x_pred += weights_(i) * Xsig_pred_.col(i);
  }
  
  // predict state covariance matrix
  P_pred.setZero();
  for(size_t i=0; i<2*n_aug_+1; ++i){
      P_pred += weights_(i) * (Xsig_pred_.col(i) - x_pred) * (Xsig_pred_.col(i) -x_pred).transpose();
  }

}
void UKF::AugmentedSigmaPoints(MatrixXd& Xsig_aug) {

  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // create sigma point matrix
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented mean state
  x_aug.setZero();
  x_aug.head(n_x_) = x_;
  
  // create augmented covariance matrix
  P_aug.setZero();
  P_aug.block(0,0,n_x_,n_x_) = P_;
  P_aug(n_x_, n_x_) = pow(std_a_,2);
  P_aug(n_x_+1, n_x_+1) = pow(std_yawdd_,2);
  
  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  // create augmented sigma points
  for(size_t i=0; i < 2*n_aug_+1; i++){
      Xsig_aug.col(i) = x_aug;
  }
  Xsig_aug.block(0,1, n_aug_, n_aug_) += sqrt(lambda_ + n_aug_) * A;
  Xsig_aug.block(0,n_aug_+1,n_aug_, n_aug_) -= sqrt(lambda_ + n_aug_) * A;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  MatrixXd Xsig_aug;
  AugmentedSigmaPoints(Xsig_aug);
  
  SigmaPointPrediction(Xsig_aug, delta_t);
  VectorXd x_pred; MatrixXd P_pred;
  PredictMeanAndCovariance(x_pred, P_pred);
  x_ = x_pred;
  P_ = P_pred;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  VectorXd z_pred; MatrixXd Zsig, S;
  PredictLidarMeasurement(z_pred, Zsig, S);
  UpdateState(Zsig, z_pred, S, meas_package.raw_measurements_);

  double NIS = computeNIS(meas_package.raw_measurements_, z_pred, S);
  //std::cout<<"Lidar NIS: "<<NIS<<std::endl;
  UKF::lidar_nis_.push_back(NIS);
}




void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  VectorXd z_pred; MatrixXd Zsig, S;
  PredictRadarMeasurement(z_pred, Zsig, S);
  UpdateState(Zsig, z_pred, S, meas_package.raw_measurements_);

  double NIS = computeNIS(meas_package.raw_measurements_, z_pred, S);
  //std::cout<<"Radar NIS: "<<NIS<<std::endl;
  UKF::radar_nis_.push_back(NIS);
}

double UKF::computeNIS(VectorXd& measurement_z, VectorXd& z_pred, MatrixXd& S){
  // normalized square root (NIS)
  return (measurement_z - z_pred).transpose() * S.inverse() * (measurement_z - z_pred);
}

void UKF::PredictRadarMeasurement(VectorXd& z_pred, MatrixXd& Zsig, MatrixXd& S) {

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;


  // create matrix for sigma points in measurement space
  Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  S = MatrixXd(n_z,n_z);

  // transform sigma points into measurement space
  for(size_t i=0; i<2*n_aug_ + 1; ++i){
      Zsig(0,i) = sqrt(pow(Xsig_pred_(0,i), 2) + pow(Xsig_pred_(1,i), 2));
      Zsig(1,i) = atan2(Xsig_pred_(1,i), Xsig_pred_(0,i));
      Zsig(2,i) = (Xsig_pred_(0,i) * cos(Xsig_pred_(3,i)) * Xsig_pred_(2,i) + Xsig_pred_(1,i) * sin(Xsig_pred_(3,i)) * Xsig_pred_(2,i) ) / sqrt(pow(Xsig_pred_(0,i), 2) + pow(Xsig_pred_(1,i), 2));
      
  }
  // calculate mean predicted measurement
  
  z_pred.setZero();
  for(size_t i=0; i<2*n_aug_+1;++i){
      z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  S.setZero();
  for(size_t i=0; i<2*n_aug_+1; ++i){
      S += weights_(i) * (Zsig.col(i) - z_pred) * (Zsig.col(i) - z_pred).transpose();
  }
  S(0,0) += std_radr_ * std_radr_;
  S(1,1) += std_radphi_ * std_radphi_;
  S(2,2) += std_radrd_ * std_radrd_;
  
}


void UKF::PredictLidarMeasurement(VectorXd& z_pred, MatrixXd& Zsig, MatrixXd& S) {

  // set measurement dimension, radar can measure pos1, pos2 (x, y)
  int n_z = 2;

  // mean predicted measurement
  z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  S = MatrixXd(n_z,n_z);

  // transform sigma points into measurement space
  Zsig = Xsig_pred_.block(0, 0, 2, 2*n_aug_+1);
  
  // calculate mean predicted measurement
  z_pred.setZero();
  for(size_t i=0; i<2*n_aug_+1;++i){
      z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  S.setZero();
  for(size_t i=0; i<2*n_aug_+1; ++i){
      S += weights_(i) * (Zsig.col(i) - z_pred) * (Zsig.col(i) - z_pred).transpose();
  }
  S(0,0) += std_laspx_ * std_laspx_;
  S(1,1) += std_laspy_ * std_laspy_;
  
}

void UKF::UpdateState(MatrixXd& Zsig, VectorXd& z_pred,MatrixXd& S, VectorXd& mesuarement_z) {
  int n_z = z_pred.size();
  assert(n_z == mesuarement_z.size());

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);


  // calculate cross correlation matrix
  Tc.setZero();
  for(size_t i=0; i<2 * n_aug_ + 1; ++i){
    Tc += weights_(i) * (Xsig_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
  } 
  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  // update state mean and covariance matrix
  x_ += K*(mesuarement_z - z_pred);
  P_ -= K * S * K.transpose();
}

void UKF::PlotNIS(){
  plt::figure_size(1200, 780);
  unsigned int size = std::min(UKF::lidar_nis_.size(), UKF::radar_nis_.size());

  std::cout<<"size: " <<size<<std::endl;
  unsigned int start_idx = 10;
  std::vector<double> lidar_nis(UKF::lidar_nis_.begin() + start_idx, UKF::lidar_nis_.begin() + size);
  std::vector<double> radar_nis(UKF::radar_nis_.begin() + start_idx, UKF::radar_nis_.begin() + size);
  std::vector<double> axis_x(size - start_idx);
  std::vector<double> lidar_nis_tresh(size - start_idx, 5.991);   // 5% for 2 dimensions
  std::vector<double> radar_nis_tresh(size - start_idx, 7.815);   // 5% for 3 dimensions
  for(unsigned int i=0; i<axis_x.size(); i++){
    axis_x[i] = static_cast<double>(i + start_idx);
  }
  
  plt::named_plot("lidar NIS",axis_x, lidar_nis, "b");
  plt::named_plot("radar NIS",axis_x, radar_nis, "r");

  plt::named_plot("lidar 5% line", axis_x, lidar_nis_tresh, "b--");
  plt::named_plot("radar 5% line", axis_x, radar_nis_tresh, "r--");
  /*
  plt::plot(axis_x, lidar_nis, "b");
  plt::plot(axis_x, radar_nis, "r");

  plt::plot(axis_x, lidar_nis_tresh, "b--");
  plt::plot(axis_x, radar_nis_tresh, "r--");
  */
  plt::title("NIS");
  plt::legend();
  plt::show();
}