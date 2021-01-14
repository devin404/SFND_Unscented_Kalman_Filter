#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>


using Eigen::MatrixXd;
using Eigen::VectorXd;

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
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3.5;
  
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

  // set state dimension
  n_x_ = 5;  //px, py, v, phi, phi_dot CTRV model

  n_aug_ = 7; //nx + 2 to account for process noise due to yaw and longitudinal acceleration

  // set spreading parameter
  lambda_ = 3 - n_aug_;

  //Initialize to zero
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1); //To avoid division by zero

  is_initialized_ = false;

  // Vector for weights
  weights = VectorXd(2*n_aug_+1);
  
  // set weights
  double weight_0 = lambda_/(lambda_ + n_aug_);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; ++i) {  // 2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights(i) = weight;
  }

  I_ = MatrixXd::Identity(n_x_, n_x_);
  H_ = MatrixXd(2,5);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;

  P_<< 1,0,0,0,0,
      0,1,0,0,0,
      0,0,1000,0,0,
      0,0,0,0.0225,0,
      0,0,0,0,0.0225;

  R_ = MatrixXd(2,2);
  R_<< std_laspx_*std_laspx_, 0,
    0, std_laspy_*std_laspy_; 
}

UKF::~UKF() {}

void UKF::initState(MeasurementPackage meas_package) {

  if(meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER && use_laser_) //LIDAR Px, Py
  {
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
      x_(2) = 0;
      x_(3) = 0;
      x_(4) = 0;
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
  
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR && use_radar_) //RADAR RHO, PHI, RHO_DOT
  {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);
      double vx = rho_dot*cos(phi);
      double vy = rho_dot*sin(phi);
      x_(0) = rho*cos(phi);    
      x_(1) = rho*sin(phi);
      x_(2) = sqrt(vx*vx+vy*vy);
      x_(3) = 0;
      x_(4) = 0;

      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;

  }

      
      
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if(!is_initialized_)
  {
    initState(meas_package);
  }
  else
  {

    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    Prediction(delta_t);

    if(meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER && use_laser_) //LIDAR Px, Py
    {
      UpdateLidar(meas_package);
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR && use_radar_) //RADAR RHO, PHI, RHO_DOT
    {
      UpdateRadar(meas_package);
    } 
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
  x_aug.head(n_x_) = x_;
  x_aug[5] = 0;
  x_aug[6] = 0;

  // create augmented covariance matrix
  P_aug.setZero();
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();    
  
  // create augmented sigma points
  A = A * sqrt(lambda_ + n_aug_);    
    
  Xsig_aug.col(0) = x_aug;

  for(int i = 0; i< n_aug_; i++)
  {
     Xsig_aug.col(1+i) = x_aug + A.col(i);
     Xsig_aug.col(1 + n_aug_ + i) = x_aug - A.col(i);
  }

}

void UKF::SigmaPointPrediction(MatrixXd& Xsig_aug, double delta_t) {

  // predict sigma points

  // avoid division by zero

  // write predicted sigma points into right column

  for(int i =0; i < 2 * n_aug_ +1; ++i)
  {
      double p_x = Xsig_aug(0,i);
      double p_y = Xsig_aug(1,i);
      double v = Xsig_aug(2,i);
      double yaw = Xsig_aug(3,i);
      double yaw_rate = Xsig_aug(4,i); 
      double nu_a = Xsig_aug(5,i); //noise due to acceleration
      double nu_yaw_rate = Xsig_aug(6,i); //noise due to yaw rate
      
      double px_p, py_p;


      if(fabs(yaw_rate) > 0.001)
      {
        px_p = p_x + (v * (sin(yaw + yaw_rate*delta_t) - sin(yaw)))/yaw_rate;
        py_p = p_y + (v * (cos(yaw) - cos(yaw + yaw_rate*delta_t)))/yaw_rate;
      }
      else
      {
        px_p = p_x + v*cos(yaw)*delta_t;
        py_p = p_y + v*sin(yaw)*delta_t;
      }
    
      double v_p = v;
      double yaw_p = yaw;
      double yaw_rate_p = yaw_rate;
      
      //Add noise

      px_p += 0.5*delta_t*delta_t*cos(yaw)*nu_a;
      py_p += 0.5*delta_t*delta_t*sin(yaw)*nu_a;
      v_p += nu_a*delta_t;
      yaw_p += yaw_rate*delta_t + 0.5*delta_t*delta_t*nu_yaw_rate;
      yaw_rate_p += delta_t*nu_yaw_rate;

      Xsig_pred_(0,i) = px_p;
      Xsig_pred_(1,i) = py_p;
      Xsig_pred_(2,i) = v_p;
      Xsig_pred_(3,i) = yaw_p;
      Xsig_pred_(4,i) = yaw_rate_p;
      
  }

}

void UKF::PredictMeanAndCovariance() {

  // predict state mean
  x_.fill(0);
  
  for(int i =0; i<Xsig_pred_.cols(); i++)
  {
    x_ += weights(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  
  P_.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights(i) * x_diff * x_diff.transpose() ;
  }

}


void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  MatrixXd Xsig_aug;
  //generateSigmaPoints

  AugmentedSigmaPoints(Xsig_aug);  //Xsig_out now contains sigma augmented matrix;

  SigmaPointPrediction(Xsig_aug, delta_t);  //Predicted Sigma points are stored in the function call

  PredictMeanAndCovariance();      //New Mean x and Covariance P is updated 

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  int n_z = 2;
  Eigen::VectorXd z = Eigen::VectorXd(n_z);
  z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);

  Eigen::VectorXd z_pred = H_ * x_;
  Eigen::VectorXd y = z - z_pred;
  Eigen::MatrixXd Ht = H_.transpose();
  Eigen::MatrixXd S = H_ * P_ * Ht + R_;
  Eigen::MatrixXd K = P_*Ht *S.inverse();

  //new estimate
  x_ = x_ + (K * y);
  while (x_(3)> M_PI) x_(3)-=2.*M_PI;
  while (x_(3)<-M_PI) x_(3)+=2.*M_PI;

  P_ = (I_ - K * H_) * P_;

}

void UKF::PredictRadarMeasurement(MatrixXd& Zsig, VectorXd& z_pred, MatrixXd& S, MatrixXd& R) {

  //Radar Measurement needs to be predicted in similar way as the measurement model is non Linear.

  // calculate mean predicted measurement
  z_pred.fill(0);
  for(int i =0; i< 2 * n_aug_ + 1;i++)
  {
      z_pred+= weights(i)*Zsig.col(i);
  }


  // calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
  
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  
    S = S + weights(i) * z_diff * z_diff.transpose();
  }
  
  S= S+R;
}

void UKF::UpdateState(MatrixXd& Zsig, VectorXd& z_pred, VectorXd& z, MatrixXd& S, int n_z) {

  // set measurement dimension, radar can measure r, phi, and r_dot

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);

  // calculate cross correlation matrix
  
  for(int i = 0; i< 2 * n_aug_ + 1; i++)
  {
      VectorXd Z_diff = Zsig.col(i)- z_pred;
      

      while (Z_diff(1)> M_PI) Z_diff(1)-=2.*M_PI;
      while (Z_diff(1)<-M_PI) Z_diff(1)+=2.*M_PI;
    
      VectorXd X_diff = Xsig_pred_.col(i) - x_;
      while (X_diff(3)> M_PI) X_diff(3)-=2.*M_PI;
      while (X_diff(3)<-M_PI) X_diff(3)+=2.*M_PI;
      
      Tc = Tc + weights(i)*X_diff*(Z_diff.transpose());
  }
  

  // calculate Kalman gain K;
  
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd Z_diff = z - z_pred;

  while (Z_diff(1)> M_PI) Z_diff(1)-=2.*M_PI;
  while (Z_diff(1)<-M_PI) Z_diff(1)+=2.*M_PI;

  x_ = x_ + K*(Z_diff);
  P_ = P_ - K*S*K.transpose();

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred(n_z);
 
  // measurement covariance matrix S
  MatrixXd S(n_z,n_z);

  // transform sigma points into measurement space

  for(int i =0; i< 2 * n_aug_ + 1; i++ )
  {
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig(1,i) = atan2(p_y,p_x);                                // phi
    Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  }

  MatrixXd R(n_z,n_z);
  R<< std_radr_*std_radr_, 0 ,0,
  0, std_radphi_*std_radphi_, 0,
  0, 0, std_radrd_*std_radrd_;

  VectorXd z(n_z);

  PredictRadarMeasurement(Zsig, z_pred, S, R);

  z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);
  UpdateState(Zsig, z_pred, z, S, n_z);

}

