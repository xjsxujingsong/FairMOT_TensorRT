#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include "dataType.h"

class KalmanFilterTracking
{
    public:
        static const double chi2inv95[10];
        KalmanFilterTracking(const double& dt = 1.);
        KAL_DATA initiate(const DETECTBOX& measurement);
        void predict(KAL_MEAN& mean, KAL_COVA& covariance);
        KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);
        KAL_DATA update(const KAL_MEAN& mean,
                        const KAL_COVA& covariance,
                        const DETECTBOX& measurement);

        Eigen::Matrix<float, 1, -1> gating_distance(
                const KAL_MEAN& mean,
                const KAL_COVA& covariance,
                const std::vector<DETECTBOX>& measurements,
                bool only_position = false);

        void set_dt(const float& dt);

    private:
        Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
        Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
        float _std_weight_position;
        float _std_weight_velocity;
};

#endif // KALMANFILTER_H
