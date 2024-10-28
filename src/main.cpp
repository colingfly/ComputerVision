#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <DBoW2/DBoW2.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/linear_solver_dense.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <Open3D/Open3D.h>

// Function to perform pose graph optimization
void optimizePoseGraph(std::vector<cv::Mat>& poses) {
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    auto linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverTraits<6, 6>::PoseMatrixType>>();
    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver)));
    optimizer.setAlgorithm(solver);

    for (size_t i = 0; i < poses.size(); i++) {
        g2o::VertexSE3* v = new g2o::VertexSE3();
        v->setId(i);
        v->setEstimate(g2o::SE3Quat(poses[i]));
        optimizer.addVertex(v);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    for (size_t i = 0; i < poses.size(); i++) {
        auto* v = static_cast<g2o::VertexSE3*>(optimizer.vertex(i));
        poses[i] = v->estimate().matrix();
    }
}

// Function to display the point cloud
void displayPointCloud(const std::vector<cv::Point3f>& pointCloud) {
    open3d::geometry::PointCloud pcd;
    for (const auto& pt : pointCloud) {
        pcd.points_.push_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
    }

    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("3D SLAM", 800, 600);
    vis.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(pcd));
    vis.Run();
    vis.DestroyVisualizerWindow();
}

int main() {
    std::string leftPath = "/Users/colingibbons-fly/SLAM_Project/data/image_00/data/";
    std::string rightPath = "/Users/colingibbons-fly/SLAM_Project/data/image_01/data/";
    
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 15);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);

    cv::Mat prevGray, gray;
    std::vector<cv::Point2f> prevPoints, nextPoints;
    cv::Mat R_f = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_f = cv::Mat::zeros(3, 1, CV_64F);
    bool firstFrame = true;
    std::vector<cv::Mat> poses;
    
    DBoW2::ORBDatabase db;
    std::vector<cv::Mat> descriptors;

    for (const auto& entryLeft : std::filesystem::directory_iterator(leftPath)) {
        std::string leftImageFile = entryLeft.path().string();
        std::string rightImageFile = rightPath + entryLeft.path().filename().string();
        cv::Mat leftImage = cv::imread(leftImageFile, cv::IMREAD_GRAYSCALE);
        cv::Mat rightImage = cv::imread(rightImageFile, cv::IMREAD_GRAYSCALE);
        
        if (leftImage.empty() || rightImage.empty()) continue;

        cv::Mat disparity, disparityNorm;
        stereo->compute(leftImage, rightImage, disparity);
        cv::normalize(disparity, disparityNorm, 0, 255, cv::NORM_MINMAX, CV_8U);

        gray = leftImage.clone();

        if (firstFrame) {
            std::vector<cv::KeyPoint> keypoints;
            orb->detect(gray, keypoints);
            cv::KeyPoint::convert(keypoints, prevPoints);
            firstFrame = false;
        } else {
            std::vector<uchar> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(prevGray, gray, prevPoints, nextPoints, status, err);

            std::vector<cv::Point2f> prevPointsFiltered, nextPointsFiltered;
            for (size_t i = 0; i < nextPoints.size(); i++) {
                if (status[i]) {
                    prevPointsFiltered.push_back(prevPoints[i]);
                    nextPointsFiltered.push_back(nextPoints[i]);
                }
            }

            cv::Mat E = cv::findEssentialMat(nextPointsFiltered, prevPointsFiltered, 700, cv::Point2d(640 / 2, 480 / 2), cv::RANSAC, 0.999, 1.0, status);
            cv::Mat R, t;
            if (!E.empty() && E.cols == 3 && E.rows == 3) {
                cv::recoverPose(E, nextPointsFiltered, prevPointsFiltered, R, t, 700, cv::Point2d(640 / 2, 480 / 2));
                R_f = R * R_f;
                t_f = t_f + R_f * t;
                poses.push_back(R_f.clone());  // Store pose for optimization
            }

            for (size_t i = 0; i < nextPointsFiltered.size(); i++) {
                cv::line(leftImage, prevPointsFiltered[i], nextPointsFiltered[i], cv::Scalar(0, 255, 0), 2);
                cv::circle(leftImage, nextPointsFiltered[i], 3, cv::Scalar(0, 0, 255), -1);
            }

            prevPoints = nextPointsFiltered;
        }

        prevGray = gray.clone();
        cv::imshow("Disparity", disparityNorm);
        cv::imshow("Feature Tracking", leftImage);

        // Loop closure detection
        cv::Mat descriptorsCurrent;
        std::vector<cv::KeyPoint> keypointsCurrent;
        orb->detectAndCompute(gray, cv::Mat(), keypointsCurrent, descriptorsCurrent);

        int loopIndex = db.query(descriptorsCurrent);
        if (loopIndex != -1) {
            std::cout << "Loop closure detected with frame " << loopIndex << std::endl;
            optimizePoseGraph(poses);  // Optimize the pose graph on loop closure
        }

        db.add(descriptorsCurrent);  // Add to BoW database
        descriptors.push_back(descriptorsCurrent);

        if (cv::waitKey(30) == 'q') break;
    }

    cv::destroyAllWindows();
    return 0;
}