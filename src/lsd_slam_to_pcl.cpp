#include "lsd_slam_to_pcl/lsd_slam_to_pcl.hpp"

#include <limits> // quiet_NaN

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PointIndices.h>
#include <pcl_ros/pcl_nodelet.h>
#include <geometry_msgs/PoseStamped.h>

LSDSLAMToPCL::LSDSLAMToPCL(ros::NodeHandle& nh, std::string& name) :
    nh_(nh),
    node_name_(name),
    sparsify_factor_(1),
    min_near_support_(7),
    scaled_depth_var_thresh_(0.001),
    abs_depth_var_thresh_(0.1)
{}

LSDSLAMToPCL::~LSDSLAMToPCL() {}

bool LSDSLAMToPCL::Init()
{
    depth_subscriber_ = nh_.subscribe("input", 10, &LSDSLAMToPCL::depthCB, this);
    cloud_publisher_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("output_points", 10);
    indices_publisher_ = nh_.advertise<pcl_ros::PCLNodelet::PointIndices>("output_indices", 10);
    pose_publisher_ = nh_.advertise<geometry_msgs::PoseStamped>("cam_pose",10);

    point_invalid_.x = std::numeric_limits<float>::quiet_NaN();
    point_invalid_.y = std::numeric_limits<float>::quiet_NaN();
    point_invalid_.z = std::numeric_limits<float>::quiet_NaN();
    point_invalid_.data[3] = 1.0f;


    ROS_INFO_STREAM_ONCE("LSD-SLAM to PCL Init finished");
    return true;
}

void LSDSLAMToPCL::depthCB(const lsd_slam_to_pcl::keyframeMsgConstPtr msg)
{
    ROS_INFO_STREAM_ONCE("LSD-SLAM to PCL depthCB start");
    ROS_INFO_STREAM_ONCE("sparsify_factor:" << sparsify_factor_ << "min_near_support_" << min_near_support_ << "scaled_" <<scaled_depth_var_thresh_<<"abs_"<<abs_depth_var_thresh_);
    if (msg->isKeyframe)
    {

        pcl::PointCloud<pcl::PointXYZRGB> cloud_pcl;
        pcl_ros::PCLNodelet::PointIndices indices_ros;
        
        const float fxi  = 1 / msg->fx;
        const float fyi  = 1 / msg->fy;
        const float cxi  = -msg->cx / msg->fx;
        const float cyi  = -msg->cy / msg->fy;
        
        const int width  = msg->width;
        const int height = msg->height;

        int num_pts_total = 0;

        boost::shared_ptr<InputPointDense> input_points(new InputPointDense[width * height]);

        Sophus::Sim3f cam_to_world = Sophus::Sim3f();

        memcpy(cam_to_world.data(), msg->camToWorld.data(), 7 * sizeof(float));
        memcpy(input_points.get(), msg->pointcloud.data(), width * height * sizeof(InputPointDense));

        Sophus::Quaternionf quat = cam_to_world.quaternion().cast<float>();
        Eigen::Vector3f trans = cam_to_world.translation().cast<float>();
        float length = quat.norm();
        geometry_msgs::PoseStamped pose;
        
        pose.header.frame_id = "/frame";
        pose.header.stamp = ros::Time::now();
        
        pose.pose.position.x = trans[0];
        pose.pose.position.y = trans[1];
        pose.pose.position.z = trans[2];
        ROS_INFO_STREAM("xyzw:" << quat.x()<<","<<quat.y()<<","<<quat.z()<< "," <<quat.w());
        pose.pose.orientation.x = quat.x()/length;
        pose.pose.orientation.y = quat.y()/length;
        pose.pose.orientation.z = quat.z()/length;
        pose.pose.orientation.w = quat.w()/length;

        const float cam_to_world_scale = cam_to_world.scale();

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {

                // Catch boundary points
                if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
                {
                    cloud_pcl.push_back(point_invalid_);
                    continue;
                }

                int point_indice = x + y * width;

                const InputPointDense point_in_curr = input_points.get()[point_indice];

                if (point_in_curr.idepth <= 0)
                {
                    cloud_pcl.push_back(point_invalid_);
                    continue;
                }

                num_pts_total++;

                if (sparsify_factor_ > 1 && rand() % sparsify_factor_ != 0)
                {
                    cloud_pcl.push_back(point_invalid_);
                    continue;
                }

                const float depth = 1 / point_in_curr.idepth;
                const float depth4 = depth * depth * depth * depth;

                if (point_in_curr.idepth_var * depth4 > scaled_depth_var_thresh_)
                {
                    cloud_pcl.push_back(point_invalid_);
                    continue;
                }

                if (point_in_curr.idepth_var * depth4 * cam_to_world_scale * cam_to_world_scale > abs_depth_var_thresh_)
                {
                    cloud_pcl.push_back(point_invalid_);
                    continue;
                }

                if (min_near_support_ > 1)
                {
                    int near_support = 0;

                    for (int dx = -1; dx < 2; ++dx)
                    {
                        for (int dy = -1; dy < 2; ++dy)
                        {

                            const InputPointDense point_in_near = input_points.get()[x + dx + (y + dy) * width];

                            if (point_in_near.idepth > 0)
                            {
                                const float diff = point_in_near.idepth - point_in_curr.idepth;
                                if (diff * diff < 2 * point_in_curr.idepth_var) { near_support++; }
                            }
                        }
                    }

                    if (near_support < min_near_support_)
                    {
                        cloud_pcl.push_back(point_invalid_);
                        continue;
                    }
                }

                Eigen::Vector3f point_eigen = cam_to_world * (Eigen::Vector3f((x * fxi + cxi), (y * fyi + cyi), 1) * depth);

                pcl::PointXYZRGB point_pcl;
                point_pcl.x = point_eigen[0];
                point_pcl.y = point_eigen[1];
                point_pcl.z = point_eigen[2];
                point_pcl.r = point_in_curr.color[2];
                point_pcl.g = point_in_curr.color[1];
                point_pcl.b = point_in_curr.color[0];
                cloud_pcl.push_back(point_pcl);
                indices_ros.indices.push_back(point_indice);

            }
        }

        // uint64_t time_stamp = ros::Time::now().toNSec();

        cloud_pcl.width = width;
        cloud_pcl.height = height;
        cloud_pcl.is_dense = false;
        // indices_ros.header.stamp = pcl_conversions::fromPCL(time_stamp);
        cloud_pcl.header.frame_id = "frame";
        pcl_conversions::toPCL(ros::Time::now(), cloud_pcl.header.stamp);
        indices_ros.header.stamp = ros::Time::now();

        cloud_publisher_.publish(boost::make_shared<const pcl::PointCloud<pcl::PointXYZRGB> >(cloud_pcl));
        indices_publisher_.publish(boost::make_shared<const pcl_ros::PCLNodelet::PointIndices>(indices_ros));
        pose_publisher_.publish(pose);

        ROS_INFO_STREAM("Published " << num_pts_total << " points to pointcloud, dimensions [" << cloud_pcl.width << " " << cloud_pcl.height << "]");

    }

    else
    {
        ROS_INFO_STREAM("Error, must subscribe to keyframe");
    }

}