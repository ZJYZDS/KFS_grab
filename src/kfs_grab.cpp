#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/extract_indices.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <serial/serial.h>
#include <Eigen/Dense>
#include <cmath>


using namespace std;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// 串口配置
#define SERIAL_PORT   "/dev/ttyUSB0"
#define SERIAL_BAUDRATE 115200
serial::Serial ser;

ros::Publisher pub_plane_points;
ros::Publisher pub_center;
int a ;

// float x = 0.0f;
// float y = 0.0f;
// float z = 0.0f;

// 计算中心点
PointT computeCloudCenter(const PointCloudT::Ptr& cloud)
{
    if(cloud->empty())
    {
         ROS_WARN("center_cloud is empty");
    }
    PointT center;
    float sum_x = 0, sum_y = 0, sum_z = 0;
    int n = cloud->size();
    for (auto& p : cloud->points) {
        sum_x += p.x; sum_y += p.y; sum_z += p.z;
    }
    center.x = sum_x / n;
    center.y = sum_y / n;
    center.z = sum_z / n;
    return center;
}

// 初始化串口
bool initSerial()
{
    try {
        ser.setPort(SERIAL_PORT);
        ser.setBaudrate(SERIAL_BAUDRATE);
        serial::Timeout timeout = serial::Timeout::simpleTimeout(100);
        ser.setTimeout(timeout);
        ser.open();
    } catch (...) {
        ROS_WARN("serial open defect");
        return false;
    }
    return ser.isOpen();
}

// 发送数据
void sendCenterToSerial(float x, float y, float z)
{
    if (!ser.isOpen()) return;
    float data[3] = {x*1000,y*1000,z*1000};
    uint8_t ser_data[16];
    
    memcpy(ser_data+2 , (uint8_t*)data , 12);
    ser_data[0] = 0x0B;
    ser_data[1] = 0x1C;
    ser_data[14] = 0x1A;
    ser_data[15] = 0x0C;
    ser.write(ser_data , sizeof(ser_data));
    ROS_INFO("ser message send success");
}
// 角度转弧度
inline double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

// 相机坐标系：
// 先绕 Z 逆时针 15° → 再绕 Y 逆时针 20°
Eigen::Vector3d rotatePointCameraFrame(const Eigen::Vector3d& p , int& model_check)
{
    // // 旋转角度
    // double theta_z = deg2rad(15.0);  // 绕Z逆时针15°
    // double theta_y = deg2rad(20.0);  // 绕Y逆时针20°

    // // 绕 Z 轴旋转矩阵
    // Eigen::Matrix3d Rz;
    // Rz << cos(theta_z), -sin(theta_z), 0,
    //       sin(theta_z),  cos(theta_z), 0,
    //       0,             0,            1;

    // // 绕 Y 轴旋转矩阵
    // Eigen::Matrix3d Ry;
    // Ry << cos(theta_y),  0, sin(theta_y),
    //       0,             1, 0,
    //       -sin(theta_y), 0, cos(theta_y);

    

    // // 组合旋转：先Z 后Y 👉 R = Ry * Rz
    // Eigen::Matrix3d R = Ry * Rz;
    // // Eigen::Matrix3d R = Ry;
    // // 组合旋转：先Z 后Y
    // // R = Ry * Rz;
    // // camera link -> base link (1)
    Eigen::Vector3d p_rot = p;

    // ==================== 2. 坐标映射 (x,y,z) → (y, -z, x) ====================
    double x = p_rot.x();
    double y = p_rot.y();
    double z = p_rot.z();

    Eigen::Vector3d p_final;
    p_final.x() = z;       // 新 x = 原 y
    p_final.y() = x;      // 新 y = -原 z
    p_final.z() = -y;       // 新 z = 原 x

    //camera link -> base link (2)
    p_final.z() += 0.170;
    p_final.y() += 0.02;
    p_final.x() += 0.120;

    if (model_check == 1)
    {
        p_final.z() -= 0.140;
        p_final.x() -= 0.160;
    }
    

    ROS_INFO("x%.2f , y%.2f , z%.2f" , p_final.x(),p_final.y(),p_final.z());

    return p_final;
}

// 回调函数
void callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
              const geometry_msgs::PointStampedConstPtr& seed_msg)
{
    int model_check = -1;
    // 1. 点云转换
    PointCloudT::Ptr cloud(new PointCloudT);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // 2. seed点（仅用来定位大范围区域）
    PointT seed;
    seed.x = seed_msg->point.x / 1000.0f;
    seed.y = seed_msg->point.y / 1000.0f;
    seed.z = seed_msg->point.z / 1000.0f;

    // ====================== 关键：半径搜索，获取大片平面点 ======================
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    vector<int> indices;
    vector<float> distances;
    float search_radius = 0.12; 
    kdtree.radiusSearch(seed, search_radius, indices, distances);

    // 提取大片点云
    PointCloudT::Ptr neighbor_cloud(new PointCloudT);
    pcl::copyPointCloud(*cloud, indices, *neighbor_cloud);

    if (neighbor_cloud->size() < 400) {
        ROS_WARN("points num too low");
        return;
    }

    // ====================== 大范围平面拟合 ======================
    pcl::SampleConsensusModelPlane<PointT>::Ptr model(new pcl::SampleConsensusModelPlane<PointT>(neighbor_cloud));
    pcl::RandomSampleConsensus<PointT> ransac(model);
    ransac.setDistanceThreshold(0.01); // 1cm 容错
    ransac.computeModel();

    vector<int> inliers;
    ransac.getInliers(inliers);

    if (inliers.size() < 30) {
        ROS_WARN("plane seem defect");
        return;
    }

    // ====================== 法线判断 ======================
    Eigen::VectorXf coeff;
    ransac.getModelCoefficients(coeff);
    Eigen::Vector3f normal(coeff[0], coeff[1], coeff[2]);
    normal.normalize();

    Eigen::Vector3f Y_normal(0, -1, 0); // top平面法线方向
    float dot_top = normal.dot(Y_normal);
    dot_top = std::abs(dot_top);

    Eigen::Vector3f Z_normal(0,0,1); // 正面法线方向
    float dot_front = normal.dot(Z_normal);
    dot_front = std::abs(dot_front);

    if (dot_top < 0.7 && dot_front < 0.8) {
        ROS_WARN("Normal is not align to top face and front face");
        return;
    }else if (dot_front > 0.7 && dot_front < 0.6)
    {
        model_check = 1;
        ROS_INFO("Get top face , dot :%.3f",dot_top);
        // model -> 1 -> top face
    }else if(dot_top < 0.6 && dot_front > 0.8)
    {
        model_check = 2;
        ROS_INFO("Get front face , dot: %,3f",dot_front);
        //model -> 2 -> front face
    }


    // ====================== 提取大平面点云 ======================
    PointCloudT::Ptr plane_cloud(new PointCloudT);
    pcl::copyPointCloud(*neighbor_cloud, inliers, *plane_cloud);

    // 计算中心
    PointT center = computeCloudCenter(plane_cloud);
    ROS_INFO("center point coord_src：x=%.3f y=%.3f z=%.3f  num_poins：%d",
             center.x, center.y, center.z, (int)plane_cloud->size());

    

    // 发布
    sensor_msgs::PointCloud2 plane_msg;
    pcl::toROSMsg(*plane_cloud, plane_msg);
    plane_msg.header = cloud_msg->header;
    pub_plane_points.publish(plane_msg);

    geometry_msgs::PointStamped center_msg;
    center_msg.header = cloud_msg->header;

    Eigen::Vector3d c_v = {center.x , center.y , center.z};
    if(model_check <= 0)
    {
        ROS_WARN("not valid Norm  top_dot:%.2f  fromt_dot:%.2f, skip this frame",dot_top,dot_front);
        return;
    }

    Eigen::Vector3d new_c_v = rotatePointCameraFrame(c_v, model_check);
    center_msg.point.x = new_c_v.x();
    center_msg.point.y = new_c_v.y();
    center_msg.point.z = new_c_v.z();
   
    pub_center.publish(center_msg);

    // 发送串口
    if (a == 0) sendCenterToSerial(new_c_v.x(), new_c_v.y(),new_c_v.z());
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "big_plane_fit");
    ros::NodeHandle nh;

    cout << "输入0进入串口模式：";
    cin >> a;
    if (a == 0) initSerial();

    // 同步订阅
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud(nh, "/kfs_pointcloud", 10);
    message_filters::Subscriber<geometry_msgs::PointStamped> sub_seed(nh, "/center_point_with_header", 10);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, geometry_msgs::PointStamped> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_cloud, sub_seed);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    pub_plane_points = nh.advertise<sensor_msgs::PointCloud2>("/plane_points", 1);
    pub_center = nh.advertise<geometry_msgs::PointStamped>("/plane_center", 1);

    ROS_INFO("plane seem node start");
    ros::spin();
    return 0;
}
