#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <librealsense2/rs.hpp>
#include <cv_bridge/cv_bridge.h>

ros::Publisher pub_point;
ros::Time temp_time = ros::Time(0);

// 全局变量存储话题名（也可封装为类，此处简化）
std::string sub_time_topic;   // 时间戳订阅话题
std::string sub_yolo_topic;   // YOLO点云订阅话题
std::string pub_point_topic; // 点coord发布话题

void yoloCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {

    if (temp_time.is_zero()){
        ROS_INFO("not get time");
        return;
    }
    // 解析Float32MultiArray中的数据（x、y、z）
    float x = msg->data[0];
    float y = msg->data[1];
    float z = msg->data[2];

    // 构造Header（C++中无只读限制）
    std_msgs::Header header;
    header.stamp = temp_time;
    header.frame_id = "map";

    // 构造PointStamped并发布
    geometry_msgs::PointStamped point_msg;
    point_msg.header = header;
    point_msg.point.x = x;
    point_msg.point.y = y;
    point_msg.point.z = z;
    pub_point.publish(point_msg);
    ROS_INFO("success add header");

}

void call_time_back(const sensor_msgs::PointCloud2::ConstPtr& msg){
    temp_time = msg->header.stamp;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "point_publisher_node");
    ros::NodeHandle nh("~"); // 使用私有命名空间，方便launch文件传参

    // 从参数服务器读取话题名，若未读取到则使用默认值
    nh.param<std::string>("sub_time_topic", sub_time_topic, "/kfs_pointcloud");
    nh.param<std::string>("sub_yolo_topic", sub_yolo_topic, "/center_coord");
    nh.param<std::string>("pub_point_topic", pub_point_topic, "/center_point_with_header");

    // 初始化发布者
    pub_point = nh.advertise<geometry_msgs::PointStamped>(pub_point_topic, 10);

    // 初始化订阅者
    ros::Subscriber sub_time = nh.subscribe<sensor_msgs::PointCloud2>(sub_time_topic, 10, call_time_back);
    ros::Subscriber sub = nh.subscribe(sub_yolo_topic, 10, yoloCallback);

    ROS_INFO("Topic config:");
    ROS_INFO("  sub_time_topic: %s", sub_time_topic.c_str());
    ROS_INFO("  sub_yolo_topic: %s", sub_yolo_topic.c_str());
    ROS_INFO("  pub_point_topic: %s", pub_point_topic.c_str());

    ros::spin();
    return 0;
}





// #include <ros/ros.h>
// #include <geometry_msgs/PointStamped.h>
// #include <std_msgs/Float32MultiArray.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <librealsense2/rs.hpp>
// #include <cv_bridge/cv_bridge.h>

// ros::Publisher pub_point;
// ros::Publisher pub_camera;
// ros::Time temp_time = ros::Time(0);




// void yoloCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {

//     if (temp_time.is_zero()){
//         ROS_INFO("not get time");
//         return;
//     }
//     // 解析Float32MultiArray中的数据（x、y、z）
//     float x = msg->data[0];
//     float y = msg->data[1];
//     float z = msg->data[2];

//     // 构造Header（C++中无只读限制）
//     std_msgs::Header header;
//     header.stamp = temp_time;
//     header.frame_id = "map";

//     // 构造PointStamped并发布
//     geometry_msgs::PointStamped point_msg;
//     point_msg.header = header;
//     point_msg.point.x = x;
//     point_msg.point.y = y;
//     point_msg.point.z = z;
//     pub_point.publish(point_msg);
//     ROS_INFO("success add header");

// }

// void call_time_back(const sensor_msgs::PointCloud2::ConstPtr& msg){
//     temp_time = msg->header.stamp;
// }



// int main(int argc, char** argv) {
//     ros::init(argc, argv, "point_publisher_node");
//     ros::NodeHandle nh;
//     pub_point = nh.advertise<geometry_msgs::PointStamped>("/PointXYZ", 10);


//     // 先调用 pc_node -> get 时间戳
//     ros::Subscriber sub_time = nh.subscribe<sensor_msgs::PointCloud2>("/realsense_pc",10,call_time_back);
//     ros::Subscriber sub = nh.subscribe("/point_no_header", 10, yoloCallback);
    

//     ros::spin();
//     return 0;
// }
