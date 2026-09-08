/**
 * capture_pcd.cpp — grab N seconds off the L2 over UDP and write one .pcd.
 *
 * Written for the smalldog repo: the file it produces is read directly by
 * 3d/tools/pcview.py, so a real capture can be drawn next to the cloud lidar.py
 * predicts for the same scene.  Accumulating several seconds is deliberate — the
 * L2's scan is non-repetitive, so a longer dwell keeps filling the field in, and
 * that fill is one of the things worth checking against the model.
 *
 *   ./capture_pcd out.pcd [seconds] [lidar_ip] [local_ip]
 */
#include "unitree_lidar_sdk.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace unilidar_sdk2;

int main(int argc, char *argv[])
{
    std::string out = (argc > 1) ? argv[1] : "capture.pcd";
    double secs     = (argc > 2) ? atof(argv[2]) : 3.0;
    std::string lidar_ip = (argc > 3) ? argv[3] : "192.168.1.62";
    std::string local_ip = (argc > 4) ? argv[4] : "192.168.1.2";

    UnitreeLidarReader *lreader = createUnitreeLidarReader();
    if (lreader->initializeUDP(6101, lidar_ip, 6201, local_ip)) {
        printf("initializeUDP failed (%s -> %s)\n", lidar_ip.c_str(), local_ip.c_str());
        return 1;
    }
    lreader->startLidarRotation();
    sleep(1);
    lreader->setLidarWorkMode(0);
    sleep(1);

    std::string vFirm, vHard, vSdk;
    for (int i = 0; i < 200000 && !lreader->getVersionOfLidarFirmware(vFirm); i++)
        lreader->runParse();
    lreader->getVersionOfLidarHardware(vHard);
    lreader->getVersionOfSDK(vSdk);
    printf("hardware %s   firmware %s   sdk %s\n",
           vHard.c_str(), vFirm.c_str(), vSdk.c_str());

    float dirty = -1.0f;
    for (int i = 0; i < 200000 && !lreader->getDirtyPercentage(dirty); i++)
        lreader->runParse();
    printf("dirty %.2f %%\n", dirty);

    std::vector<PointUnitree> all;
    PointCloudUnitree cloud;
    LidarImuData imu;
    int frames = 0, imus = 0;
    // The L2's own IMU, kept so the cloud can be levelled to gravity later.  A lidar
    // cloud in the sensor frame is nearly unreadable when the sensor is not upright:
    // walls lean, the floor is a diagonal, and colour-by-height means nothing.
    float q[4] = {0, 0, 0, 1};
    float acc[3] = {0, 0, 0};
    double t0 = getSystemTimeStamp(), t = t0;
    while ((t = getSystemTimeStamp()) - t0 < secs) {
        int r = lreader->runParse();
        if (r == LIDAR_POINT_DATA_PACKET_TYPE && lreader->getPointCloud(cloud)) {
            all.insert(all.end(), cloud.points.begin(), cloud.points.end());
            frames++;
        } else if (r == LIDAR_IMU_DATA_PACKET_TYPE && lreader->getImuData(imu)) {
            memcpy(q, imu.quaternion, sizeof(q));
            memcpy(acc, imu.linear_acceleration, sizeof(acc));
            imus++;
        }
    }
    printf("%d frames, %d imu msgs, %zu points in %.1f s (%.0f pts/s, %.1f frames/s)\n",
           frames, imus, all.size(), t - t0, all.size() / (t - t0), frames / (t - t0));

    FILE *f = fopen(out.c_str(), "w");
    if (!f) { printf("cannot write %s\n", out.c_str()); return 1; }
    fprintf(f, "# .PCD v0.7 - unitree L2, %d frames over %.1f s\n"
               "# imu_quaternion_xyzw %.6f %.6f %.6f %.6f\n"
               "# imu_accel_xyz %.4f %.4f %.4f\n"
               "VERSION 0.7\nFIELDS x y z intensity ring time\n"
               "SIZE 4 4 4 4 4 4\nTYPE F F F F U F\nCOUNT 1 1 1 1 1 1\n"
               "WIDTH %zu\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %zu\nDATA ascii\n",
            frames, t - t0, q[0], q[1], q[2], q[3], acc[0], acc[1], acc[2],
            all.size(), all.size());
    for (const auto &p : all)
        fprintf(f, "%.5f %.5f %.5f %.2f %u %.6f\n",
                p.x, p.y, p.z, p.intensity, p.ring, p.time);
    fclose(f);
    printf("wrote %s\n", out.c_str());
    return 0;
}
