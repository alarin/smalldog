/**
 * stream_pcd.cpp — serve the L2's cloud over TCP, frame by frame, for a live viewer.
 *
 * The companion to capture_pcd.cpp: same SDK, same UDP input, but instead of accumulating
 * to a file it hands each parsed frame straight to a TCP client.  3d/tools/pcview.py
 * --stream connects to it and redraws.  A server rather than a client because the machine
 * with the lidar on it is the one that stays put: the mac connects to the Pi, so nothing
 * has to know the viewer's address and no inbound port has to be open on the laptop.
 *
 *   ./stream_pcd [port] [lidar_ip] [local_ip]        default port 9910
 *
 * WIRE FORMAT, little-endian, one message per lidar frame:
 *
 *     magic   char[4]   "ULF3"
 *     n       uint32    number of points in this frame
 *     stamp   float64   the sensor's own frame stamp, seconds
 *     accel   float32[3]     the L2's own accelerometer, m/s^2, sensor frame
 *     points  float32[4*n]   x y z intensity, metres, sensor frame (+Z = optical axis)
 *
 * The accelerometer rides along because a cloud in the sensor frame is barely readable
 * when the sensor is not level - and on this robot it is bolted at 45 degrees.  At rest
 * an accelerometer points UP, which is all the viewer needs to stand the room upright.
 *
 * Deliberately not PCD: a header per frame is 16 bytes against ~200 for a PCD one, and
 * the reader wants a length up front so it can frame the stream without parsing text.
 * A frame is ~5200 points = 83 kB, 12 times a second, so this is ~1 MB/s on the wire.
 */
#include "unitree_lidar_sdk.h"
#include <arpa/inet.h>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

using namespace unilidar_sdk2;

static int listen_fd = -1, client_fd = -1;

static void bye(int) {
    if (client_fd >= 0) close(client_fd);
    if (listen_fd >= 0) close(listen_fd);
    printf("\nstream_pcd: closed\n");
    _exit(0);
}

/** send the whole buffer or say the client is gone */
static bool send_all(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    while (len) {
        ssize_t k = send(fd, p, len, MSG_NOSIGNAL);
        if (k <= 0) return false;
        p += k; len -= (size_t)k;
    }
    return true;
}

int main(int argc, char *argv[])
{
    int port = (argc > 1) ? atoi(argv[1]) : 9910;
    std::string lidar_ip = (argc > 2) ? argv[2] : "192.168.1.62";
    std::string local_ip = (argc > 3) ? argv[3] : "192.168.1.2";

    signal(SIGINT, bye);
    signal(SIGTERM, bye);
    signal(SIGPIPE, SIG_IGN);

    UnitreeLidarReader *lreader = createUnitreeLidarReader();
    if (lreader->initializeUDP(6101, lidar_ip, 6201, local_ip)) {
        printf("initializeUDP failed (%s -> %s)\n", lidar_ip.c_str(), local_ip.c_str());
        return 1;
    }
    lreader->startLidarRotation();
    sleep(1);
    lreader->setLidarWorkMode(0);
    sleep(1);

    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);
    if (bind(listen_fd, (sockaddr *)&addr, sizeof(addr)) || listen(listen_fd, 1)) {
        perror("bind/listen");
        return 1;
    }
    printf("stream_pcd: serving %s on port %d — ctrl-C to stop\n", lidar_ip.c_str(), port);

    PointCloudUnitree cloud;
    LidarImuData imu;
    float acc[3] = {0.0f, 0.0f, 0.0f};
    std::vector<float> flat;
    while (true) {
        sockaddr_in peer{};
        socklen_t plen = sizeof(peer);
        printf("waiting for a viewer ...\n");
        client_fd = accept(listen_fd, (sockaddr *)&peer, &plen);
        if (client_fd < 0) continue;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
        printf("viewer connected from %s\n", inet_ntoa(peer.sin_addr));

        // Drain whatever the SDK buffered while nobody was watching, so the first frame
        // the viewer sees is the room as it is now and not a second of history.
        for (int i = 0; i < 500; i++) lreader->runParse();

        long frames = 0;
        bool live = true;
        while (live) {
            int r = lreader->runParse();
            if (r == LIDAR_IMU_DATA_PACKET_TYPE && lreader->getImuData(imu)) {
                memcpy(acc, imu.linear_acceleration, sizeof(acc));
                continue;
            }
            if (r != LIDAR_POINT_DATA_PACKET_TYPE) continue;
            if (!lreader->getPointCloud(cloud)) continue;
            uint32_t n = (uint32_t)cloud.points.size();
            if (!n) continue;
            flat.resize(4 * (size_t)n);
            for (uint32_t i = 0; i < n; i++) {
                flat[4*i+0] = cloud.points[i].x;
                flat[4*i+1] = cloud.points[i].y;
                flat[4*i+2] = cloud.points[i].z;
                flat[4*i+3] = cloud.points[i].intensity;
            }
            double stamp = cloud.stamp;
            live = send_all(client_fd, "ULF3", 4)
                && send_all(client_fd, &n, sizeof(n))
                && send_all(client_fd, &stamp, sizeof(stamp))
                && send_all(client_fd, acc, sizeof(acc))
                && send_all(client_fd, flat.data(), flat.size() * sizeof(float));
            if (live && ++frames % 120 == 0)
                printf("  %ld frames sent\n", frames);
        }
        printf("viewer went away after %ld frames\n", frames);
        close(client_fd);
        client_fd = -1;
    }
}
