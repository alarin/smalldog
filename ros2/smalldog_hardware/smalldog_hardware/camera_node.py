#!/usr/bin/env python3
"""The IMX415 as a ROS 2 topic: the UVC module's own MJPEG frames -> sensor_msgs/CompressedImage.

    ros2 run smalldog_hardware camera                                   # /dev/video0, 1280x720
    ros2 run smalldog_hardware camera --ros-args -p width:=1920 -p height:=1080
    ros2 run smalldog_hardware camera --ros-args -p input:=lavfi -p device:=testsrc=size=1280x720:rate=20
                                                                        # no camera: a test pattern

    /camera/image/compressed     sensor_msgs/CompressedImage, format "jpeg", frame
                                 camera_optical_frame, SensorDataQoS, at the camera's own
                                 rate (20 fps indoors, 30 in the light — 3d/ref/camera/README.md)

The module is a UVC camera that delivers MJPEG, and a JPEG is what Foxglove's Image panel
wants, so nothing is decoded: `ffmpeg -c:v copy` hands the frames over as they come off the
USB and this node cuts the stream at the JPEG markers and stamps each one. One ffmpeg
process, no OpenCV, ~2 % of a core at 720p.

The image is upside-down on the standing robot (the board is in the mount that way up;
the same README). It stays upside-down here: `flip:=true` re-encodes every frame through
`-vf hflip,vflip`, which is a whole JPEG encode on the Pi per frame, and the Image panel
rotates for free — the layout in `foxglove/robot.json` does. Turn the flip on for a
consumer that cannot rotate.
"""
import os
import shutil
import subprocess
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

SOI, EOI = b'\xff\xd8', b'\xff\xd9'        # JPEG start / end of image


class CameraNode(Node):
    def __init__(self):
        super().__init__('smalldog_camera')
        p = self.declare_parameter
        p('input', 'v4l2')                 # ffmpeg input format: v4l2 on the Pi, lavfi for a pattern
        p('device', '/dev/video0')
        p('width', 1280)
        p('height', 720)
        p('fps', 30)                       # what is asked; the exposure decides what arrives
        p('flip', False)
        p('topic', '/camera/image/compressed')
        p('frame_id', 'camera_optical_frame')
        g = lambda k: self.get_parameter(k).value    # noqa: E731

        self.frame_id = g('frame_id')
        self.pub = self.create_publisher(CompressedImage, g('topic'), qos_profile_sensor_data)
        self.frames = 0
        self.cmd = self.ffmpeg_command(g('input'), g('device'), int(g('width')),
                                       int(g('height')), int(g('fps')), bool(g('flip')))
        self.proc = None
        self.get_logger().info(' '.join(self.cmd))
        self.reader = threading.Thread(target=self.read_forever, daemon=True)
        self.reader.start()
        self.create_timer(5.0, self.report)

    @staticmethod
    def ffmpeg_command(inp, device, w, h, fps, flip):
        if shutil.which('ffmpeg') is None:
            raise SystemExit('ffmpeg is not on PATH (apt install ffmpeg)')
        cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error']
        if inp == 'v4l2':
            cmd += ['-f', inp, '-input_format', 'mjpeg', '-video_size', f'{w}x{h}',
                    '-framerate', str(fps)]
        else:
            cmd += ['-re', '-f', inp]          # a synthetic source at its own rate, not flat out
        cmd += ['-i', device]
        if flip:
            cmd += ['-vf', 'hflip,vflip', '-c:v', 'mjpeg', '-q:v', '4']
        elif inp == 'v4l2':
            cmd += ['-c:v', 'copy']
        else:
            cmd += ['-c:v', 'mjpeg', '-q:v', '4']   # a synthetic source is raw video
        return cmd + ['-f', 'mjpeg', 'pipe:1']

    def read_forever(self):
        """ffmpeg's stdout is JPEG after JPEG; publish each one as it completes."""
        # the system's ffmpeg against the system's libraries: a ROS env that sets DYLD_*
        # (the mac's pixi env does) makes a homebrew ffmpeg dlopen the wrong libiconv
        env = {k: v for k, v in os.environ.items() if not k.startswith('DYLD_')}
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=env)
        buf = bytearray()
        while rclpy.ok():
            chunk = self.proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                start = buf.find(SOI)
                if start < 0:
                    buf.clear()
                    break
                end = buf.find(EOI, start + 2)
                if end < 0:
                    if start:
                        del buf[:start]
                    break
                self.publish(bytes(buf[start:end + 2]))
                del buf[:end + 2]
        err = self.proc.stderr.read().decode(errors='replace').strip()
        if rclpy.ok():
            self.get_logger().error(f'ffmpeg ended: {err or "no output"}')

    def publish(self, jpeg):
        m = CompressedImage()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.format = 'jpeg'
        m.data = jpeg
        self.pub.publish(m)
        self.frames += 1

    def report(self):
        n, self.frames = self.frames, 0
        if n:
            self.get_logger().info(f'{n / 5.0:.1f} fps')
        elif self.proc is not None and self.proc.poll() is None:
            self.get_logger().warn('no frames in 5 s')

    def destroy_node(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
