"""
loopback.py — a servo bus made of bytes, for testing without a robot.

Answers PING / READ / WRITE / SYNC_WRITE / SYNC_READ from a byte-addressable
register file, which is what a real servo is. It exists so that the packet
encoding — checksums, byte order, sign-magnitude, the SyncWrite layout — is
exercised on every run instead of on the day the hardware arrives, and so that
robot/bench/sweep.py's trajectory logic can be driven end to end with `--dry-run`.

It deliberately models NO dynamics. Position does not follow Goal Position here;
that is rl/actuator.py's job, and conflating the two would let a bug in one hide
behind the other.
"""
from __future__ import annotations

from . import registers as R

MEM = 128


class LoopbackBus:
    def __init__(self, servos: dict[int, dict], endian="little"):
        self.mem = {i: bytearray(MEM) for i in servos}
        self.endian = endian
        self._out = bytearray()
        self.corrupt_next = False
        for i, init in servos.items():
            for addr, v in init.items():
                self.set(i, addr, v)

    # ---------------------------------------------------------------- state
    def set(self, dev_id: int, addr: int, value: int):
        w = R.WIDTH.get(addr, 1)
        b = (value & 0xFF).to_bytes(1, "little") if w == 1 else \
            (value & 0xFFFF).to_bytes(2, self.endian)
        self.mem[dev_id][addr:addr + w] = b

    def get(self, dev_id: int, addr: int) -> int:
        w = R.WIDTH.get(addr, 1)
        return int.from_bytes(self.mem[dev_id][addr:addr + w], self.endian)

    # ------------------------------------------------------------ transport
    def reset_input_buffer(self):
        self._out.clear()

    def read(self, n: int) -> bytes:
        out, self._out = bytes(self._out[:n]), self._out[n:]
        return out

    def write(self, data: bytes):
        i = 0
        while i + 4 <= len(data):
            if data[i:i + 2] != b"\xff\xff":
                i += 1
                continue
            dev, ln = data[i + 2], data[i + 3]
            frame = data[i:i + 4 + ln]
            self._handle(frame)
            i += 4 + ln

    # -------------------------------------------------------------- replies
    def _reply(self, dev_id: int, params: bytes = b"", err: int = 0):
        body = bytes([dev_id, len(params) + 2, err]) + params
        chk = (~sum(body)) & 0xFF
        if self.corrupt_next:
            chk ^= 0xFF
            self.corrupt_next = False
        self._out += b"\xff\xff" + body + bytes([chk])

    def _handle(self, frame: bytes):
        dev, inst, params = frame[2], frame[4], frame[5:-1]
        if inst == R.PING:
            if dev in self.mem:
                self._reply(dev)
            return
        if inst == R.READ:
            if dev not in self.mem:
                return
            addr, n = params[0], params[1]
            self._reply(dev, bytes(self.mem[dev][addr:addr + n]))
            return
        if inst == R.WRITE:
            if dev not in self.mem:
                return
            addr, data = params[0], params[1:]
            self.mem[dev][addr:addr + len(data)] = data
            self._reply(dev)
            return
        if inst == R.SYNC_WRITE:
            addr, width, rest = params[0], params[1], params[2:]
            for k in range(0, len(rest), 1 + width):
                i, data = rest[k], rest[k + 1:k + 1 + width]
                if i in self.mem:
                    self.mem[i][addr:addr + width] = data
            return                                   # SyncWrite has no reply
        if inst == R.SYNC_READ:
            addr, n, ids = params[0], params[1], params[2:]
            for i in ids:
                if i in self.mem:
                    self._reply(i, bytes(self.mem[i][addr:addr + n]))
            return
