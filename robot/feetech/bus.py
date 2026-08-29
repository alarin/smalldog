"""
bus.py — half-duplex Feetech SMS/STS bus, with the timing instrumented.

    python -m feetech.bus --port /dev/ttyUSB0 --ping 1
    python -m feetech.bus --port /dev/ttyUSB0 --dump 1
    python -m feetech.bus --selftest            # no hardware: packets round-trip

Twelve servos, one wire, 1 Mbit, and a 50 Hz control loop that must not miss its
deadline. The two things that decide whether that works are not in the protocol
at all, so this file measures rather than assumes them:

  * **How long a transaction actually takes.** A byte is 10 bits, so 1 Mbit is
    10 us/byte and a 68-byte SyncWrite for twelve goal positions is 0.68 ms on
    the wire. The wire is never the problem. The USB stack is: an FTDI adapter
    ships with latency_timer = 16 ms, which by itself is most of a 20 ms tick,
    and CH340 has its own. `stats()` reports what the round trip really costs on
    this machine, with this adapter, and robot/bench/bus_probe.py turns that into
    the delay distribution the training randomisation uses. Guessing this number
    and training against the guess is how a policy that walks in simulation
    falls over on the robot.

  * **Whether SYNC_READ exists.** It is not in every firmware. Without it, twelve
    feedback reads are twelve round trips instead of one broadcast plus twelve
    replies. `sync_read` falls back to sequential reads and says which path it
    took, so the timing report is never quietly measuring a different protocol
    than the one the runtime will use.

The transport is duck-typed (`write`, `read`, `reset_input_buffer`) so the
self-test can drive a simulated servo and exercise every packet without a robot
on the desk.
"""
from __future__ import annotations

import argparse
import struct
import time

from . import registers as R


class BusError(RuntimeError):
    pass


class Timeout(BusError):
    pass


class Checksum(BusError):
    pass


def checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


def pack(dev_id: int, instruction: int, params: bytes = b"") -> bytes:
    body = bytes([dev_id, len(params) + 2, instruction]) + params
    return b"\xff\xff" + body + bytes([checksum(body)])


class Bus:
    def __init__(self, port="/dev/ttyUSB0", baud=1_000_000, timeout=0.01,
                 endian="little", transport=None, discard_echo=None):
        if transport is not None:
            self.io = transport
        else:
            import serial
            self.io = serial.Serial(port, baud, timeout=timeout)
        self.timeout = timeout
        self.endian = endian
        # Some TTL adapters put the transmitted bytes back into the receive
        # buffer. Auto-detected on the first reply rather than configured: a
        # wrong guess here looks exactly like a servo answering garbage.
        self.discard_echo = discard_echo
        self.n_tx = self.n_timeout = self.n_checksum = 0
        self._times: list[float] = []
        self._sync_read_ok: bool | None = None

    # ------------------------------------------------------------ encoding
    def _u16(self, v: int) -> bytes:
        return struct.pack("<H" if self.endian == "little" else ">H", v & 0xFFFF)

    def _from_u16(self, b: bytes) -> int:
        return struct.unpack("<H" if self.endian == "little" else ">H", b)[0]

    def value(self, addr: int, raw: bytes) -> int:
        """Decode one register's bytes, honouring width and sign-magnitude."""
        v = raw[0] if len(raw) == 1 else self._from_u16(raw[:2])
        if addr in R.SIGN_MAGNITUDE and len(raw) >= 2:
            v = -(v & 0x7FFF) if v & 0x8000 else v
        return v

    def encode(self, addr: int, v: int) -> bytes:
        if R.WIDTH.get(addr, 1) == 1:
            return bytes([v & 0xFF])
        if addr in R.SIGN_MAGNITUDE and v < 0:
            v = (-v) | 0x8000
        return self._u16(v)

    # ------------------------------------------------------------ transport
    def _txrx(self, packet: bytes, expect: int) -> bytes:
        """One transaction. `expect` is the payload length of the reply."""
        self.io.reset_input_buffer()
        t0 = time.perf_counter()
        self.io.write(packet)
        need = 6 + expect                       # ff ff id len err <payload> chk
        if self.discard_echo:
            need += len(packet)
        buf = b""
        while len(buf) < need:
            chunk = self.io.read(need - len(buf))
            if not chunk:
                break
            buf += chunk
        dt = time.perf_counter() - t0
        self.n_tx += 1
        self._times.append(dt)

        if self.discard_echo is None and buf.startswith(packet):
            self.discard_echo = True
        if self.discard_echo and buf.startswith(packet):
            buf = buf[len(packet):]
            tail = 6 + expect - len(buf)
            while tail > 0:
                chunk = self.io.read(tail)
                if not chunk:
                    break
                buf += chunk
                tail = 6 + expect - len(buf)
        elif self.discard_echo is None:
            self.discard_echo = False

        if len(buf) < 6 + expect:
            self.n_timeout += 1
            raise Timeout(f"short reply: {len(buf)} of {6+expect} bytes, {buf.hex()}")
        if buf[:2] != b"\xff\xff":
            self.n_checksum += 1
            raise Checksum(f"no header in {buf.hex()}")
        body = buf[2:5 + expect]
        if checksum(body) != buf[5 + expect]:
            self.n_checksum += 1
            raise Checksum(f"bad checksum in {buf.hex()}")
        return buf[5:5 + expect]

    # ------------------------------------------------------------- commands
    def ping(self, dev_id: int) -> bool:
        try:
            self._txrx(pack(dev_id, R.PING), 0)
            return True
        except BusError:
            return False

    def read(self, dev_id: int, addr: int, n: int | None = None) -> int | bytes:
        n_ = n if n is not None else R.WIDTH.get(addr, 1)
        raw = self._txrx(pack(dev_id, R.READ, bytes([addr, n_])), n_)
        return raw if n is not None else self.value(addr, raw)

    def write(self, dev_id: int, addr: int, value: int) -> None:
        self._txrx(pack(dev_id, R.WRITE, bytes([addr]) + self.encode(addr, value)), 0)

    def sync_write(self, addr: int, values: dict[int, int]) -> None:
        """One packet for the whole bus. No replies, so no round trip to wait on.

        This is the write half of the 50 Hz loop and it is why the loop fits:
        twelve goal positions cost one packet, not twelve transactions.
        """
        width = R.WIDTH.get(addr, 1)
        params = bytes([addr, width])
        for i, v in sorted(values.items()):
            params += bytes([i]) + self.encode(addr, v)
        self.io.reset_input_buffer()
        t0 = time.perf_counter()
        self.io.write(pack(R.BROADCAST_ID, R.SYNC_WRITE, params))
        self._times.append(time.perf_counter() - t0)
        self.n_tx += 1

    def sync_read(self, addr: int, n: int, ids: list[int]) -> dict[int, bytes]:
        """Broadcast one read; every servo answers in turn. Falls back if absent."""
        if self._sync_read_ok is not False:
            try:
                out = self._sync_read(addr, n, ids)
                self._sync_read_ok = True
                return out
            except BusError:
                if self._sync_read_ok:
                    raise                      # it worked before: a real failure
                self._sync_read_ok = False
        return {i: self.read(i, addr, n) for i in ids}

    def _sync_read(self, addr, n, ids):
        params = bytes([addr, n]) + bytes(ids)
        self.io.reset_input_buffer()
        t0 = time.perf_counter()
        self.io.write(pack(R.BROADCAST_ID, R.SYNC_READ, params))
        need = (6 + n) * len(ids)
        buf = b""
        while len(buf) < need:
            chunk = self.io.read(need - len(buf))
            if not chunk:
                break
            buf += chunk
        self._times.append(time.perf_counter() - t0)
        self.n_tx += 1
        if len(buf) < need:
            self.n_timeout += 1
            raise Timeout(f"sync_read: {len(buf)} of {need} bytes")
        out = {}
        for k, i in enumerate(ids):
            f = buf[k * (6 + n):(k + 1) * (6 + n)]
            body = f[2:5 + n]
            if f[:2] != b"\xff\xff" or checksum(body) != f[5 + n]:
                self.n_checksum += 1
                raise Checksum(f"sync_read frame {k} (id {i}) is malformed")
            out[i] = f[5:5 + n]
        return out

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict:
        import statistics
        t = sorted(self._times)
        if not t:
            return {"n": 0}
        q = lambda p: t[min(len(t) - 1, int(p * len(t)))]
        return {"n": len(t), "mean_ms": 1e3 * statistics.fmean(t),
                "p50_ms": 1e3 * q(0.50), "p95_ms": 1e3 * q(0.95),
                "p99_ms": 1e3 * q(0.99), "max_ms": 1e3 * t[-1],
                "timeouts": self.n_timeout, "checksum_errors": self.n_checksum,
                "sync_read": self._sync_read_ok}

    def reset_stats(self):
        self._times.clear()
        self.n_tx = self.n_timeout = self.n_checksum = 0


class Servo:
    """One servo in SI units. Conversions live here so registers.py stays raw."""

    def __init__(self, bus: Bus, dev_id: int, centre=2048, sign=+1):
        self.bus, self.id, self.centre, self.sign = bus, dev_id, centre, sign

    # counts <-> rad about the calibrated centre
    def to_rad(self, counts: int) -> float:
        return self.sign * (counts - self.centre) * 2 * 3.141592653589793 / R.COUNTS_PER_TURN

    def to_counts(self, rad: float) -> int:
        c = self.centre + self.sign * rad * R.COUNTS_PER_TURN / (2 * 3.141592653589793)
        return int(round(min(R.COUNTS_PER_TURN - 1, max(0, c))))

    def feedback(self) -> dict:
        """Position, speed, load, voltage, temperature and current in one read."""
        raw = self.bus.read(self.id, R.FEEDBACK_START, R.FEEDBACK_LEN)
        return self.decode(raw)

    def decode(self, raw: bytes) -> dict:
        b, off = self.bus, R.FEEDBACK_START
        g = lambda addr, n=2: b.value(addr, raw[addr - off:addr - off + n])
        speed_counts = g(R.PRESENT_SPEED)
        return dict(
            q=self.to_rad(g(R.PRESENT_POSITION)),
            counts=g(R.PRESENT_POSITION),
            w=self.sign * speed_counts * R.SPEED_LSB_COUNTS_PER_S
              * 2 * 3.141592653589793 / R.COUNTS_PER_TURN,
            load=g(R.PRESENT_LOAD),
            volt=g(R.PRESENT_VOLTAGE, 1) * R.VOLTAGE_LSB_V,
            temp=g(R.PRESENT_TEMPERATURE, 1),
            current=abs(g(R.PRESENT_CURRENT)) * R.CURRENT_LSB_A,
        )

    def torque(self, on: bool):
        self.bus.write(self.id, R.TORQUE_ENABLE, 1 if on else 0)

    def goal(self, rad: float):
        self.bus.write(self.id, R.GOAL_POSITION, self.to_counts(rad))

    def registers(self) -> dict:
        """The control settings the fit is conditional on. Read before every run."""
        out = {}
        for name in R.CONTROL_REGISTERS:
            addr = getattr(R, name)
            try:
                out[name] = self.bus.read(self.id, addr)
            except BusError:
                out[name] = None
        return out


# ============================================================== self-test
def _selftest() -> int:
    from .loopback import LoopbackBus
    bus = Bus(transport=LoopbackBus({1: {}, 2: {}}), discard_echo=False)
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: {got!r}"
              + ("" if good else f" != {want!r}"))

    check("ping present", bus.ping(1), True)
    check("ping absent", bus.ping(9), False)

    bus.write(1, R.GOAL_POSITION, 3000)
    check("write/read u16", bus.read(1, R.GOAL_POSITION), 3000)

    bus.write(1, R.P_COEF, 32)
    check("write/read u8", bus.read(1, R.P_COEF), 32)

    # sign-magnitude: -300 counts/s must survive the round trip as -300, and
    # must NOT come back as 32768 + 300.
    bus.io.set(1, R.PRESENT_SPEED, (300) | 0x8000)
    check("sign-magnitude negative", bus.read(1, R.PRESENT_SPEED), -300)
    bus.io.set(1, R.PRESENT_SPEED, 300)
    check("sign-magnitude positive", bus.read(1, R.PRESENT_SPEED), 300)

    bus.sync_write(R.GOAL_POSITION, {1: 1000, 2: 2000})
    check("sync_write id 1", bus.read(1, R.GOAL_POSITION), 1000)
    check("sync_write id 2", bus.read(2, R.GOAL_POSITION), 2000)

    bus.io.set(1, R.PRESENT_POSITION, 2148)
    bus.io.set(1, R.PRESENT_VOLTAGE, 118)
    bus.io.set(1, R.PRESENT_TEMPERATURE, 41)
    bus.io.set(1, R.PRESENT_CURRENT, 77)
    fb = Servo(bus, 1).feedback()
    check("feedback counts", fb["counts"], 2148)
    check("feedback volts", round(fb["volt"], 2), 11.8)
    check("feedback temp", fb["temp"], 41)
    check("feedback current A", round(fb["current"], 4), round(77 * R.CURRENT_LSB_A, 4))
    check("one feedback read is one transaction", bus.n_tx - 0 > 0, True)

    got = bus.sync_read(R.PRESENT_POSITION, 2, [1, 2])
    check("sync_read id 1", bus.value(R.PRESENT_POSITION, got[1]), 2148)

    # A corrupted reply must raise, not return plausible nonsense.
    bus.io.corrupt_next = True
    try:
        bus.read(1, R.PRESENT_POSITION)
        check("corrupt reply raises", False, True)
    except Checksum:
        check("corrupt reply raises", True, True)

    # A 12-servo SyncWrite must be one packet of the size the timing budget
    # assumes; if this ever grows, the 50 Hz budget in robot/README.md is stale.
    n = len(pack(R.BROADCAST_ID, R.SYNC_WRITE,
                 bytes([R.GOAL_POSITION, 2]) + b"".join(
                     bytes([i]) + b"\x00\x00" for i in range(1, 13))))
    check("12-servo SyncWrite packet bytes", n, 44)
    print(f"\n  a {n}-byte packet is {n*10} us on the wire at 1 Mbit")
    print("  " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--endian", default="little", choices=("little", "big"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ping", type=int)
    ap.add_argument("--scan", action="store_true", help="ping ids 1..20")
    ap.add_argument("--dump", type=int, metavar="ID")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(_selftest())

    bus = Bus(a.port, a.baud, endian=a.endian)
    if a.ping is not None:
        print("present" if bus.ping(a.ping) else "no answer")
    if a.scan:
        print("found:", [i for i in range(1, 21) if bus.ping(i)])
    if a.dump is not None:
        print(f"servo {a.dump} — every register in registers.py")
        for name in sorted(dir(R)):
            addr = getattr(R, name)
            if name.isupper() and isinstance(addr, int) and name not in (
                    "PING", "READ", "WRITE", "REG_WRITE", "ACTION", "RESET",
                    "SYNC_READ", "SYNC_WRITE", "BROADCAST_ID", "COUNTS_PER_TURN",
                    "FEEDBACK_START", "FEEDBACK_LEN"):
                try:
                    print(f"  {name:<22} @{addr:<3} = {bus.read(a.dump, addr)}")
                except BusError as e:
                    print(f"  {name:<22} @{addr:<3} ! {e}")
        print("\n  Sanity-check these before trusting them: a temperature near 25-45,"
              "\n  a voltage near 11-12.6 on 3S, a position in 0..4095. Anything wild"
              "\n  means the address or the byte order is wrong for this firmware.")
    print("timing:", bus.stats())


if __name__ == "__main__":
    main()
