import CaptureCore
import Darwin
import Foundation

/// Line-delimited JSON over a Unix socket (owner-only permissions) so the audioref CLI / MCP server can ask the
/// app to capture. Commands: {"cmd":"status"} | {"cmd":"stop"} | {"cmd":"capture", "out":..., "max_seconds":...,
/// "stop_on_silence":..., "wait_for_audio":..., "threshold_db":..., "simulate":...}. A capture streams events
/// (waiting / started / level / finished / error) on the same connection until it ends.
final class ControlServer {
    private let path: String
    private weak var model: AppModel?
    private var fd: Int32 = -1

    init(path: String, model: AppModel) {
        self.path = path
        self.model = model
    }

    func start() throws {
        unlink(path)
        fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw CaptureError("socket() failed") }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8CString)
        guard bytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else { throw CaptureError("socket path too long") }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: bytes.count) { p in
                for (i, b) in bytes.enumerated() { p[i] = b }
            }
        }
        let len = socklen_t(MemoryLayout<sockaddr_un>.size)
        let rc = withUnsafePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, len) } }
        guard rc == 0 else { throw CaptureError("bind() failed: \(String(cString: strerror(errno)))") }
        chmod(path, 0o600)
        guard listen(fd, 8) == 0 else { throw CaptureError("listen() failed") }
        let listener = fd
        Thread.detachNewThread { [weak self] in
            while true {
                let c = accept(listener, nil, nil)
                if c < 0 {
                    if errno == EINTR { continue }
                    break
                }
                var one: Int32 = 1
                setsockopt(c, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
                Thread.detachNewThread { self?.handle(c) }
            }
        }
    }

    private func readLine(_ c: Int32) -> String? {
        var data = Data()
        var byte: UInt8 = 0
        while data.count < 65536 {
            let n = read(c, &byte, 1)
            if n <= 0 { return data.isEmpty ? nil : String(data: data, encoding: .utf8) }
            if byte == 10 { break }
            data.append(byte)
        }
        return String(data: data, encoding: .utf8)
    }

    private func handle(_ c: Int32) {
        let writeLock = NSLock()
        func send(_ obj: [String: Any]) {
            guard var data = try? JSONSerialization.data(withJSONObject: obj) else { return }
            data.append(10)
            writeLock.lock()
            defer { writeLock.unlock() }
            data.withUnsafeBytes { raw in
                var off = 0
                while off < raw.count {
                    let n = write(c, raw.baseAddress!.advanced(by: off), raw.count - off)
                    if n <= 0 { return }
                    off += n
                }
            }
        }
        defer { close(c) }
        guard let line = readLine(c),
              let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
              let cmd = obj["cmd"] as? String else {
            send(["ok": false, "error": "bad request"])
            return
        }
        guard let model else { return }
        switch cmd {
        case "status":
            let st = DispatchQueue.main.sync { model.statusDictionary() }
            send(st)
        case "stop":
            DispatchQueue.main.async { model.stopCapture() }
            send(["ok": true])
        case "capture":
            let defaultOut = (model.config.capturesDir as NSString).appendingPathComponent("capture-\(timestamp()).wav")
            let opts = CaptureOptions(dict: obj, defaultOutput: defaultOut)
            let done = DispatchSemaphore(value: 0)
            let started = DispatchQueue.main.sync {
                model.startExternalCapture(opts) { event in
                    send(event.dictionary)
                    if event.isTerminal { done.signal() }
                }
            }
            if !started {
                send(["event": "error", "message": "AudioRef is busy with another capture or analysis"])
                return
            }
            done.wait()
        default:
            send(["ok": false, "error": "unknown command \(cmd)"])
        }
    }
}
