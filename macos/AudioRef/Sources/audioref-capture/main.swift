// audioref-capture: headless system-audio capture for the audioref CLI / MCP server.
// Prints one JSON event per line on stdout. SIGINT/SIGTERM stop the capture and keep what was recorded.
import AppKit
import CaptureCore
import Foundation

func emit(_ dict: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: dict), let s = String(data: data, encoding: .utf8) {
        print(s)
        fflush(stdout)
    }
}

let usage = """
usage: audioref-capture --out FILE.wav [--max-seconds 600] [--stop-on-silence 3] [--wait-for-audio 60]
                        [--threshold-db -55] [--simulate INPUT]
       audioref-capture --check        print {"permission": true|false}
       audioref-capture --request      show the macOS permission prompt
"""

var args = Array(CommandLine.arguments.dropFirst())
var dict: [String: Any] = [:]
var i = 0
func value(_ name: String) -> String {
    guard i + 1 < args.count else { FileHandle.standardError.write("missing value for \(name)\n".data(using: .utf8)!); exit(2) }
    i += 1
    return args[i]
}
while i < args.count {
    switch args[i] {
    case "--out": dict["out"] = value("--out")
    case "--max-seconds": dict["max_seconds"] = Double(value("--max-seconds")) ?? 600
    case "--stop-on-silence": dict["stop_on_silence"] = Double(value("--stop-on-silence")) ?? 3
    case "--wait-for-audio": dict["wait_for_audio"] = Double(value("--wait-for-audio")) ?? 60
    case "--threshold-db": dict["threshold_db"] = Double(value("--threshold-db")) ?? -55
    case "--simulate": dict["simulate"] = value("--simulate")
    case "--check":
        emit(["permission": CapturePermission.granted])
        exit(0)
    case "--request":
        emit(["permission": CapturePermission.request()])
        exit(0)
    case "-h", "--help":
        print(usage)
        exit(0)
    default:
        FileHandle.standardError.write("unknown argument \(args[i])\n\(usage)\n".data(using: .utf8)!)
        exit(2)
    }
    i += 1
}
guard dict["out"] != nil else {
    FileHandle.standardError.write("--out is required\n\(usage)\n".data(using: .utf8)!)
    exit(2)
}

signal(SIGPIPE, SIG_IGN)
// a window-server connection keeps ScreenCaptureKit happy in a command-line process
_ = NSApplication.shared
NSApp.setActivationPolicy(.prohibited)

let options = CaptureOptions(dict: dict, defaultOutput: dict["out"] as! String)
var exitCode: Int32 = 0
let session = CaptureSession(options: options) { event in
    if case .failed = event { exitCode = 1 }
    emit(event.dictionary)
}

var signalSources: [DispatchSourceSignal] = []
for sig in [SIGINT, SIGTERM] {
    signal(sig, SIG_IGN)
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    src.setEventHandler { session.requestStop() }
    src.resume()
    signalSources.append(src)
}

Task {
    await session.run()
    exit(exitCode)
}
RunLoop.main.run()
