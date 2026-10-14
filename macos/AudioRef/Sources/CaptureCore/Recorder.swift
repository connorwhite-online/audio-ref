import AVFoundation
import Foundation

/// Options for one capture.
public struct CaptureOptions {
    public var outputPath: String
    /// Hard cap on the recording length (seconds).
    public var maxSeconds: Double = 600
    /// Stop after this much continuous silence once audio has started (0 = never).
    public var stopOnSilence: Double = 3.0
    /// Give up if nothing is heard within this time (0 = start recording immediately).
    public var waitForAudio: Double = 60
    /// Level (dBFS RMS per buffer) above which the signal counts as "sound".
    public var thresholdDB: Double = -55
    /// Audio kept from just before the trigger so the first transient isn't clipped.
    public var preRoll: Double = 0.3
    /// Quiet tail kept after the last sound (reverb/decay).
    public var keepTail: Double = 0.5
    /// Test/simulation: read this audio file instead of capturing system audio.
    public var simulateInput: String?

    public init(outputPath: String) { self.outputPath = outputPath }

    /// Build from a JSON-ish dictionary (IPC / CLI).
    public init(dict: [String: Any], defaultOutput: String) {
        outputPath = (dict["out"] as? String) ?? defaultOutput
        if let v = dict["max_seconds"] as? Double { maxSeconds = v } else if let v = dict["max_seconds"] as? Int { maxSeconds = Double(v) }
        if let v = dict["stop_on_silence"] as? Double { stopOnSilence = v } else if let v = dict["stop_on_silence"] as? Int { stopOnSilence = Double(v) }
        if let v = dict["wait_for_audio"] as? Double { waitForAudio = v } else if let v = dict["wait_for_audio"] as? Int { waitForAudio = Double(v) }
        if let v = dict["threshold_db"] as? Double { thresholdDB = v } else if let v = dict["threshold_db"] as? Int { thresholdDB = Double(v) }
        simulateInput = dict["simulate"] as? String
    }
}

/// Events emitted during a capture.
public enum CaptureEvent {
    case waiting
    case started
    case level(db: Double, elapsed: Double)
    case finished(path: String, duration: Double, reason: String)
    case failed(String)

    public var isTerminal: Bool {
        switch self {
        case .finished, .failed: return true
        default: return false
        }
    }

    public var dictionary: [String: Any] {
        switch self {
        case .waiting: return ["event": "waiting"]
        case .started: return ["event": "started"]
        case let .level(db, elapsed):
            return ["event": "level", "db": (db * 10).rounded() / 10, "elapsed": (elapsed * 100).rounded() / 100]
        case let .finished(path, duration, reason):
            return ["event": "finished", "path": path, "duration": (duration * 1000).rounded() / 1000, "reason": reason]
        case let .failed(message): return ["event": "error", "message": message]
        }
    }
}

/// Gates, buffers and writes audio: starts at the first sound (with pre-roll), holds back trailing silence,
/// stops after `stopOnSilence` seconds of quiet or at `maxSeconds`. Not thread-safe: feed from one queue.
public final class Recorder {
    public enum Phase { case waiting, recording, finished }
    public enum Step { case waiting, started, recording, stop(String) }

    public private(set) var phase: Phase = .waiting
    public private(set) var writtenFrames: Int64 = 0
    public private(set) var lastLevel: Double = -120
    public private(set) var sampleRate: Double = 48000

    private let opts: CaptureOptions
    private var file: AVAudioFile?
    private var preroll: [AVAudioPCMBuffer] = []
    private var prerollFrames: Int64 = 0
    private var pending: [AVAudioPCMBuffer] = []
    private var pendingFrames: Int64 = 0
    private var waitedFrames: Int64 = 0

    public init(options: CaptureOptions) { opts = options }

    public var recordedSeconds: Double { Double(writtenFrames) / sampleRate }

    public static func rmsDB(_ b: AVAudioPCMBuffer) -> Double {
        guard let data = b.floatChannelData else { return -120 }
        let n = Int(b.frameLength)
        let channels = Int(b.format.channelCount)
        guard n > 0, channels > 0 else { return -120 }
        var sum: Double = 0
        if b.format.isInterleaved {
            let p = data[0]
            for i in 0..<(n * channels) { let v = Double(p[i]); sum += v * v }
        } else {
            for c in 0..<channels {
                let p = data[c]
                for i in 0..<n { let v = Double(p[i]); sum += v * v }
            }
        }
        let rms = (sum / Double(n * channels)).squareRoot()
        return rms > 0 ? 20 * log10(rms) : -120
    }

    public func process(_ buf: AVAudioPCMBuffer) throws -> Step {
        sampleRate = buf.format.sampleRate
        let frames = Int64(buf.frameLength)
        let db = Recorder.rmsDB(buf)
        lastLevel = db
        let loud = db > opts.thresholdDB
        switch phase {
        case .finished:
            return .stop("finished")
        case .waiting:
            waitedFrames += frames
            if loud || opts.waitForAudio <= 0 {
                phase = .recording
                try openFile(format: buf.format)
                for p in preroll { try write(p) }
                preroll.removeAll()
                prerollFrames = 0
                try write(buf)
                return .started
            }
            preroll.append(buf)
            prerollFrames += frames
            let keep = Int64(opts.preRoll * sampleRate)
            while prerollFrames - Int64(preroll[0].frameLength) >= keep, preroll.count > 1 {
                prerollFrames -= Int64(preroll.removeFirst().frameLength)
            }
            if opts.waitForAudio > 0, Double(waitedFrames) / sampleRate >= opts.waitForAudio {
                return .stop("no-audio")
            }
            return .waiting
        case .recording:
            if loud {
                for p in pending { try write(p) }
                pending.removeAll()
                pendingFrames = 0
                try write(buf)
            } else {
                pending.append(buf)
                pendingFrames += frames
                if opts.stopOnSilence > 0, Double(pendingFrames) / sampleRate >= opts.stopOnSilence {
                    return .stop("silence")
                }
            }
            if Double(writtenFrames + pendingFrames) / sampleRate >= opts.maxSeconds {
                return .stop("max-duration")
            }
            return .recording
        }
    }

    /// Close the file, keeping only a short quiet tail. Returns the recorded duration in seconds.
    @discardableResult
    public func finish() -> Double {
        if phase == .recording {
            var tail = Int64(opts.keepTail * sampleRate)
            for p in pending where tail > 0 {
                let n = min(Int64(p.frameLength), tail)
                p.frameLength = AVAudioFrameCount(n)
                try? write(p)
                tail -= n
            }
        }
        pending.removeAll()
        preroll.removeAll()
        file = nil  // AVAudioFile flushes and closes on release
        phase = .finished
        return recordedSeconds
    }

    private func openFile(format: AVAudioFormat) throws {
        let url = URL(fileURLWithPath: opts.outputPath)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var settings = format.settings
        settings[AVLinearPCMIsNonInterleaved] = false  // WAV data is interleaved on disk
        file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: format.commonFormat,
                               interleaved: format.isInterleaved)
    }

    private func write(_ b: AVAudioPCMBuffer) throws {
        guard let f = file, b.frameLength > 0 else { return }
        try f.write(from: b)
        writtenFrames += Int64(b.frameLength)
    }
}
