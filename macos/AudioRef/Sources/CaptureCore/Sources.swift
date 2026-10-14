import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import ScreenCaptureKit

/// Something that produces audio buffers on its own queue.
public protocol AudioSource: AnyObject {
    var onBuffer: ((AVAudioPCMBuffer) -> Void)? { get set }
    var onEnd: ((Error?) -> Void)? { get set }
    func start() async throws
    func stop() async
}

public struct CaptureError: LocalizedError {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

public enum CapturePermission {
    /// True when this process may capture screen + system audio.
    public static var granted: Bool { CGPreflightScreenCaptureAccess() }

    /// Ask macOS for access (shows the system prompt the first time). Returns the current state.
    @discardableResult
    public static func request() -> Bool { CGRequestScreenCaptureAccess() }

    public static let instructions =
        "Allow screen & system audio recording: System Settings → Privacy & Security → Screen & System Audio " +
        "Recording → turn on AudioRef (or the app that launched this command), then quit and reopen it."
}

/// System audio output via ScreenCaptureKit (everything playing on the Mac, minus our own process).
public final class SystemAudioSource: NSObject, AudioSource, SCStreamOutput, SCStreamDelegate {
    public var onBuffer: ((AVAudioPCMBuffer) -> Void)?
    public var onEnd: ((Error?) -> Void)?
    private var stream: SCStream?
    private let queue = DispatchQueue(label: "audioref.capture.system", qos: .userInitiated)

    public override init() { super.init() }

    public func start() async throws {
        if !CapturePermission.granted {
            CapturePermission.request()
            throw CaptureError("Screen & System Audio Recording permission is not granted. " + CapturePermission.instructions)
        }
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        } catch {
            throw CaptureError("Could not access system audio (\(error.localizedDescription)). " + CapturePermission.instructions)
        }
        guard let display = content.displays.first else { throw CaptureError("No display available for capture.") }
        let pid = ProcessInfo.processInfo.processIdentifier
        let me = content.applications.filter { $0.processID == pid }
        let filter = SCContentFilter(display: display, excludingApplications: me, exceptingWindows: [])
        let cfg = SCStreamConfiguration()
        cfg.capturesAudio = true
        cfg.excludesCurrentProcessAudio = true
        cfg.sampleRate = 48000
        cfg.channelCount = 2
        // we only want audio: keep the (mandatory) video stream tiny and slow
        cfg.width = 2
        cfg.height = 2
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 2)
        cfg.queueDepth = 3
        let s = SCStream(filter: filter, configuration: cfg, delegate: self)
        try s.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
        try await s.startCapture()
        stream = s
    }

    public func stop() async {
        guard let s = stream else { return }
        stream = nil
        try? await s.stopCapture()
    }

    public func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid, let buf = sampleBuffer.pcmBufferCopy() else { return }
        onBuffer?(buf)
    }

    public func stream(_ stream: SCStream, didStopWithError error: Error) {
        onEnd?(error)
    }
}

extension CMSampleBuffer {
    /// Copy the audio samples into a freshly allocated AVAudioPCMBuffer (safe to keep after the callback).
    func pcmBufferCopy() -> AVAudioPCMBuffer? {
        guard let fd = CMSampleBufferGetFormatDescription(self),
              let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(fd) else { return nil }
        var asbd = asbdPtr.pointee
        guard let format = AVAudioFormat(streamDescription: &asbd) else { return nil }
        let frames = AVAudioFrameCount(CMSampleBufferGetNumSamples(self))
        guard frames > 0, let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return nil }
        pcm.frameLength = frames
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(self, at: 0, frameCount: Int32(frames),
                                                                   into: pcm.mutableAudioBufferList)
        return status == noErr ? pcm : nil
    }
}

/// Reads an audio file and feeds it like a live source (tests, and re-capturing from a file).
public final class FileAudioSource: AudioSource {
    public var onBuffer: ((AVAudioPCMBuffer) -> Void)?
    public var onEnd: ((Error?) -> Void)?
    private let url: URL
    private let tailSilence: Double
    private let realtime: Bool
    private var cancelled = false

    public init(path: String, tailSilence: Double = 6.0, realtime: Bool = false) {
        url = URL(fileURLWithPath: path)
        self.tailSilence = tailSilence
        self.realtime = realtime
    }

    public func start() async throws {
        let file = try AVAudioFile(forReading: url)
        let fmt = file.processingFormat
        let chunk: AVAudioFrameCount = 1024
        Thread.detachNewThread { [weak self] in
            guard let self else { return }
            while !self.cancelled {
                guard let b = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: chunk) else { break }
                do { try file.read(into: b, frameCount: chunk) } catch { break }
                if b.frameLength == 0 { break }
                self.onBuffer?(b)
                if self.realtime { Thread.sleep(forTimeInterval: Double(b.frameLength) / fmt.sampleRate) }
            }
            // trailing digital silence so silence detection can end the capture, like a track ending
            var remaining = Int(self.tailSilence * fmt.sampleRate)
            while !self.cancelled, remaining > 0 {
                let n = AVAudioFrameCount(min(Int(chunk), remaining))
                guard let b = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: n) else { break }
                b.frameLength = n
                if let ch = b.floatChannelData {
                    for c in 0..<Int(fmt.channelCount) { memset(ch[c], 0, Int(n) * MemoryLayout<Float>.size) }
                }
                self.onBuffer?(b)
                remaining -= Int(n)
            }
            self.onEnd?(nil)
        }
    }

    public func stop() async { cancelled = true }
}
