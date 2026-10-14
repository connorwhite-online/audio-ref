import AVFoundation
import Foundation

/// One capture from start to finished/failed. All recorder work runs on a private serial queue.
public final class CaptureSession {
    public let options: CaptureOptions
    private let onEvent: (CaptureEvent) -> Void
    private let recorder: Recorder
    private let queue = DispatchQueue(label: "audioref.capture.session")
    private var source: AudioSource?
    private var done = false
    private var lastLevelEmit: Double = -1
    private var startedAt: Date?
    private var waitTimer: DispatchSourceTimer?
    private var completion: (() -> Void)?

    public init(options: CaptureOptions, onEvent: @escaping (CaptureEvent) -> Void) {
        self.options = options
        self.onEvent = onEvent
        recorder = Recorder(options: options)
    }

    /// Runs until the capture finishes or fails.
    public func run() async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async { self.completion = { cont.resume() } }
            Task { await self.begin() }
        }
    }

    /// Stop now and keep what was recorded.
    public func requestStop() {
        queue.async { self.finish(reason: "stopped") }
    }

    private func begin() async {
        let src: AudioSource
        if let sim = options.simulateInput {
            src = FileAudioSource(path: sim)
        } else {
            src = SystemAudioSource()
        }
        src.onBuffer = { [weak self] b in self?.queue.sync { self?.handle(b) } }
        src.onEnd = { [weak self] err in self?.queue.async { self?.finish(reason: "input-ended", error: err) } }
        queue.sync { self.source = src }
        onEvent(.waiting)
        do {
            try await src.start()
        } catch {
            queue.async { self.finish(reason: "error", error: error) }
            return
        }
        // watchdog in case the source delivers no buffers at all while waiting
        if options.waitForAudio > 0 {
            let t = DispatchSource.makeTimerSource(queue: queue)
            t.schedule(deadline: .now() + options.waitForAudio + 2)
            t.setEventHandler { [weak self] in
                guard let self, self.recorder.phase == .waiting else { return }
                self.finish(reason: "no-audio")
            }
            t.resume()
            queue.async { self.waitTimer = t }
        }
    }

    private func handle(_ b: AVAudioPCMBuffer) {
        guard !done else { return }
        do {
            let step = try recorder.process(b)
            switch step {
            case .started:
                startedAt = Date()
                onEvent(.started)
            case let .stop(reason):
                finish(reason: reason)
                return
            default: break
            }
            let elapsed = recorder.recordedSeconds
            let now = Date().timeIntervalSince1970
            if now - lastLevelEmit >= 0.1 {
                lastLevelEmit = now
                onEvent(.level(db: recorder.lastLevel, elapsed: elapsed))
            }
        } catch {
            finish(reason: "error", error: error)
        }
    }

    private func finish(reason: String, error: Error? = nil) {
        guard !done else { return }
        done = true
        waitTimer?.cancel()
        let src = source
        Task { await src?.stop() }
        let duration = recorder.finish()
        if let error {
            onEvent(.failed((error as? LocalizedError)?.errorDescription ?? error.localizedDescription))
        } else if reason == "no-audio" {
            onEvent(.failed("No audio detected within \(Int(options.waitForAudio)) s. Is something playing on this Mac? " +
                            "(Audio from AudioRef itself is excluded.)"))
        } else if duration <= 0 {
            try? FileManager.default.removeItem(atPath: options.outputPath)
            onEvent(.failed("Stopped before any audio was heard."))
        } else {
            onEvent(.finished(path: options.outputPath, duration: duration, reason: reason))
        }
        completion?()
        completion = nil
    }
}
