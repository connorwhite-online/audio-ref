import AppKit
import CaptureCore
import Foundation
import ServiceManagement

/// App state. Mutated on the main thread only.
final class AppModel: ObservableObject {
    static let shared = AppModel()

    enum Phase: String { case idle, waiting, recording, analyzing, done, failed }

    @Published var phase: Phase = .idle
    @Published var message = "Ready"
    @Published var elapsed: Double = 0
    @Published var level: Double = -120
    @Published var progress: Double = 0
    @Published var name = ""
    @Published var requestedByAgent = false
    @Published var recent: [TrackItem] = []
    @Published var lastResult: TrackItem?
    @Published var permissionGranted = CapturePermission.granted

    @Published var depth: String { didSet { defaults.set(depth, forKey: "depth") } }
    @Published var analyzeAfter: Bool { didSet { defaults.set(analyzeAfter, forKey: "analyzeAfter") } }
    @Published var stopOnSilence: Double { didSet { defaults.set(stopOnSilence, forKey: "stopOnSilence") } }
    @Published var maxMinutes: Double { didSet { defaults.set(maxMinutes, forKey: "maxMinutes") } }

    let config = AppConfig.load()
    private let defaults = UserDefaults.standard
    private var session: CaptureSession?
    private var server: ControlServer?
    private var analysis: Process?

    var isBusy: Bool { [.waiting, .recording, .analyzing].contains(phase) }
    var isCapturing: Bool { phase == .waiting || phase == .recording }

    init() {
        depth = defaults.string(forKey: "depth") ?? "standard"
        analyzeAfter = defaults.object(forKey: "analyzeAfter") as? Bool ?? true
        stopOnSilence = defaults.object(forKey: "stopOnSilence") as? Double ?? 3
        maxMinutes = defaults.object(forKey: "maxMinutes") as? Double ?? 10
    }

    func startServices() {
        signal(SIGPIPE, SIG_IGN)
        let srv = ControlServer(path: controlSocketPath, model: self)
        do { try srv.start(); server = srv } catch { message = "Agent link unavailable: \(error.localizedDescription)" }
        refreshRecent()
    }

    var menuIcon: String {
        switch phase {
        case .waiting: return "ear"
        case .recording: return "record.circle.fill"
        case .analyzing: return "waveform.badge.magnifyingglass"
        case .failed: return "exclamationmark.triangle"
        default: return "waveform"
        }
    }

    // MARK: capture

    /// Start a capture from the menu (analysis runs here when enabled).
    func startCapture() {
        guard !isBusy else { return }
        let label = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let base = label.isEmpty ? "capture" : slugify(label)
        var opts = CaptureOptions(outputPath: (config.capturesDir as NSString).appendingPathComponent("\(base)-\(timestamp()).wav"))
        opts.stopOnSilence = stopOnSilence
        opts.maxSeconds = maxMinutes * 60
        opts.waitForAudio = 120
        requestedByAgent = false
        run(opts) { [weak self] event in
            guard let self else { return }
            if case let .finished(path, _, _) = event, self.analyzeAfter {
                self.analyze(path: path, name: label.isEmpty ? nil : label)
            }
        }
    }

    /// Start a capture requested over the control socket; events are forwarded to `sink`.
    func startExternalCapture(_ opts: CaptureOptions, sink: @escaping (CaptureEvent) -> Void) -> Bool {
        guard !isBusy else { return false }
        requestedByAgent = true
        run(opts, forward: sink) { [weak self] event in
            if case .finished = event {
                self?.phase = .done
                self?.message = "Captured for the agent (it runs the analysis)"
            }
        }
        return true
    }

    func stopCapture() {
        session?.requestStop()
    }

    private func run(_ opts: CaptureOptions, forward: ((CaptureEvent) -> Void)? = nil,
                     onTerminal: @escaping (CaptureEvent) -> Void) {
        phase = .waiting
        message = "Waiting for audio… press play"
        elapsed = 0
        level = -120
        let s = CaptureSession(options: opts) { [weak self] event in
            forward?(event)
            DispatchQueue.main.async { self?.apply(event, onTerminal: onTerminal) }
        }
        session = s
        Task.detached { await s.run() }
    }

    private func apply(_ event: CaptureEvent, onTerminal: (CaptureEvent) -> Void) {
        switch event {
        case .waiting:
            phase = .waiting
        case .started:
            phase = .recording
            message = requestedByAgent ? "Recording for the agent…" : "Recording…"
        case let .level(db, t):
            level = db
            elapsed = t
        case let .finished(_, duration, reason):
            session = nil
            phase = .done
            let why = reason == "silence" ? "track ended" : reason == "max-duration" ? "length limit" : "stopped"
            message = "Captured \(formatTime(duration)) (\(why))"
            permissionGranted = true
            onTerminal(event)
        case let .failed(msg):
            session = nil
            phase = .failed
            message = msg
            permissionGranted = CapturePermission.granted
            onTerminal(event)
        }
    }

    // MARK: analysis

    func analyze(path: String, name: String?) {
        guard !config.cli.isEmpty else {
            phase = .failed
            message = "audioref CLI not configured. Run `audioref app install` from the audio-ref repo."
            return
        }
        phase = .analyzing
        progress = 0
        message = "Analyzing (\(depth))…"
        let p = Process()
        p.executableURL = URL(fileURLWithPath: config.cli[0])
        var args = Array(config.cli.dropFirst()) + ["--cache", config.cacheDir, "analyze", path, "--depth", depth, "--json"]
        if let name { args += ["--name", name] }
        p.arguments = args
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + (env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin")
        p.environment = env
        let out = Pipe()
        let err = Pipe()
        p.standardOutput = out
        p.standardError = err
        var stdoutData = Data()
        var stderrTail = ""
        let lock = NSLock()
        out.fileHandleForReading.readabilityHandler = { h in
            let d = h.availableData
            lock.lock(); stdoutData.append(d); lock.unlock()
        }
        err.fileHandleForReading.readabilityHandler = { [weak self] h in
            guard let s = String(data: h.availableData, encoding: .utf8), !s.isEmpty else { return }
            lock.lock(); stderrTail = String((stderrTail + s).suffix(2000)); lock.unlock()
            // progress lines look like "\r  [ 45.0%] analysing harmony   12.3s"
            for chunk in s.components(separatedBy: CharacterSet(charactersIn: "\r\n")) {
                guard let open = chunk.firstIndex(of: "["), let pct = chunk.firstIndex(of: "%"), open < pct,
                      let value = Double(chunk[chunk.index(after: open)..<pct].trimmingCharacters(in: .whitespaces)) else { continue }
                let rest = chunk[chunk.index(after: pct)...].replacingOccurrences(of: "]", with: "")
                    .trimmingCharacters(in: .whitespaces)
                let msg = rest.split(separator: " ").filter { !$0.hasSuffix("s") || Double($0.dropLast()) == nil }.joined(separator: " ")
                DispatchQueue.main.async {
                    self?.progress = value / 100
                    if !msg.isEmpty { self?.message = msg.prefix(1).uppercased() + msg.dropFirst() + "…" }
                }
            }
        }
        p.terminationHandler = { [weak self] proc in
            out.fileHandleForReading.readabilityHandler = nil
            err.fileHandleForReading.readabilityHandler = nil
            lock.lock()
            let rest = out.fileHandleForReading.readDataToEndOfFile()
            stdoutData.append(rest)
            let data = stdoutData
            let tail = stderrTail
            lock.unlock()
            DispatchQueue.main.async { self?.analysisFinished(status: proc.terminationStatus, stdout: data, stderr: tail) }
        }
        do {
            try p.run()
            analysis = p
        } catch {
            phase = .failed
            message = "Could not start analyzer: \(error.localizedDescription)"
        }
    }

    private func analysisFinished(status: Int32, stdout: Data, stderr: String) {
        analysis = nil
        guard status == 0, let obj = try? JSONSerialization.jsonObject(with: stdout) as? [String: Any],
              let tid = obj["track_id"] as? String else {
            phase = .failed
            let last = stderr.split(separator: "\n").last.map(String.init) ?? "exit \(status)"
            message = "Analysis failed: \(last.trimmingCharacters(in: .whitespaces))"
            return
        }
        refreshRecent()
        lastResult = recent.first { $0.id == tid }
        phase = .done
        progress = 1
        message = "Done — \(lastResult?.summary ?? tid)"
        name = ""
    }

    func cancelAnalysis() {
        analysis?.terminate()
    }

    // MARK: library actions

    func refreshRecent() {
        let dir = config.cacheDir
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let items = Library.recent(cacheDir: dir)
            DispatchQueue.main.async { self?.recent = items }
        }
    }

    func openBrief(_ item: TrackItem) {
        NSWorkspace.shared.open(item.dir.appendingPathComponent("brief.md"))
    }

    func reveal(_ item: TrackItem) {
        NSWorkspace.shared.activateFileViewerSelecting([item.dir.appendingPathComponent("analysis.json")])
    }

    func copyID(_ item: TrackItem) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(item.id, forType: .string)
        message = "Copied track id \(item.id) — give it to your agent (get_brief)"
    }

    func openCache() {
        NSWorkspace.shared.open(URL(fileURLWithPath: config.cacheDir))
    }

    func requestPermission() {
        permissionGranted = CapturePermission.request()
        if !permissionGranted {
            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!)
        }
    }

    // MARK: login item

    var opensAtLogin: Bool {
        SMAppService.mainApp.status == .enabled
    }

    func setOpensAtLogin(_ on: Bool) {
        do {
            if on { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
        } catch {
            message = "Login item: \(error.localizedDescription)"
        }
        objectWillChange.send()
    }

    func statusDictionary() -> [String: Any] {
        ["ok": true, "app": "AudioRef", "version": appVersion, "phase": phase.rawValue, "message": message,
         "permission": CapturePermission.granted, "cache_dir": config.cacheDir, "captures_dir": config.capturesDir]
    }
}
