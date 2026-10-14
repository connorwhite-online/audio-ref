import AppKit
import SwiftUI

struct PanelView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            if !model.permissionGranted && !model.isCapturing {
                permissionCard
            }
            captureCard
            if !model.recent.isEmpty {
                recentList
            }
            Divider()
            footer
        }
        .padding(14)
        .frame(width: 360)
        .onAppear { model.refreshRecent() }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "waveform")
                .font(.title2)
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 1) {
                Text("AudioRef").font(.headline)
                Text("Play a reference, get a producer's brief").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            StatusDot(phase: model.phase)
        }
    }

    private var permissionCard: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "lock.shield").foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 6) {
                Text("Needs permission to hear system audio").font(.subheadline.weight(.semibold))
                Text("macOS calls this \u{201C}Screen & System Audio Recording\u{201D}. AudioRef only records sound, never the screen.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Button("Grant access…") { model.requestPermission() }.controlSize(.small)
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.orange.opacity(0.08)))
    }

    private var captureCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            TextField("Track name (optional)", text: $model.name)
                .textFieldStyle(.roundedBorder)
                .disabled(model.isBusy)
            Picker("Depth", selection: $model.depth) {
                Text("Quick").tag("quick")
                Text("Standard").tag("standard")
                Text("Deep").tag("deep")
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .disabled(model.isBusy)
            Toggle("Analyze when the capture ends", isOn: $model.analyzeAfter)
                .toggleStyle(.checkbox)
                .font(.callout)
                .disabled(model.isBusy)

            if model.isCapturing {
                LevelMeter(db: model.level, active: model.phase == .recording)
                    .frame(height: 8)
            }
            if model.phase == .analyzing {
                ProgressView(value: model.progress).controlSize(.small)
            }

            HStack(alignment: .center, spacing: 10) {
                Text(model.message)
                    .font(.callout)
                    .foregroundStyle(model.phase == .failed ? Color.red : Color.secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 4)
                primaryButton
            }
            if model.phase == .idle || model.phase == .done || model.phase == .failed {
                Text("Press Capture, then play a song anywhere on this Mac. Recording starts with the sound and stops after \(Int(model.stopOnSilence)) s of silence.")
                    .font(.caption).foregroundStyle(.tertiary).fixedSize(horizontal: false, vertical: true)
            }
            if let r = model.lastResult, model.phase == .done {
                TrackRow(item: r, highlight: true)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.04)))
    }

    @ViewBuilder private var primaryButton: some View {
        switch model.phase {
        case .waiting, .recording:
            Button { model.stopCapture() } label: { Label("Stop", systemImage: "stop.fill") }
                .buttonStyle(.borderedProminent).tint(.red)
        case .analyzing:
            Button("Cancel") { model.cancelAnalysis() }.buttonStyle(.bordered)
        default:
            Button { model.startCapture() } label: { Label("Capture", systemImage: "record.circle") }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
        }
    }

    private var recentList: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Recent").font(.subheadline.weight(.semibold)).foregroundStyle(.secondary)
            ForEach(model.recent.prefix(5)) { item in
                TrackRow(item: item, highlight: false)
            }
        }
    }

    private var footer: some View {
        HStack {
            Menu {
                Picker("Stop after silence", selection: $model.stopOnSilence) {
                    Text("2 seconds").tag(2.0)
                    Text("3 seconds").tag(3.0)
                    Text("5 seconds").tag(5.0)
                    Text("10 seconds").tag(10.0)
                    Text("Never (stop manually)").tag(0.0)
                }
                Picker("Maximum length", selection: $model.maxMinutes) {
                    Text("5 minutes").tag(5.0)
                    Text("10 minutes").tag(10.0)
                    Text("20 minutes").tag(20.0)
                }
                Divider()
                Toggle("Open at login", isOn: Binding(get: { model.opensAtLogin }, set: { model.setOpensAtLogin($0) }))
                Button("Open analysis folder") { model.openCache() }
            } label: {
                Image(systemName: "gearshape")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            Text("Agents: \u{201C}capture_audio\u{201D} MCP tool").font(.caption2).foregroundStyle(.tertiary)
            Spacer()
            Button("Quit") { NSApp.terminate(nil) }
                .buttonStyle(.borderless)
                .font(.callout)
        }
    }
}

struct TrackRow: View {
    @EnvironmentObject var model: AppModel
    let item: TrackItem
    let highlight: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "music.note")
                .foregroundStyle(highlight ? Color.accentColor : Color.secondary)
                .frame(width: 14)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.name).font(.callout.weight(.medium)).lineLimit(1)
                if !item.summary.isEmpty {
                    Text(item.summary).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                if let loop = item.loop {
                    Text(loop).font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(1)
                }
            }
            Spacer(minLength: 4)
            HStack(spacing: 2) {
                IconButton(symbol: "doc.text", help: "Open brief") { model.openBrief(item) }
                IconButton(symbol: "doc.on.doc", help: "Copy track id for your agent") { model.copyID(item) }
                IconButton(symbol: "folder", help: "Show files") { model.reveal(item) }
            }
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 6)
        .background(RoundedRectangle(cornerRadius: 6).fill(highlight ? Color.accentColor.opacity(0.08) : Color.clear))
    }
}

struct IconButton: View {
    let symbol: String
    let help: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: symbol).frame(width: 20, height: 18)
        }
        .buttonStyle(.borderless)
        .help(help)
    }
}

struct LevelMeter: View {
    let db: Double
    let active: Bool

    var body: some View {
        GeometryReader { geo in
            let norm = max(0, min(1, (db + 60) / 60))
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.08))
                Capsule()
                    .fill(LinearGradient(colors: [.green, .yellow, .orange], startPoint: .leading, endPoint: .trailing))
                    .frame(width: geo.size.width * norm)
                    .opacity(active ? 1 : 0.5)
                    .animation(.linear(duration: 0.1), value: norm)
            }
        }
    }
}

struct StatusDot: View {
    let phase: AppModel.Phase

    var body: some View {
        let (color, text): (Color, String) = {
            switch phase {
            case .idle: return (.secondary, "Ready")
            case .waiting: return (.orange, "Listening")
            case .recording: return (.red, "Recording")
            case .analyzing: return (.blue, "Analyzing")
            case .done: return (.green, "Done")
            case .failed: return (.red, "Error")
            }
        }()
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 7, height: 7)
            Text(text).font(.caption).foregroundStyle(.secondary)
        }
    }
}
