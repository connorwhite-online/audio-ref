import AppKit
import Foundation

let appVersion = "0.1.0"

/// ~/Library/Application Support/AudioRef
let supportDir: URL = {
    let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
    let dir = base.appendingPathComponent("AudioRef", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}()

let controlSocketPath = supportDir.appendingPathComponent("control.sock").path

/// Written by `audioref app install`: how to run the analyzer and where its cache lives.
struct AppConfig {
    var cli: [String]
    var cacheDir: String

    static func load() -> AppConfig {
        let url = supportDir.appendingPathComponent("config.json")
        var cli: [String] = []
        var cache = NSString(string: "~/.audioref").expandingTildeInPath
        if let data = try? Data(contentsOf: url),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            cli = obj["cli"] as? [String] ?? []
            if let c = obj["cache_dir"] as? String { cache = NSString(string: c).expandingTildeInPath }
        }
        if cli.isEmpty {
            for p in ["~/.local/bin/audioref", "/opt/homebrew/bin/audioref", "/usr/local/bin/audioref"] {
                let path = NSString(string: p).expandingTildeInPath
                if FileManager.default.isExecutableFile(atPath: path) { cli = [path]; break }
            }
        }
        return AppConfig(cli: cli, cacheDir: cache)
    }

    var capturesDir: String { (cacheDir as NSString).appendingPathComponent("captures") }
}

/// An analysed track in the shared cache.
struct TrackItem: Identifiable, Equatable {
    let id: String
    let name: String
    let summary: String
    let loop: String?
    let depth: String?
    let updated: Date
    let dir: URL
}

enum Library {
    static func recent(cacheDir: String, limit: Int = 8) -> [TrackItem] {
        let root = URL(fileURLWithPath: cacheDir)
        guard let entries = try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else { return [] }
        var items: [TrackItem] = []
        for dir in entries {
            let metaURL = dir.appendingPathComponent("meta.json")
            guard let data = try? Data(contentsOf: metaURL),
                  let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  (meta["status"] as? String) == "done" else { continue }
            let updated = Date(timeIntervalSince1970: (meta["updated"] as? Double) ?? (meta["created"] as? Double) ?? 0)
            items.append(TrackItem(id: dir.lastPathComponent, name: meta["name"] as? String ?? dir.lastPathComponent,
                                   summary: meta["summary"] as? String ?? "", loop: nil, depth: meta["depth"] as? String,
                                   updated: updated, dir: dir))
        }
        items.sort { $0.updated > $1.updated }
        return items.prefix(limit).map { item in
            // the chord loop is the most useful one-liner; read it from analysis.json
            var loop: String?
            if let data = try? Data(contentsOf: item.dir.appendingPathComponent("analysis.json")),
               let a = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let h = a["harmony"] as? [String: Any], let l = h["loop"] as? [String: Any],
               let syms = l["symbols"] as? [String], !syms.isEmpty {
                loop = syms.joined(separator: " – ")
            }
            return TrackItem(id: item.id, name: item.name, summary: item.summary, loop: loop, depth: item.depth,
                             updated: item.updated, dir: item.dir)
        }
    }
}

func slugify(_ s: String) -> String {
    let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_ "))
    let cleaned = String(s.unicodeScalars.filter { allowed.contains($0) })
    let words = cleaned.split(whereSeparator: { $0 == " " || $0 == "_" }).joined(separator: "-")
    return words.isEmpty ? "capture" : String(words.prefix(60))
}

func timestamp() -> String {
    let f = DateFormatter()
    f.dateFormat = "yyyyMMdd-HHmmss"
    return f.string(from: Date())
}

func formatTime(_ s: Double) -> String {
    let t = Int(s.rounded(.down))
    return String(format: "%d:%02d", t / 60, t % 60)
}
