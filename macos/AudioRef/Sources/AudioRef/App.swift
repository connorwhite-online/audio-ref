import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // a menu bar app has no windows, so macOS would otherwise consider it idle and quit it
        ProcessInfo.processInfo.disableAutomaticTermination("AudioRef lives in the menu bar")
        ProcessInfo.processInfo.disableSuddenTermination()
        AppModel.shared.startServices()
    }
}

@main
struct AudioRefApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var model = AppModel.shared

    var body: some Scene {
        MenuBarExtra {
            PanelView()
                .environmentObject(model)
        } label: {
            MenuBarLabel()
                .environmentObject(model)
        }
        .menuBarExtraStyle(.window)
    }
}

struct MenuBarLabel: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: model.menuIcon)
            if model.phase == .recording {
                Text(formatTime(model.elapsed)).monospacedDigit()
            } else if model.phase == .analyzing {
                Text("\(Int(model.progress * 100))%").monospacedDigit()
            }
        }
    }
}
