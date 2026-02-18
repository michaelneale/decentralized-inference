import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var menuBarController: MenuBarController?
    private var settingsWindowController: SettingsWindowController?
    private var meshManager: MeshProcessManager?
    private let settingsStore = MeshSettingsStore()
    private let stateStore = AppStateStore()
    private let logService = LogService()
    private let loginItemService = LoginItemService()
    private var startupModelDisplayName: String = MeshDefaults.model

    func applicationDidFinishLaunching(_ notification: Notification) {
        logService.append("MeshLLMMenuBar launched")
        let settings = settingsStore.load()
        let runtime = makeRuntime(from: settings)
        startupModelDisplayName = runtime.modelDisplayName
        stateStore.configurationError = settings.validationError
        do {
            try loginItemService.apply(enabled: settings.startOnLogin)
        } catch {
            logService.append("Failed to apply login item setting: \(error.localizedDescription)")
        }
        self.meshManager = runtime.meshManager
        let clipboardService = ClipboardService()
        menuBarController = MenuBarController(
            stateStore: stateStore,
            meshManager: runtime.meshManager,
            tokenProvider: runtime.tokenProvider,
            clipboardService: clipboardService,
            consoleURL: runtime.consoleURL,
            openLogsHandler: { [weak self] in self?.openLogs() },
            openSettingsHandler: { [weak self] in self?.openSettingsWindow() }
        )
    }

    func applicationWillTerminate(_ notification: Notification) {
        logService.append("MeshLLMMenuBar terminating")
        try? meshManager?.stop()
    }

    private func openSettingsWindow() {
        let controller = SettingsWindowController(settingsStore: settingsStore) { [weak self] settings in
            guard let self else { return }
            self.applySettings(settings)
        }
        settingsWindowController = controller
        controller.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func openLogs() {
        logService.openInConsoleApp()
    }

    private func handleMeshOutputLine(_ line: String) {
        let isTransitioning = stateStore.meshStatus == .starting || stateStore.meshStatus == .restarting
        let lower = line.lowercased()
        if line.hasPrefix("Invite token:") {
            let token = line.replacingOccurrences(of: "Invite token:", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            if !token.isEmpty {
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.stateStore.token = token
                    self.menuBarController?.refresh()
                }
            }
        }

        if lower.contains("error:") {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.stateStore.meshStatus = .error
                self.stateStore.startupDetail = nil
                self.stateStore.lastErrorMessage = line
                self.menuBarController?.refresh()
            }
            return
        }

        guard isTransitioning else { return }

        var detail: String?
        var shouldMarkRunning = false
        if lower.contains("download") {
            detail = "Downloading \(startupModelDisplayName)..."
        } else if lower.contains("waiting for peers") || lower.contains("waiting for peer") {
            detail = "Waiting for peers..."
            shouldMarkRunning = true
        } else if lower.contains("ready") || lower.contains("llama-server ready") {
            detail = "Finishing startup..."
            shouldMarkRunning = true
        }

        guard let detail else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            if shouldMarkRunning {
                self.stateStore.meshStatus = .running
            }
            self.stateStore.startupDetail = detail
            self.menuBarController?.refresh()
        }
    }

    private func handleMeshOutputText(_ text: String) {
        _ = text
    }

    private func handleMeshProgressPercent(_ percent: String) {
        guard stateStore.meshStatus == .starting || stateStore.meshStatus == .restarting else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            let next = "Downloading \(self.startupModelDisplayName) \(percent)%"
            if self.stateStore.startupDetail != next {
                self.stateStore.startupDetail = next
                self.menuBarController?.refresh()
            }
        }
    }

    private static func displayModelName(from modelPathOrName: String?) -> String {
        guard let raw = modelPathOrName?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return MeshDefaults.model
        }
        let last = URL(fileURLWithPath: raw).lastPathComponent
        return last.isEmpty ? raw : last
    }

    private func makeRuntime(from settings: MeshRuntimeSettings) -> (meshManager: MeshProcessManager, tokenProvider: HTTPTokenProvider, consoleURL: URL, modelDisplayName: String) {
        let meshManager = MeshProcessManager(
            launchArguments: settings.launchArguments(),
            logService: logService
        )
        meshManager.onOutputLine = { [weak self] line in
            self?.handleMeshOutputLine(line)
        }
        meshManager.onOutputText = { [weak self] text in
            self?.handleMeshOutputText(text)
        }
        meshManager.onProgressPercent = { [weak self] percent in
            self?.handleMeshProgressPercent(percent)
        }

        let tokenProvider = HTTPTokenProvider(
            apiBaseURL: URL(string: "http://127.0.0.1:\(settings.apiPort)")!,
            consoleBaseURL: URL(string: "http://127.0.0.1:\(settings.consolePort)")!
        )
        return (
            meshManager,
            tokenProvider,
            URL(string: "http://127.0.0.1:\(settings.consolePort)")!,
            Self.displayModelName(from: settings.modelPath)
        )
    }

    private func applySettings(_ settings: MeshRuntimeSettings) {
        let wasRunning = stateStore.meshStatus == .running
        try? meshManager?.stop()

        let runtime = makeRuntime(from: settings)
        startupModelDisplayName = runtime.modelDisplayName
        meshManager = runtime.meshManager
        do {
            try loginItemService.apply(enabled: settings.startOnLogin)
        } catch {
            logService.append("Failed to apply login item setting: \(error.localizedDescription)")
        }
        stateStore.configurationError = settings.validationError
        stateStore.lastErrorMessage = nil
        stateStore.startupDetail = nil

        menuBarController?.updateRuntime(
            meshManager: runtime.meshManager,
            tokenProvider: runtime.tokenProvider,
            consoleURL: runtime.consoleURL
        )

        guard wasRunning, settings.validationError == nil else {
            stateStore.meshStatus = .stopped
            menuBarController?.refresh()
            return
        }

        Task { @MainActor in
            stateStore.meshStatus = .starting
            stateStore.startupDetail = "Applying settings..."
            menuBarController?.refresh()
            do {
                try runtime.meshManager.start()
                try await runtime.tokenProvider.waitForReady()
                stateStore.meshStatus = .running
                stateStore.startupDetail = nil
                do {
                    stateStore.token = try await runtime.tokenProvider.fetchToken()
                } catch {
                    stateStore.token = nil
                    stateStore.lastErrorMessage = "Mesh restarted but token unavailable: \(error.localizedDescription)"
                }
            } catch {
                stateStore.meshStatus = .error
                stateStore.startupDetail = nil
                stateStore.lastErrorMessage = "Failed to apply settings: \(error.localizedDescription)"
            }
            menuBarController?.refresh()
        }
    }
}
