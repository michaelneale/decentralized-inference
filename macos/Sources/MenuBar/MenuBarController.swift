import AppKit
import Foundation

final class MenuBarController: NSObject, NSMenuDelegate, NSMenuItemValidation {
    private let stateStore: AppStateStore
    private var meshManager: MeshProcessControlling
    private var tokenProvider: TokenProviding
    private let clipboardService: ClipboardService
    private var consoleURL: URL
    private let openLogsHandler: () -> Void
    private let openSettingsHandler: () -> Void

    private let statusItem: NSStatusItem
    private let menu: NSMenu
    private let baseStatusIcon: NSImage?
    private let grayStatusIcon: NSImage?
    private let whiteStatusIcon: NSImage?
    private let redStatusIcon: NSImage?
    private let blueStatusIcon: NSImage?

    private let statusMenuItem = NSMenuItem(title: "", action: nil, keyEquivalent: "")
    private lazy var startMenuItem = NSMenuItem(title: "Start Mesh", action: #selector(startMesh), keyEquivalent: "s")
    private lazy var stopMenuItem = NSMenuItem(title: "Stop Mesh", action: #selector(stopMesh), keyEquivalent: "x")
    private lazy var restartMenuItem = NSMenuItem(title: "Restart Mesh", action: #selector(restartMesh), keyEquivalent: "r")
    private lazy var copyTokenMenuItem = NSMenuItem(title: "Copy Token", action: #selector(copyToken), keyEquivalent: "c")
    private lazy var openConsoleMenuItem = NSMenuItem(title: "Open Console", action: #selector(openConsole), keyEquivalent: "o")
    private lazy var openLogsMenuItem = NSMenuItem(title: "Open Logs", action: #selector(openLogs), keyEquivalent: "l")
    private lazy var settingsMenuItem = NSMenuItem(title: "Settings...", action: #selector(openSettings), keyEquivalent: ",")
    private lazy var errorMenuItem = NSMenuItem(title: "", action: nil, keyEquivalent: "")
    private lazy var quitMenuItem = NSMenuItem(title: "Quit", action: #selector(quit), keyEquivalent: "q")

    init(
        stateStore: AppStateStore,
        meshManager: MeshProcessControlling,
        tokenProvider: TokenProviding,
        clipboardService: ClipboardService,
        consoleURL: URL,
        openLogsHandler: @escaping () -> Void,
        openSettingsHandler: @escaping () -> Void
    ) {
        self.stateStore = stateStore
        self.meshManager = meshManager
        self.tokenProvider = tokenProvider
        self.clipboardService = clipboardService
        self.consoleURL = consoleURL
        self.openLogsHandler = openLogsHandler
        self.openSettingsHandler = openSettingsHandler
        self.baseStatusIcon = Self.loadIcon(named: "MeshLLM")
        self.grayStatusIcon = Self.loadIcon(named: "MeshLLM-gray")
        self.whiteStatusIcon = Self.loadIcon(named: "MeshLLM-white")
        self.redStatusIcon = Self.loadIcon(named: "MeshLLM-red")
        self.blueStatusIcon = Self.loadIcon(named: "MeshLLM-blue")
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        menu = NSMenu()
        super.init()
        buildMenu()
        refreshMenuState()
    }

    private func buildMenu() {
        if let button = statusItem.button {
            if let icon = baseStatusIcon {
                button.image = icon
                button.imagePosition = .imageOnly
                button.toolTip = "Mesh LLM"
            } else {
                button.title = "Mesh LLM"
            }
        }
        statusItem.menu = menu
        menu.delegate = self

        startMenuItem.target = self
        stopMenuItem.target = self
        restartMenuItem.target = self
        copyTokenMenuItem.target = self
        openConsoleMenuItem.target = self
        openLogsMenuItem.target = self
        settingsMenuItem.target = self
        quitMenuItem.target = self

        errorMenuItem.isHidden = true

        menu.addItem(statusMenuItem)
        menu.addItem(.separator())
        menu.addItem(startMenuItem)
        menu.addItem(stopMenuItem)
        menu.addItem(restartMenuItem)
        menu.addItem(.separator())
        menu.addItem(copyTokenMenuItem)
        menu.addItem(openConsoleMenuItem)
        menu.addItem(openLogsMenuItem)
        menu.addItem(.separator())
        menu.addItem(settingsMenuItem)
        menu.addItem(errorMenuItem)
        menu.addItem(.separator())
        menu.addItem(quitMenuItem)
    }

    private func refreshMenuState() {
        switch stateStore.meshStatus {
        case .starting, .restarting:
            if let detail = stateStore.startupDetail, !detail.isEmpty {
                statusMenuItem.title = "\(stateStore.meshStatus.rawValue) (\(detail))"
            } else {
                statusMenuItem.title = "\(stateStore.meshStatus.rawValue) (first run may take a while)"
            }
        default:
            statusMenuItem.title = stateStore.meshStatus.rawValue
        }
        statusItem.button?.toolTip = statusMenuItem.title
        updateStatusIcon()

        let errorText = stateStore.lastErrorMessage ?? stateStore.configurationError
        if let message = errorText, !message.isEmpty {
            errorMenuItem.title = "Error: \(message)"
            errorMenuItem.isHidden = false
        } else {
            errorMenuItem.title = ""
            errorMenuItem.isHidden = true
        }
    }

    func refresh() {
        refreshMenuState()
    }

    func menuWillOpen(_ menu: NSMenu) {
        refreshMenuState()
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        refreshMenuState()
    }

    func validateMenuItem(_ menuItem: NSMenuItem) -> Bool {
        guard let action = menuItem.action else { return menuItem.isEnabled }
        switch action {
        case #selector(startMesh):
            return stateStore.canStart
        case #selector(stopMesh):
            return stateStore.canStop
        case #selector(restartMesh):
            return stateStore.canRestart
        case #selector(copyToken):
            return stateStore.canCopyToken
        case #selector(openConsole):
            return stateStore.canOpenConsole
        case #selector(openLogs), #selector(openSettings), #selector(quit):
            return true
        default:
            return menuItem.isEnabled
        }
    }

    func updateRuntime(
        meshManager: MeshProcessControlling,
        tokenProvider: TokenProviding,
        consoleURL: URL
    ) {
        self.meshManager = meshManager
        self.tokenProvider = tokenProvider
        self.consoleURL = consoleURL
        refreshMenuState()
    }

    private func updateStatusIcon() {
        guard let button = statusItem.button else { return }
        switch stateStore.meshStatus {
        case .running:
            if let icon = whiteStatusIcon ?? baseStatusIcon {
                button.image = icon
                return
            }
        case .error:
            if let icon = redStatusIcon ?? baseStatusIcon {
                button.image = icon
                return
            }
        case .starting, .stopping, .restarting:
            if let icon = blueStatusIcon ?? baseStatusIcon {
                button.image = icon
                return
            }
        case .stopped:
            if let icon = grayStatusIcon ?? baseStatusIcon {
                button.image = icon
                return
            }
        }
        guard let base = baseStatusIcon else { return }
        let fallbackColor: NSColor
        switch stateStore.meshStatus {
        case .running: fallbackColor = .white
        case .error: fallbackColor = .systemRed
        case .starting, .stopping, .restarting: fallbackColor = .systemBlue
        case .stopped: fallbackColor = .systemGray
        }
        button.image = makeBadgedIcon(base: base, badgeColor: fallbackColor)
        button.imagePosition = .imageOnly
    }

    private func makeBadgedIcon(base: NSImage, badgeColor: NSColor) -> NSImage {
        let size = base.size
        let image = NSImage(size: size)
        image.lockFocus()
        base.draw(in: NSRect(origin: .zero, size: size))

        let badgeDiameter: CGFloat = max(5, size.width * 0.34)
        let x = size.width - badgeDiameter - 1
        let y: CGFloat = 1
        let badgeRect = NSRect(x: x, y: y, width: badgeDiameter, height: badgeDiameter)

        NSColor.black.withAlphaComponent(0.45).setFill()
        NSBezierPath(ovalIn: badgeRect.insetBy(dx: -0.8, dy: -0.8)).fill()
        badgeColor.setFill()
        NSBezierPath(ovalIn: badgeRect).fill()

        image.unlockFocus()
        image.isTemplate = false
        return image
    }

    private static func loadIcon(named name: String) -> NSImage? {
        for ext in ["icns", "png"] {
            if let path = Bundle.main.path(forResource: name, ofType: ext),
               let icon = NSImage(contentsOfFile: path) {
                icon.size = NSSize(width: 18, height: 18)
                icon.isTemplate = false
                return icon
            }
        }

        for ext in ["icns", "png"] {
            let devPath = FileManager.default.currentDirectoryPath + "/Sources/Resources/\(name).\(ext)"
            if FileManager.default.fileExists(atPath: devPath),
               let icon = NSImage(contentsOfFile: devPath) {
                icon.size = NSSize(width: 18, height: 18)
                icon.isTemplate = false
                return icon
            }
        }
        return nil
    }

    @objc private func startMesh() {
        Task { await runStartFlow() }
    }

    @objc private func stopMesh() {
        Task { await runStopFlow() }
    }

    @objc private func restartMesh() {
        Task { await runRestartFlow() }
    }

    @objc private func copyToken() {
        guard let token = stateStore.token, !token.isEmpty else { return }
        clipboardService.copy(token)
    }

    @objc private func openConsole() {
        NSWorkspace.shared.open(consoleURL)
    }

    @objc private func openLogs() {
        openLogsHandler()
    }

    @objc private func openSettings() {
        openSettingsHandler()
    }

    @objc private func quit() {
        NSApplication.shared.terminate(nil)
    }

    private func runStartFlow() async {
        stateStore.meshStatus = .starting
        stateStore.startupDetail = "waiting for health"
        stateStore.lastErrorMessage = nil
        refreshMenuState()

        do {
            try meshManager.start()
            try await tokenProvider.waitForReady()
            stateStore.meshStatus = .running
            stateStore.startupDetail = nil

            do {
                stateStore.token = try await tokenProvider.fetchToken()
            } catch {
                stateStore.token = nil
                stateStore.lastErrorMessage = "Mesh started but token unavailable: \(error.localizedDescription)"
            }
        } catch {
            stateStore.meshStatus = .error
            stateStore.lastErrorMessage = "Failed to start mesh: \(error.localizedDescription)"
            stateStore.startupDetail = nil
        }

        refreshMenuState()
    }

    private func runStopFlow() async {
        stateStore.meshStatus = .stopping
        stateStore.startupDetail = nil
        stateStore.lastErrorMessage = nil
        refreshMenuState()

        do {
            try meshManager.stop()
            stateStore.meshStatus = .stopped
            stateStore.token = nil
        } catch {
            stateStore.meshStatus = .error
            stateStore.lastErrorMessage = "Failed to stop mesh: \(error.localizedDescription)"
        }

        refreshMenuState()
    }

    private func runRestartFlow() async {
        stateStore.meshStatus = .restarting
        stateStore.startupDetail = "waiting for health"
        stateStore.lastErrorMessage = nil
        refreshMenuState()

        do {
            try meshManager.restart()
            try await tokenProvider.waitForReady()
            stateStore.meshStatus = .running
            stateStore.startupDetail = nil

            do {
                stateStore.token = try await tokenProvider.fetchToken()
            } catch {
                stateStore.token = nil
                stateStore.lastErrorMessage = "Mesh restarted but token unavailable: \(error.localizedDescription)"
            }
        } catch {
            stateStore.meshStatus = .error
            stateStore.lastErrorMessage = "Failed to restart mesh: \(error.localizedDescription)"
            stateStore.startupDetail = nil
        }

        refreshMenuState()
    }
}
