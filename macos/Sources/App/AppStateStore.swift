import Foundation

enum MeshStatus: String {
    case stopped = "Stopped"
    case starting = "Starting"
    case running = "Running"
    case stopping = "Stopping"
    case restarting = "Restarting"
    case error = "Error"
}

final class AppStateStore {
    var meshStatus: MeshStatus = .stopped
    var configurationError: String?
    var startupDetail: String?
    var token: String?
    var lastErrorMessage: String?

    var canStart: Bool {
        (meshStatus == .stopped || meshStatus == .error) && configurationError == nil
    }

    var canStop: Bool {
        meshStatus == .running || meshStatus == .starting
    }

    var canRestart: Bool {
        meshStatus == .running
    }

    var canOpenConsole: Bool {
        meshStatus == .running || meshStatus == .starting
    }

    var canCopyToken: Bool {
        guard let token else { return false }
        return !token.isEmpty && (meshStatus == .running || meshStatus == .starting)
    }
}
