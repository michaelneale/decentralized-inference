import AppKit
import Foundation

final class LogService {
    private let fileManager: FileManager
    private let logDirectory: URL
    let logFileURL: URL

    init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
        let base = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/MeshLLMMenuBar", isDirectory: true)
        self.logDirectory = base
        self.logFileURL = base.appendingPathComponent("mesh-llm.log")
    }

    func openAppendHandle() throws -> FileHandle {
        try prepareLogFileIfNeeded()
        let handle = try FileHandle(forWritingTo: logFileURL)
        try handle.seekToEnd()
        return handle
    }

    func append(_ message: String) {
        do {
            let handle = try openAppendHandle()
            defer { try? handle.close() }
            let line = "\(timestamp()) \(message)\n"
            if let data = line.data(using: .utf8) {
                try handle.write(contentsOf: data)
            }
        } catch {
            // Best effort logging; avoid crashing app path.
        }
    }

    func openInConsoleApp() {
        do {
            try prepareLogFileIfNeeded()
        } catch {
            return
        }

        let consoleAppURL = URL(fileURLWithPath: "/System/Applications/Utilities/Console.app")
        if fileManager.fileExists(atPath: consoleAppURL.path) {
            NSWorkspace.shared.open(
                [logFileURL],
                withApplicationAt: consoleAppURL,
                configuration: NSWorkspace.OpenConfiguration()
            )
        } else {
            NSWorkspace.shared.open(logFileURL)
        }
    }

    private func prepareLogFileIfNeeded() throws {
        if !fileManager.fileExists(atPath: logDirectory.path) {
            try fileManager.createDirectory(at: logDirectory, withIntermediateDirectories: true)
        }
        if !fileManager.fileExists(atPath: logFileURL.path) {
            fileManager.createFile(atPath: logFileURL.path, contents: nil)
        }
    }

    private func timestamp() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: Date())
    }
}
