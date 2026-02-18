import Foundation
import Darwin

protocol MeshProcessControlling {
    func start() throws
    func stop() throws
    func restart() throws
}

enum MeshProcessError: LocalizedError {
    case invalidState(String)
    case binaryNotFound
    case failedToOpenLog(String)
    case failedToLaunch(String)

    var errorDescription: String? {
        switch self {
        case .invalidState(let message):
            return message
        case .binaryNotFound:
            return "Bundled mesh-llm binary not found"
        case .failedToOpenLog(let message):
            return "Failed to open mesh log file: \(message)"
        case .failedToLaunch(let message):
            return "Failed to launch mesh-llm: \(message)"
        }
    }
}

enum MeshBinaryLocator {
    static func findMeshBinaryPath() -> String? {
        let fileManager = FileManager.default
        if let bundled = Bundle.main.path(forResource: "mesh-llm", ofType: nil),
           fileManager.isExecutableFile(atPath: bundled) {
            return bundled
        }

        let devPath = fileManager.currentDirectoryPath + "/../mesh-llm/target/release/mesh-llm"
        if fileManager.isExecutableFile(atPath: devPath) {
            return devPath
        }
        return nil
    }
}

final class MeshProcessManager: MeshProcessControlling {
    private var process: Process?
    private var stdoutBuffer = ""
    private var stderrBuffer = ""
    private var progressBuffer = ""
    private var lastProgressPercent: String?
    private let lock = NSLock()
    private let launchArguments: [String]
    private let logService: LogService
    var onOutputLine: ((String) -> Void)?
    var onOutputText: ((String) -> Void)?
    var onProgressPercent: ((String) -> Void)?

    init(launchArguments: [String] = [], logService: LogService) {
        self.launchArguments = launchArguments
        self.logService = logService
    }

    func restart() throws {
        try stop()
        try start()
    }

    func start() throws {
        lock.lock()
        if let process, process.isRunning {
            lock.unlock()
            throw MeshProcessError.invalidState("mesh-llm is already running")
        }
        lock.unlock()

        guard let binaryPath = MeshBinaryLocator.findMeshBinaryPath() else {
            throw MeshProcessError.binaryNotFound
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: binaryPath)
        process.arguments = launchArguments

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        let argsString = launchArguments.joined(separator: " ")
        logService.append("Launching mesh-llm: \(binaryPath) \(argsString)")
        process.terminationHandler = { [weak self] terminated in
            self?.logService.append("mesh-llm exited with status \(terminated.terminationStatus)")
        }
        setupOutputHandlers(stdoutPipe: stdoutPipe, stderrPipe: stderrPipe)

        do {
            try process.run()
        } catch {
            throw MeshProcessError.failedToLaunch(error.localizedDescription)
        }

        lock.lock()
        self.process = process
        lock.unlock()
    }

    func stop() throws {
        lock.lock()
        guard let process = process else {
            lock.unlock()
            return
        }
        lock.unlock()

        if !process.isRunning {
            lock.lock()
            self.process = nil
            lock.unlock()
            return
        }

        process.terminate()
        let deadline = Date().addingTimeInterval(5)
        while process.isRunning && Date() < deadline {
            usleep(100_000)
        }

        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
            usleep(100_000)
        }
        teardownOutputHandlers(process: process)
        stdoutBuffer = ""
        stderrBuffer = ""
        progressBuffer = ""
        lastProgressPercent = nil

        lock.lock()
        self.process = nil
        lock.unlock()
    }

    private func setupOutputHandlers(stdoutPipe: Pipe, stderrPipe: Pipe) {
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.consumeOutput(from: handle, isError: false)
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.consumeOutput(from: handle, isError: true)
        }
    }

    private func teardownOutputHandlers(process: Process) {
        if let stdoutPipe = process.standardOutput as? Pipe {
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
        }
        if let stderrPipe = process.standardError as? Pipe {
            stderrPipe.fileHandleForReading.readabilityHandler = nil
        }
    }

    private func consumeOutput(from handle: FileHandle, isError: Bool) {
        let data = handle.availableData
        guard !data.isEmpty else { return }

        let text = String(data: data, encoding: .utf8) ?? ""
        onOutputText?(text)
        updateProgress(from: text)
        if isError {
            stderrBuffer.append(text)
            flushLines(from: &stderrBuffer)
        } else {
            stdoutBuffer.append(text)
            flushLines(from: &stdoutBuffer)
        }
    }

    private func updateProgress(from text: String) {
        progressBuffer.append(text)
        if progressBuffer.count > 8192 {
            progressBuffer = String(progressBuffer.suffix(2048))
        }

        let pattern = #"([0-9]{1,3}(?:\.[0-9])?)%"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return
        }
        let range = NSRange(progressBuffer.startIndex..<progressBuffer.endIndex, in: progressBuffer)
        let matches = regex.matches(in: progressBuffer, range: range)
        guard let match = matches.last, let valueRange = Range(match.range(at: 1), in: progressBuffer) else {
            return
        }

        let percent = String(progressBuffer[valueRange])
        guard percent != lastProgressPercent else { return }
        lastProgressPercent = percent
        onProgressPercent?(percent)
    }

    private func flushLines(from buffer: inout String) {
        let normalized = buffer.replacingOccurrences(of: "\r", with: "\n")
        let parts = normalized.components(separatedBy: "\n")
        if parts.isEmpty {
            return
        }

        let completed = parts.dropLast()
        buffer = parts.last ?? ""
        for rawLine in completed {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { continue }
            logService.append(line)
            onOutputLine?(line)
        }
    }
}
