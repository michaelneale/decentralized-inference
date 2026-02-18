import Foundation

enum MeshSettingsKeys {
    static let modelPath = "mesh.modelPath"
    static let joinToken = "mesh.joinToken"
    static let apiPort = "mesh.apiPort"
    static let consolePort = "mesh.consolePort"
    static let startOnLogin = "mesh.startOnLogin"
}

enum MeshDefaults {
    static let model = "Qwen2.5-32B-Instruct-Q4_K_M"
    static let apiPort = 9337
    static let consolePort = 3131
}

struct MeshRuntimeSettings {
    let modelPath: String?
    let joinToken: String?
    let apiPort: Int
    let consolePort: Int
    let startOnLogin: Bool

    var validationError: String? {
        let hasModel = !(modelPath?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
        let hasJoin = !(joinToken?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
        if hasModel || hasJoin {
            return nil
        }
        return "Configure model path or join token before starting mesh"
    }

    func launchArguments() -> [String] {
        var args: [String] = [
            "--port", String(apiPort),
            "--console", String(consolePort)
        ]
        if let modelPath, !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args.append(contentsOf: ["--model", modelPath])
        }
        if let joinToken, !joinToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args.append(contentsOf: ["--join", joinToken])
        }
        return args
    }
}

final class MeshSettingsStore {
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    func load() -> MeshRuntimeSettings {
        let modelPath = env("MESH_LLM_MODEL_PATH")
            ?? defaults.string(forKey: MeshSettingsKeys.modelPath)
            ?? MeshDefaults.model
        let joinToken = env("MESH_LLM_JOIN_TOKEN") ?? defaults.string(forKey: MeshSettingsKeys.joinToken)
        let apiPort = intEnv("MESH_LLM_API_PORT")
            ?? defaults.integer(forKey: MeshSettingsKeys.apiPort).nonZeroOrNil
            ?? MeshDefaults.apiPort
        let consolePort = intEnv("MESH_LLM_CONSOLE_PORT")
            ?? defaults.integer(forKey: MeshSettingsKeys.consolePort).nonZeroOrNil
            ?? MeshDefaults.consolePort

        return MeshRuntimeSettings(
            modelPath: modelPath,
            joinToken: joinToken,
            apiPort: apiPort,
            consolePort: consolePort,
            startOnLogin: boolEnv("MESH_LLM_START_ON_LOGIN")
                ?? defaults.object(forKey: MeshSettingsKeys.startOnLogin) as? Bool
                ?? true
        )
    }

    func save(modelPath: String, joinToken: String, apiPort: Int, consolePort: Int, startOnLogin: Bool) {
        defaults.set(modelPath, forKey: MeshSettingsKeys.modelPath)
        defaults.set(joinToken, forKey: MeshSettingsKeys.joinToken)
        defaults.set(apiPort, forKey: MeshSettingsKeys.apiPort)
        defaults.set(consolePort, forKey: MeshSettingsKeys.consolePort)
        defaults.set(startOnLogin, forKey: MeshSettingsKeys.startOnLogin)
    }

    private func env(_ key: String) -> String? {
        let value = ProcessInfo.processInfo.environment[key]
        if let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return value
        }
        return nil
    }

    private func intEnv(_ key: String) -> Int? {
        guard let raw = env(key), let value = Int(raw), value > 0 else {
            return nil
        }
        return value
    }

    private func boolEnv(_ key: String) -> Bool? {
        guard let raw = env(key)?.lowercased() else { return nil }
        switch raw {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            return nil
        }
    }
}

private extension Int {
    var nonZeroOrNil: Int? {
        self > 0 ? self : nil
    }
}
