import Foundation

protocol TokenProviding {
    func waitForReady() async throws
    func fetchToken() async throws -> String
}

enum TokenProviderError: LocalizedError {
    case healthTimeout
    case invalidStatusPayload
    case missingToken

    var errorDescription: String? {
        switch self {
        case .healthTimeout:
            return "mesh-llm is still starting (model download/load can take a long time)"
        case .invalidStatusPayload:
            return "mesh-llm console returned invalid status payload"
        case .missingToken:
            return "invite token not present in mesh-llm status"
        }
    }
}

private struct ConsoleStatusResponse: Decodable {
    let token: String?
}

final class HTTPTokenProvider: TokenProviding {
    private let apiBaseURL: URL
    private let consoleBaseURL: URL
    private let session: URLSession
    private let timeoutSeconds: Int

    init(
        apiBaseURL: URL = URL(string: "http://127.0.0.1:9337")!,
        consoleBaseURL: URL = URL(string: "http://127.0.0.1:3131")!,
        timeoutSeconds: Int = 3600,
        session: URLSession = .shared
    ) {
        self.apiBaseURL = apiBaseURL
        self.consoleBaseURL = consoleBaseURL
        self.timeoutSeconds = timeoutSeconds
        self.session = session
    }

    func fetchToken() async throws -> String {
        return try await fetchInviteToken()
    }

    func waitForReady() async throws {
        try await waitForHealth()
    }

    private func waitForHealth() async throws {
        let deadline = Date().addingTimeInterval(TimeInterval(timeoutSeconds))
        let healthURL = apiBaseURL.appendingPathComponent("health")
        while Date() < deadline {
            var request = URLRequest(url: healthURL)
            request.timeoutInterval = 2
            do {
                let (_, response) = try await session.data(for: request)
                if let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) {
                    return
                }
            } catch {
                // mesh startup can take time; keep polling until timeout.
            }
            try await Task.sleep(nanoseconds: 1_000_000_000)
        }
        throw TokenProviderError.healthTimeout
    }

    private func fetchInviteToken() async throws -> String {
        let statusURL = consoleBaseURL
            .appendingPathComponent("api")
            .appendingPathComponent("status")
        var request = URLRequest(url: statusURL)
        request.timeoutInterval = 5
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw TokenProviderError.invalidStatusPayload
        }
        let decoded = try JSONDecoder().decode(ConsoleStatusResponse.self, from: data)
        guard let token = decoded.token?.trimmingCharacters(in: .whitespacesAndNewlines), !token.isEmpty else {
            throw TokenProviderError.missingToken
        }
        return token
    }
}
