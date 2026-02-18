import AppKit

final class SettingsWindowController: NSWindowController {
    private let settingsStore: MeshSettingsStore
    private let onSave: (MeshRuntimeSettings) -> Void

    private let modelField = NSTextField(string: "")
    private let joinField = NSTextField(string: "")
    private let apiPortField = NSTextField(string: "")
    private let consolePortField = NSTextField(string: "")
    private let startOnLoginCheckbox = NSButton(checkboxWithTitle: "Start app on login", target: nil, action: nil)
    private let infoLabel = NSTextField(labelWithString: "")
    private let titleLabel = NSTextField(labelWithString: "Mesh LLM Settings")
    private let subtitleLabel = NSTextField(labelWithString: "Configure runtime and connection options.")

    init(settingsStore: MeshSettingsStore, onSave: @escaping (MeshRuntimeSettings) -> Void) {
        self.settingsStore = settingsStore
        self.onSave = onSave

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 700, height: 430),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Mesh LLM Settings"
        window.minSize = NSSize(width: 700, height: 430)
        window.center()

        super.init(window: window)
        buildUI()
        loadValues()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func buildUI() {
        guard let content = window?.contentView else { return }

        titleLabel.font = .systemFont(ofSize: 20, weight: .semibold)
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        subtitleLabel.textColor = .secondaryLabelColor
        subtitleLabel.translatesAutoresizingMaskIntoConstraints = false

        let runtimeBox = NSBox()
        runtimeBox.title = "Runtime"
        runtimeBox.contentViewMargins = NSSize(width: 12, height: 12)
        runtimeBox.translatesAutoresizingMaskIntoConstraints = false

        let runtimeGrid = NSGridView(views: [
            [label("Model"), modelField],
            [label("Join Token"), joinField]
        ])
        runtimeGrid.rowSpacing = 10
        runtimeGrid.columnSpacing = 12
        runtimeGrid.translatesAutoresizingMaskIntoConstraints = false
        runtimeGrid.column(at: 0).xPlacement = .trailing
        runtimeGrid.column(at: 0).width = 100
        runtimeGrid.column(at: 1).xPlacement = .fill
        runtimeBox.contentView?.addSubview(runtimeGrid)

        let networkBox = NSBox()
        networkBox.title = "Network"
        networkBox.contentViewMargins = NSSize(width: 12, height: 12)
        networkBox.translatesAutoresizingMaskIntoConstraints = false

        let networkGrid = NSGridView(views: [
            [label("API Port"), apiPortField],
            [label("Console Port"), consolePortField]
        ])
        networkGrid.rowSpacing = 10
        networkGrid.columnSpacing = 12
        networkGrid.translatesAutoresizingMaskIntoConstraints = false
        networkGrid.column(at: 0).xPlacement = .trailing
        networkGrid.column(at: 0).width = 100
        networkGrid.column(at: 1).xPlacement = .fill
        networkBox.contentView?.addSubview(networkGrid)

        infoLabel.stringValue = "Save applies immediately. If mesh is running, it restarts with the new settings."
        infoLabel.textColor = .secondaryLabelColor
        infoLabel.translatesAutoresizingMaskIntoConstraints = false
        infoLabel.lineBreakMode = .byWordWrapping
        infoLabel.maximumNumberOfLines = 0

        let buttons = NSStackView()
        buttons.orientation = .horizontal
        buttons.spacing = 8
        buttons.alignment = .centerY
        buttons.translatesAutoresizingMaskIntoConstraints = false

        let saveButton = NSButton(title: "Save & Apply", target: self, action: #selector(savePressed))
        saveButton.bezelStyle = .rounded
        saveButton.keyEquivalent = "\r"
        let cancelButton = NSButton(title: "Cancel", target: self, action: #selector(cancelPressed))
        cancelButton.bezelStyle = .rounded
        buttons.addArrangedSubview(saveButton)
        buttons.addArrangedSubview(cancelButton)

        content.addSubview(titleLabel)
        content.addSubview(subtitleLabel)
        content.addSubview(runtimeBox)
        content.addSubview(networkBox)
        content.addSubview(startOnLoginCheckbox)
        content.addSubview(infoLabel)
        content.addSubview(buttons)

        NSLayoutConstraint.activate([
            titleLabel.topAnchor.constraint(equalTo: content.topAnchor, constant: 18),
            titleLabel.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),

            subtitleLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 4),
            subtitleLabel.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            subtitleLabel.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),

            runtimeBox.topAnchor.constraint(equalTo: subtitleLabel.bottomAnchor, constant: 14),
            runtimeBox.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            runtimeBox.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            runtimeBox.heightAnchor.constraint(equalToConstant: 138),

            runtimeGrid.topAnchor.constraint(equalTo: runtimeBox.contentView!.topAnchor),
            runtimeGrid.leadingAnchor.constraint(equalTo: runtimeBox.contentView!.leadingAnchor),
            runtimeGrid.trailingAnchor.constraint(equalTo: runtimeBox.contentView!.trailingAnchor),
            runtimeGrid.bottomAnchor.constraint(equalTo: runtimeBox.contentView!.bottomAnchor),

            networkBox.topAnchor.constraint(equalTo: runtimeBox.bottomAnchor, constant: 12),
            networkBox.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            networkBox.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            networkBox.heightAnchor.constraint(equalToConstant: 104),

            networkGrid.topAnchor.constraint(equalTo: networkBox.contentView!.topAnchor),
            networkGrid.leadingAnchor.constraint(equalTo: networkBox.contentView!.leadingAnchor),
            networkGrid.trailingAnchor.constraint(equalTo: networkBox.contentView!.trailingAnchor),
            networkGrid.bottomAnchor.constraint(equalTo: networkBox.contentView!.bottomAnchor),

            startOnLoginCheckbox.topAnchor.constraint(equalTo: networkBox.bottomAnchor, constant: 10),
            startOnLoginCheckbox.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),

            infoLabel.topAnchor.constraint(equalTo: startOnLoginCheckbox.bottomAnchor, constant: 8),
            infoLabel.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            infoLabel.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),

            buttons.topAnchor.constraint(equalTo: infoLabel.bottomAnchor, constant: 12),
            buttons.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            buttons.trailingAnchor.constraint(lessThanOrEqualTo: content.trailingAnchor, constant: -20),
            buttons.bottomAnchor.constraint(lessThanOrEqualTo: content.bottomAnchor, constant: -20)
        ])

        modelField.placeholderString = "Qwen2.5-32B-Instruct-Q4_K_M"
        joinField.placeholderString = "Optional invite token"
        apiPortField.placeholderString = "9337"
        consolePortField.placeholderString = "3131"
        modelField.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        joinField.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        [modelField, joinField, apiPortField, consolePortField].forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
            $0.widthAnchor.constraint(greaterThanOrEqualToConstant: 460).isActive = true
        }
        startOnLoginCheckbox.target = self
        startOnLoginCheckbox.action = #selector(startOnLoginToggled)
        startOnLoginCheckbox.translatesAutoresizingMaskIntoConstraints = false
    }

    private func label(_ text: String) -> NSTextField {
        let field = NSTextField(labelWithString: text)
        field.font = .systemFont(ofSize: 12, weight: .medium)
        return field
    }

    private func loadValues() {
        let settings = settingsStore.load()
        modelField.stringValue = settings.modelPath ?? MeshDefaults.model
        joinField.stringValue = settings.joinToken ?? ""
        apiPortField.stringValue = String(settings.apiPort)
        consolePortField.stringValue = String(settings.consolePort)
        startOnLoginCheckbox.state = settings.startOnLogin ? .on : .off
    }

    @objc private func savePressed() {
        let model = modelField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        let join = joinField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let apiPort = Int(apiPortField.stringValue), apiPort > 0,
              let consolePort = Int(consolePortField.stringValue), consolePort > 0 else {
            infoLabel.stringValue = "Ports must be positive integers."
            infoLabel.textColor = .systemRed
            return
        }

        settingsStore.save(
            modelPath: model,
            joinToken: join,
            apiPort: apiPort,
            consolePort: consolePort,
            startOnLogin: startOnLoginCheckbox.state == .on
        )
        onSave(settingsStore.load())
        window?.close()
    }

    @objc private func cancelPressed() {
        window?.close()
    }

    @objc private func startOnLoginToggled() {
        infoLabel.stringValue = "Save applies immediately. If mesh is running, it restarts with the new settings."
        infoLabel.textColor = .secondaryLabelColor
    }
}
