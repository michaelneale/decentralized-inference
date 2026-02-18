import Foundation
import ServiceManagement

final class LoginItemService {
    func apply(enabled: Bool) throws {
        guard #available(macOS 13.0, *) else {
            return
        }
        if enabled {
            try SMAppService.mainApp.register()
        } else {
            try SMAppService.mainApp.unregister()
        }
    }
}
