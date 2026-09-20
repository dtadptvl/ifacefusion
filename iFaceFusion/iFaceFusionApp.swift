import SwiftUI

@main
struct iFaceFusionApp: App {
    @StateObject private var store = FusionStore()
    var body: some Scene {
        WindowGroup { FusionView().environmentObject(store) }
    }
}