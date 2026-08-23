import SwiftUI
import ARKit
import SceneKit

struct ARCameraView: UIViewRepresentable {

    @ObservedObject var manager: ARSessionManager

    func makeUIView(context: Context) -> ARSCNView {

        let view = ARSCNView(frame: .zero)

        view.session = manager.session
        view.scene = SCNScene()
        view.automaticallyUpdatesLighting = true

        return view
    }

    func updateUIView(
        _ uiView: ARSCNView,
        context: Context
    ) {
    }
}
