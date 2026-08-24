import SwiftUI

struct ContentView: View {

    @StateObject private var manager =
        ARSessionManager()

    var body: some View {

        ZStack {

            ARCameraView(manager: manager)
                .ignoresSafeArea()

            VStack {

                statusPanel

                Spacer()

                VStack(spacing: 10) {
                    Text(manager.captureStatus)
                        .font(.caption)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 16)

                    if manager.scanState == .recording {
                        Button("Stop Scan") {
                            manager.stopScan()
                        }
                        .buttonStyle(.borderedProminent)
                    } else if manager.scanState == .finalizing {
                        ProgressView("Finalizing...")
                    } else {
                        Button(
                            manager.scanState == .readyToTransfer
                                ? "Start New Scan"
                                : "Start Scan"
                        ) {
                            manager.startScan()
                        }
                        .buttonStyle(.borderedProminent)
                    }

                    Text("Move the phone slowly")
                        .font(.footnote)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .padding(.bottom, 30)
            }
            .padding(.top, 8)
        }
        .onAppear {
            manager.start()
        }
        .onDisappear {
            manager.pause()
        }
    }

    private var statusPanel: some View {

        VStack(alignment: .leading, spacing: 8) {

            Text("iPhone 3D Scanner")
                .font(.headline)

            Divider()

            HStack {
                Text("Scan")
                Spacer()
                Text(manager.scanState.rawValue)
            }

            HStack {
                Text("LiDAR")
                Spacer()
                Text(
                    manager.lidarSupported
                    ? "SUPPORTED"
                    : "UNAVAILABLE"
                )
            }

            HStack {
                Text("Tracking")
                Spacer()
                Text(manager.trackingStatus)
            }

            HStack {
                Text("Depth")
                Spacer()
                Text(manager.depthResolution)
            }

            HStack {
                Text("Center")

                Spacer()

                if let depth = manager.centerDepth {
                    Text(
                        String(
                            format: "%.3f m",
                            depth
                        )
                    )
                } else {
                    Text("-")
                }
            }

            HStack {
                Text("Confidence")
                Spacer()
                Text(manager.confidence)
            }
        }
        .font(.system(.body, design: .monospaced))
        .padding()
        .background(.ultraThinMaterial)
        .clipShape(
            RoundedRectangle(cornerRadius: 16)
        )
        .padding(.horizontal)
    }
}
