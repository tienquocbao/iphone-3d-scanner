import SwiftUI

struct ContentView: View {

    @StateObject private var manager =
        ARSessionManager()
    @State private var showingDeleteConfirmation = false
    @Environment(\.scenePhase) private var scenePhase

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

                    if manager.scanState == .recording || manager.scanState == .completed {
                        HStack(spacing: 18) {
                            Text("Frames \(manager.capturedFrameCount)")
                            Text("Duration \(manager.durationText)")
                            Text("Storage \(manager.storageText)")
                        }
                        .font(.caption.monospaced())
                    }

                    if manager.scanState == .recording {
                        Button("Stop Scan") {
                            manager.stopScan()
                        }
                        .buttonStyle(.borderedProminent)
                    } else if manager.scanState == .finalizing {
                        ProgressView("Finalizing...")
                    } else if manager.scanState == .completed {
                        TextField("Windows receiver URL", text: $manager.serverURLText)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .textFieldStyle(.roundedBorder)
                        SecureField("Bearer token (Keychain)", text: $manager.authTokenText)
                            .textFieldStyle(.roundedBorder)
                        HStack {
                            Button("Test Connection") {
                                manager.testConnection()
                            }
                            .buttonStyle(.bordered)
                            .disabled(manager.isTransferring)
                            Button(manager.isTransferring ? "Uploading..." : "Upload") {
                                manager.transferCompletedSession()
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(manager.isTransferring)
                            Button("Delete Session", role: .destructive) {
                                showingDeleteConfirmation = true
                            }
                            .buttonStyle(.bordered)
                            .disabled(manager.isTransferring)
                        }
                        if manager.isTransferring && !manager.uploadProgressText.isEmpty {
                            Text(manager.uploadProgressText)
                                .font(.caption.monospaced())
                        }
                        Text("LAN or HTTPS receiver. Local session is deleted only after VERIFIED ACK.")
                            .font(.caption2)
                            .multilineTextAlignment(.center)
                    } else {
                        Button("Start Scan") {
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
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .background {
                manager.pause()
            }
        }
        .alert("Delete completed session?", isPresented: $showingDeleteConfirmation) {
            Button("Delete", role: .destructive) {
                manager.deleteCompletedSession()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes only the current completed session from the iPhone.")
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
